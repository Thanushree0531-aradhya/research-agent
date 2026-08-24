"""
Research Scientist Agent — core LangGraph pipeline.

Pipeline:
    Plan -> Retrieve -> Analyze -> Verify -> (loop back to Retrieve if
    evidence is insufficient) -> Finalize

Each node is a small, inspectable function. The graph is what makes this
an *agent* rather than a single prompt: the LLM's own verification step
can send the process back to search for more evidence before answering.

Paper mode
----------
The agent detects three different "I want papers" intents in the
question:

  * SPECIFIC  - a particular paper is named ("...the paper called X...",
                "...the 'X' paper...") -> search for that exact paper and
                answer only about it.
  * COUNT     - a specific number of papers is requested ("...from 2
                research papers...") -> retrieve and cite exactly that
                many genuine academic papers.
  * BROAD     - papers are mentioned but no number is given ("...based on
                research papers...") -> retrieve a wide candidate pool
                (up to ~20) of genuine academic papers.

In all three cases retrieval is restricted to academic sources (arXiv,
Semantic Scholar, ResearchGate, ACM, IEEE) and duplicate papers (same
URL, or same paper indexed under a different URL) are filtered out
before they ever reach the model.

Topic locking
-------------
The main proper-noun/topic of the original question (e.g. "LangChain")
is extracted once up front and re-injected into every query generated
during planning *and* during the verification retry loop, so later
iterations can't drift onto a different topic.

RAG (Chroma)
------------
Alongside live web/arXiv search, retrieve_node also queries a local
Chroma vector store (vector_store.py) for semantically similar content
gathered from *previous* runs. analyze_node writes newly gathered
sources back into Chroma after synthesizing an answer, so the knowledge
base grows over time and future queries can hit cached, relevant
evidence instantly instead of only relying on live search.

Claim-level verification
-------------------------
verify_node splits the draft answer into individual factual claims (each
tied to its [n] citation) and checks each one against the specific
source it cites, rather than passing/failing the answer as a whole. This
gives a support_ratio (fraction of claims that hold up) instead of a
single true/false, and finalize_node uses it to strip any claim that
didn't hold up before the answer is scored by RAGAS.

verify_node only shows the verifier LLM the sources actually cited in
the draft (falling back to a reranked top-N when the draft has no
citations at all), rather than the full retrieved pool — otherwise a
large source pool (20-60+ sources) blows past small models' per-minute
token limits on Groq and causes 413 "Request too large" errors.

JSON robustness (verify + plan)
--------------------------------
Both plan_node and verify_node ask the LLM to return JSON. Three layers
guard against malformed output instead of one brittle json.loads call:
  1. `response_format: json_object` is set on the underlying Groq calls
     for both models, which materially reduces malformed JSON.
  2. `_extract_json` falls back to `json_repair` (if installed) when a
     strict parse fails, fixing common issues like unescaped quotes or
     a truncated trailing object.
  3. verify_node retries the LLM call itself (cheap) up to once more on
     a parse failure, instead of forcing the whole retrieve->analyze->
     verify loop to re-run just to get another shot at valid JSON.
"""

import os
import json
import re
import time
from collections import Counter
from vector_store import query_chroma, add_to_chroma
from evaluation import evaluate_answer
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import TypedDict, List, Dict, Optional

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from tools import web_search, SearchResult, search_arxiv_structured

try:
    from flashrank import Ranker, RerankRequest
    _FLASHRANK_AVAILABLE = True
except ImportError:
    _FLASHRANK_AVAILABLE = False

try:
    from json_repair import repair_json
    _JSON_REPAIR_AVAILABLE = True
except ImportError:
    _JSON_REPAIR_AVAILABLE = False

MODEL_NAME = os.environ.get("RESEARCH_AGENT_MODEL", "llama-3.3-70b-versatile")
# How many concurrent search calls to run at once during retrieval, and how
# long to wait on any single search before giving up on it.
SEARCH_MAX_WORKERS = int(os.environ.get("RESEARCH_AGENT_SEARCH_MAX_WORKERS", "8"))
SEARCH_TIMEOUT_SECONDS = int(os.environ.get("RESEARCH_AGENT_SEARCH_TIMEOUT_SECONDS", "15"))
# plan_node and verify_node only need to produce structured JSON, not deep
# reasoning — using a smaller/cheaper model for them roughly halves total
# token usage per run against the strong model's daily cap.
PLANNING_MODEL_NAME = os.environ.get("RESEARCH_AGENT_PLANNING_MODEL", "llama-3.1-8b-instant")
# If the strong model hits a *daily* token quota (not a short transient
# limit), waiting won't help within a reasonable session — fall back to
# this model instead of blocking for minutes.
FALLBACK_MODEL_NAME = os.environ.get("RESEARCH_AGENT_FALLBACK_MODEL", "llama-3.1-8b-instant")
MAX_ITERATIONS = int(os.environ.get("RESEARCH_AGENT_MAX_ITERATIONS", "2"))
LLM_MAX_RETRIES = int(os.environ.get("RESEARCH_AGENT_LLM_MAX_RETRIES", "3"))
# How many times verify_node will retry ONLY the verification LLM call
# (cheap) if the response fails to parse as JSON, before giving up and
# treating the result as unparseable. This is intentionally separate
# from LLM_MAX_RETRIES (which handles 429s) — this handles malformed
# JSON on an otherwise-successful response.
VERIFY_PARSE_RETRIES = int(os.environ.get("RESEARCH_AGENT_VERIFY_PARSE_RETRIES", "2"))
# Cap on how many sources are actually shown to the LLM during synthesis
# (analyze_node), independent of how many were retrieved. Retrieval can
# gather 20-30+ candidates across web/arxiv/chroma, but handing all of
# them to the LLM at once dilutes grounding — it starts blending claims
# across weak, tangential sources instead of sticking to the strongest
# ones, which directly hurts RAGAS faithfulness/relevancy. Only applied
# outside paper mode; count/specific/broad paper modes already have their
# own tighter pool caps in retrieve_node and need the full set to find
# the right paper(s).
MAX_SYNTHESIS_SOURCES = int(os.environ.get("RESEARCH_AGENT_MAX_SYNTHESIS_SOURCES", "8"))
# Cross-encoder reranker model used to pick the MAX_SYNTHESIS_SOURCES best
# sources out of the full retrieved pool, instead of a naive front-slice.
# ms-marco-MiniLM-L-12-v2 is a small (~130MB) model that flashrank
# downloads and caches locally on first use.
RERANKER_MODEL_NAME = os.environ.get("RESEARCH_AGENT_RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")

_reranker_instance = None


def _get_reranker():
    """Lazily construct the reranker once per process (loading the model
    is the expensive part, not calling it) and reuse it across requests."""
    global _reranker_instance
    if _reranker_instance is None and _FLASHRANK_AVAILABLE:
        _reranker_instance = Ranker(model_name=RERANKER_MODEL_NAME, cache_dir="/tmp/flashrank_cache")
    return _reranker_instance
# Never silently block longer than this on a single retry wait.
MAX_SINGLE_WAIT_SECONDS = int(os.environ.get("RESEARCH_AGENT_MAX_WAIT_SECONDS", "90"))
HEARTBEAT_INTERVAL_SECONDS = 15

# Minimum fraction of claims that must be individually supported by their
# cited source for the draft to be considered "verified". Below this, the
# graph loops back to retrieval (if iterations remain) instead of
# finalizing on weak evidence.
MIN_SUPPORT_RATIO = float(os.environ.get("RESEARCH_AGENT_MIN_SUPPORT_RATIO", "0.8"))

# response_format=json_object forces the model to emit syntactically
# valid JSON (Groq/OpenAI-compatible constrained decoding), which is the
# single biggest fix for the "Expecting ',' delimiter" parse failures
# seen from plan_node/verify_node's free-text JSON requests. Both models
# used for structured output get this; the strong synthesis model does
# NOT get it since analyze_node's draft answer is prose, not JSON.
_JSON_MODE_KWARGS = {"response_format": {"type": "json_object"}}

llm = ChatGroq(model=MODEL_NAME, temperature=0, max_tokens=2000)
planning_llm = ChatGroq(
    model=PLANNING_MODEL_NAME, temperature=0, max_tokens=800,
    model_kwargs=_JSON_MODE_KWARGS,
)
fallback_llm = ChatGroq(model=FALLBACK_MODEL_NAME, temperature=0, max_tokens=2000)
# Claim-level verification asks for a JSON array with one entry per
# sentence in the draft — for a typical 8-12 sentence answer this can
# easily exceed planning_llm's 800-token cap and get truncated mid-JSON,
# which silently fails to parse and produces a false "0 claims" result.
# Give this call its own larger budget instead of reusing planning_llm.
verification_llm = ChatGroq(
    model=PLANNING_MODEL_NAME, temperature=0, max_tokens=1800,
    model_kwargs=_JSON_MODE_KWARGS,
)


def _is_daily_quota_error(error_message: str) -> bool:
    """Distinguishes a daily-cap (TPD) limit from a short transient rate limit."""
    return bool(re.search(r"tokens per day|\bTPD\b", error_message, re.IGNORECASE))


def _sleep_with_heartbeat(total_seconds: float, interval: float = HEARTBEAT_INTERVAL_SECONDS):
    """Sleeps in small chunks, printing progress, so a wait never looks like a hang."""
    remaining = total_seconds
    while remaining > 0:
        chunk = min(interval, remaining)
        time.sleep(chunk)
        remaining -= chunk
        if remaining > 0:
            print(f"    ...still waiting ({remaining:.0f}s left)")


def _invoke_with_retry(model, prompt: str, max_retries: int = LLM_MAX_RETRIES, fallback=None):
    """
    Wraps llm.invoke with backoff on rate-limit (429) errors. Groq's error
    message includes a suggested wait time (e.g. "try again in 11m50s");
    if present we parse and honor it, otherwise we fall back to simple
    exponential backoff.

    A DAILY quota error (tokens-per-day) is handled differently: waiting
    won't resolve it within a session, so if a `fallback` model is given
    we switch to it immediately instead of blocking. Any wait is also
    capped and reported via heartbeat prints so nothing looks stuck.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return model.invoke(prompt)
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
            if not is_rate_limit:
                raise

            if _is_daily_quota_error(msg):
                if fallback is not None:
                    print("  [rate limit] daily token quota reached for this model — "
                          "falling back to a lighter model instead of waiting.")
                    return fallback.invoke(prompt)
                if attempt == max_retries:
                    raise
                print("  [rate limit] daily token quota reached and no fallback model set — "
                      "this will likely keep failing until the quota resets.")

            if attempt == max_retries:
                raise

            wait_s = _parse_retry_after(msg)
            if wait_s is None:
                wait_s = min(60, 2 ** attempt * 5)  # 5s, 10s, 20s... capped at 60s
            capped_wait_s = min(wait_s, MAX_SINGLE_WAIT_SECONDS)
            note = f" (capped from {wait_s:.0f}s)" if capped_wait_s < wait_s else ""
            print(f"  [rate limit] attempt {attempt + 1}/{max_retries} — waiting {capped_wait_s:.0f}s{note} before retry...")
            _sleep_with_heartbeat(capped_wait_s)
    raise last_err


def _parse_retry_after(error_message: str) -> Optional[float]:
    """Extract a suggested wait time like '11m50.208s' or '30s' from a Groq error message."""
    m = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", error_message, re.IGNORECASE)
    if not m:
        return None
    minutes = float(m.group(1)) if m.group(1) else 0.0
    seconds = float(m.group(2))
    return minutes * 60 + seconds + 1  # +1s buffer


ACADEMIC_DOMAINS = [
    "arxiv.org",
    "semanticscholar.org",
    "researchgate.net",
    "dl.acm.org",
    "ieeexplore.ieee.org",
]

WORD_TO_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "a": 1, "single": 1}

# How many candidate papers to aim for when the user wants papers but
# didn't say how many.
BROAD_MODE_TARGET = 20

_STOPWORDS = {
    "the", "explain", "what", "how", "why", "give", "tell", "using",
    "based", "from", "with", "about", "research", "paper", "papers",
    "study", "studies", "please", "can", "you", "me", "in", "a", "an",
    "compare", "summarize", "describe", "discuss", "list", "show",
    "analyze", "outline", "define", "write", "find",
    # Generic qualifiers that add no topic information on their own, but
    # would otherwise get swept up into the "longest capitalized run"
    # heuristic below (especially for ALL-CAPS questions, where every
    # word looks "capitalized"). e.g. "DESCRIBE MCP IN DETAIL" should
    # lock onto "MCP", not "MCP IN DETAIL".
    "detail", "details", "depth", "brief", "briefly", "overview",
    "of", "on", "to", "and", "or", "for", "is", "are", "does", "do",
    "it", "its", "into", "at", "as",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    pending_queries: List[str]
    sources: List[SearchResult]          # flattened, deduped, in citation order
    draft_answer: str
    verified: bool
    verification_notes: str
    iteration: int
    final_answer: str
    paper_mode: bool                     # True for count / broad / specific
    paper_query_mode: str                # "none" | "count" | "broad" | "specific"
    paper_count: Optional[int]           # target count (broad -> BROAD_MODE_TARGET)
    paper_title: Optional[str]           # set only in "specific" mode
    topic_entity: Optional[str]          # locked topic, re-injected into every query
    eval_scores: Dict[str, float]        # RAGAS quality scores, set in finalize_node
    support_ratio: Optional[float]       # fraction of claims individually supported; None if unparseable
    unsupported_claims: List[Dict]       # claims that failed claim-level verification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str):
    """
    LLMs sometimes wrap JSON in ```json fences or add preamble. Strip it,
    then parse. If a strict parse fails (e.g. an unescaped quote inside a
    claim string, or a response truncated mid-object), fall back to
    json_repair if it's installed rather than raising immediately — this
    recovers the common near-miss cases without needing another LLM call.
    """
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if _JSON_REPAIR_AVAILABLE:
            repaired = repair_json(text)
            return json.loads(repaired)
        raise


def _format_sources_block(sources: List[SearchResult], include_indices: Optional[List[int]] = None) -> str:
    """
    Renders sources as a numbered block for prompts. Numbers are always
    the source's position in the FULL `sources` list (1-indexed), even
    when only a subset is shown via `include_indices` — this keeps [n]
    citations in the draft answer pointing at the correct entry in
    `state["sources"]` later in finalize_node, regardless of how many
    sources were actually shown to the LLM at synthesis time.
    """
    if not sources:
        return "(no sources gathered yet)"
    indices = include_indices if include_indices is not None else list(range(len(sources)))
    if not indices:
        return "(no sources gathered yet)"
    return "\n".join(
        f"[{i+1}] {sources[i].title}\n    URL: {sources[i].url}\n    Snippet: {sources[i].snippet}"
        for i in indices
    )


def _naive_top_indices(sources: List[SearchResult], cap: int) -> List[int]:
    """Fallback front-slice, used only if the reranker is unavailable or errors."""
    return list(range(min(cap, len(sources))))


def _top_synthesis_indices(sources: List[SearchResult], cap: int, query: str = "") -> List[int]:
    """
    Selects which sources (by index into the full list) get shown to the
    LLM during synthesis, when the pool is larger than `cap`. Uses a
    flashrank cross-encoder reranker to score each source's relevance to
    the actual question, rather than relying on retrieval order — search
    APIs don't always return their best result first, and this is what
    actually determines which sources reach the LLM at all. Falls back to
    a naive front-slice if flashrank isn't installed or the rerank call
    fails for any reason, so this can never break the pipeline.
    """
    if len(sources) <= cap:
        return list(range(len(sources)))

    ranker = _get_reranker()
    if ranker is None or not query:
        return _naive_top_indices(sources, cap)

    try:
        passages = [
            {"id": i, "text": f"{s.title}. {s.snippet}"}
            for i, s in enumerate(sources)
        ]
        reranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        # flashrank returns passages sorted best-first, each carrying back
        # the original "id" we passed in (our index into `sources`).
        ranked_indices = [r["id"] for r in reranked][:cap]
        return ranked_indices
    except Exception as e:
        print(f"  [rerank] warning: reranking failed, falling back to retrieval order: {e}")
        return _naive_top_indices(sources, cap)


def _normalize_title(title: str) -> str:
    """
    Normalize a source title so the same paper indexed on two different
    sites (e.g. an arXiv abstract page and a ResearchGate mirror) collapses
    to one dedup key instead of showing up twice in the reference list.
    """
    if not title:
        return ""
    t = title.lower()
    # Strip trailing " - arXiv", " | Semantic Scholar", etc.
    t = re.split(r"\s*[\|\u2013\u2014-]\s*(?:arxiv|semantic scholar|researchgate|acm|ieee).*$", t)[0]
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t


def _dedup_key(source: SearchResult) -> str:
    return _normalize_title(getattr(source, "title", "")) or (getattr(source, "url", "") or "")


def _looks_like_acronym(term: str) -> bool:
    """True for short, all-caps single words like 'MCP', 'API', 'RAG' —
    the kind of term that's prone to colliding with an unrelated expansion
    elsewhere on the web (e.g. 'MCP' = Model Context Protocol vs. 'Model
    Control Protocols')."""
    return bool(term) and 2 <= len(term) <= 6 and term.isalpha() and term.isupper()


def _extract_expansion(text: str, acronym: str) -> Optional[str]:
    """
    Looks for a spelled-out expansion of `acronym` in `text` — e.g. for
    acronym="MCP", matches phrases like "Model Context Protocol" (allowing
    up to 2 lowercase filler words between the initials, e.g. "Ministry
    of Community Planning"). Returns the normalized matched phrase, or
    None if no such expansion is present in this text.
    """
    if not text:
        return None
    segments = []
    for i, ch in enumerate(acronym):
        word_pattern = rf"{ch}[a-zA-Z]*"
        if i == 0:
            segments.append(word_pattern)
        else:
            segments.append(rf"(?:[a-z]+\s+){{0,2}}{word_pattern}")
    pattern = r"\b" + r"\s+".join(segments) + r"\b"
    m = re.search(pattern, text)
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(0)).strip().lower()


def _filter_acronym_collisions(sources: List[SearchResult], acronym: Optional[str]):
    """
    Guards against a short acronym (e.g. "MCP") legitimately expanding to
    two different, unrelated things across the web (e.g. "Model Context
    Protocol" vs. "Model Control Protocols"). Retrieval/dedup has no way
    to notice this on its own since both results genuinely mention "MCP"
    — but citing the wrong expansion's source can quietly wreck
    faithfulness (the source doesn't actually support claims about the
    *other* meaning) even though claim-level verification passes, since
    the verifier is just checking claim-vs-snippet agreement, not
    checking that the snippet is about the right underlying concept.

    Detects each source's expansion (if any is stated nearby in its
    title/snippet), finds the dominant one across the pool, and drops
    sources tied to a different, minority expansion. Sources with no
    detected expansion at all are kept (benefit of the doubt — most
    sources reference the acronym without ever spelling it out).
    """
    if not _looks_like_acronym(acronym or ""):
        return sources, 0

    expansions = []
    for s in sources:
        text = f"{getattr(s, 'title', '') or ''} {getattr(s, 'snippet', '') or ''}"
        expansions.append(_extract_expansion(text, acronym))

    counts = Counter(e for e in expansions if e)
    if len(counts) <= 1:
        return sources, 0  # no collision detected — nothing to filter

    dominant, _ = counts.most_common(1)[0]
    kept = []
    dropped = 0
    for s, e in zip(sources, expansions):
        if e is None or e == dominant:
            kept.append(s)
        else:
            dropped += 1
    return kept, dropped


# Title/URL signals that a result is NOT an actual paper, even though it
# was returned by a `site:` search against an academic domain (blog posts,
# guides, and listing pages get indexed there too).
_NON_PAPER_TITLE_SIGNALS = (
    "overview", "guide", "blog", "news", "tutorial", "introduction to",
    "what is", "how to", "top ", "best ", "checklist", "cheat sheet",
    "roundup", "explained", "for beginners",
)
_NON_PAPER_URL_SIGNALS = ("/blog/", "/news/", "/guide/", "/tutorial/", "/topics/")

# URL shapes that DO look like an actual paper record (id-like paths,
# abstract/PDF pages, DOIs). If a result matches one of these, it's kept
# even if a title signal above would otherwise flag it — avoids false
# positives on legitimately-titled papers (e.g. "An Introduction to RAG").
_PAPER_URL_SIGNALS = (
    "arxiv.org/abs/", "arxiv.org/pdf/", "arxiv.org/html/",
    "semanticscholar.org/paper/", "researchgate.net/publication/",
    "dl.acm.org/doi/", "ieeexplore.ieee.org/document/",
)


def _looks_like_paper(source: SearchResult) -> bool:
    """
    Heuristic filter used only in paper mode: drops results that are
    almost certainly not an actual research paper, so junk never reaches
    the LLM in the first place instead of relying solely on prompt
    instructions to exclude it.
    """
    title = (getattr(source, "title", "") or "").lower()
    url = (getattr(source, "url", "") or "").lower()

    if any(sig in url for sig in _PAPER_URL_SIGNALS):
        return True
    if any(sig in url for sig in _NON_PAPER_URL_SIGNALS):
        return False
    if any(sig in title for sig in _NON_PAPER_TITLE_SIGNALS):
        return False
    return True


def extract_topic_entity(question: str) -> Optional[str]:
    """
    Extract the main proper-noun / topic term from the question so it can
    be re-injected into every follow-up query (planning + verification
    retry loop), preventing topic drift across iterations.
    """
    # Prefer an explicitly quoted term first.
    quoted = re.search(r"\"([^\"]{2,250})\"|'([^']{2,250})'", question)
    if quoted:
        return (quoted.group(1) or quoted.group(2)).strip()

    # If the question is ALL CAPS (or close to it), capitalization carries
    # no signal about proper nouns — every word "looks" capitalized, so
    # the CamelCase-run heuristic below would grab the entire sentence
    # (e.g. "DESCRIBE MCP IN DETAIL" -> "MCP IN DETAIL" instead of just
    # "MCP"), and that whole phrase then gets forced into every search
    # query, diluting retrieval. Detect this case and fall back to
    # picking the single strongest remaining token after stopword removal
    # instead of a capitalized-run match.
    letters_only = re.sub(r"[^A-Za-z]", "", question)
    is_shouted = len(letters_only) > 3 and letters_only == letters_only.upper()
    if is_shouted:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", question)
        remaining = [t for t in tokens if t.lower() not in _STOPWORDS]
        if not remaining:
            return None
        # Acronyms/product names (MCP, API, AWS) are typically shorter
        # than descriptive words (architecture, specifications) in a
        # question like this, so shortest-remaining-token is a reasonable
        # proxy for "the actual topic" when case gives no other signal.
        return min(remaining, key=len)

    # Otherwise take the longest run of capitalized / CamelCase tokens
    # that isn't just a sentence-starting stopword.
    candidates = re.findall(
        r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b", question
    )
    cleaned = []
    for c in candidates:
        tokens = c.split()
        while tokens and tokens[0].lower() in _STOPWORDS:
            tokens.pop(0)
        while tokens and tokens[-1].lower() in _STOPWORDS:
            tokens.pop()
        if tokens:
            cleaned.append(" ".join(tokens))
    if cleaned:
        return max(cleaned, key=len).strip()
    return None


def extract_paper_intent(question: str):
    """
    Detects paper-related intent in the question. Returns a tuple:
        (paper_query_mode, paper_count, paper_title)

    paper_query_mode is one of:
        "none"     - no paper intent detected
        "specific" - a named/quoted paper was mentioned -> paper_title set
        "count"    - a specific number of papers was requested
        "broad"    - papers/studies mentioned, no number given
    """
    # --- Specific paper name detection -------------------------------
    title_patterns = [
        r"paper\s+(?:called|titled|named)\s+[\"']?([^\"'.\n]{4,250}?)[\"']?(?:\s*$|[.,?!])",
        r"[\"']([^\"']{4,250})[\"']\s+paper\b",
        r"paper\s+[\"']([^\"']{4,250})[\"']",
    ]
    for pat in title_patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            if title:
                return "specific", None, title

    text = question.lower()
    count_match = re.search(
        r"\b(\d+|one|two|three|four|five|a|single)\s+(research\s+)?"
        r"(paper(s)?|stud(?:y|ies))\b",
        text,
    )
    mentions_paper = bool(re.search(r"\bpaper(s)?\b", text)) or bool(
        re.search(r"\bstud(y|ies)\b", text)
    )

    if count_match:
        val = count_match.group(1)
        count = int(val) if val.isdigit() else WORD_TO_NUM.get(val, 1)
        return "count", count, None
    elif mentions_paper:
        return "broad", BROAD_MODE_TARGET, None

    return "none", None, None


def _lock_topic(query_text: str, topic_entity: Optional[str]) -> str:
    """Ensure a query string explicitly contains the locked topic term."""
    if not topic_entity:
        return query_text
    if topic_entity.lower() in query_text.lower():
        return query_text
    return f"{topic_entity} {query_text}"


def _init_state(question: str) -> AgentState:
    paper_query_mode, paper_count, paper_title = extract_paper_intent(question)
    paper_mode = paper_query_mode != "none"
    topic_entity = extract_topic_entity(question)

    if paper_mode:
        if paper_query_mode == "specific":
            print(f"  [paper mode: specific] targeting paper: \"{paper_title}\"")
        elif paper_query_mode == "broad":
            print(f"  [paper mode: broad] no count given — gathering up to {paper_count} candidate papers")
        else:
            print(f"  [paper mode: count] targeting exactly {paper_count} academic source(s)")
    if topic_entity:
        print(f"  [topic lock] \"{topic_entity}\"")

    return {
        "query": question,
        "sub_queries": [],
        "pending_queries": [],
        "sources": [],
        "draft_answer": "",
        "verified": False,
        "verification_notes": "",
        "iteration": 0,
        "final_answer": "",
        "paper_mode": paper_mode,
        "paper_query_mode": paper_query_mode,
        "paper_count": paper_count,
        "paper_title": paper_title,
        "topic_entity": topic_entity,
        "eval_scores": {},
        "support_ratio": 0.0,
        "unsupported_claims": [],
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def plan_node(state: AgentState) -> AgentState:
    print(f"\n[PLAN] Breaking down: \"{state['query']}\"")

    topic_lock_instruction = ""
    if state["topic_entity"]:
        topic_lock_instruction = (
            f"\nCRITICAL — TOPIC LOCK: this question is specifically about "
            f"\"{state['topic_entity']}\". Every single query you write MUST "
            f"contain the exact term \"{state['topic_entity']}\" verbatim "
            f"(same spelling/casing). Do not rename it, paraphrase it, or "
            f"substitute a related/competing term.\n"
            f"If \"{state['topic_entity']}\" could plausibly refer to more "
            f"than one thing (e.g. it's a short acronym reused across "
            f"different industries — firmware, networking, chemistry,"
            f"etc.), pick the single interpretation that best fits the "
            f"overall phrasing of the question below, and use ONLY that "
            f"one interpretation in every query. Do NOT hedge your bets by "
            f"writing some queries for one meaning and other queries for a "
            f"different, unrelated meaning — that splits the search effort "
            f"across two different topics and produces a shallow, mixed "
            f"result set instead of a thorough one."
        )

    mode = state["paper_query_mode"]
    paper_instruction = ""
    if mode == "specific":
        paper_instruction = (
            f"\nThe user wants information about ONE SPECIFIC paper: "
            f"\"{state['paper_title']}\". Write queries designed to locate "
            f"that exact paper on academic sites (its title, its title plus "
            f"\"arxiv\", its title plus \"pdf\", its title plus author/abstract). "
            f"Do not write generic topic-overview queries."
        )
    elif mode == "count":
        paper_instruction = (
            f"\nThe user wants this answered using exactly {state['paper_count']} "
            f"real academic research paper(s), not blogs or marketing pages. "
            f"Write queries suited to finding papers on academic sites (include "
            f"terms like 'paper', 'study', or the topic's technical name)."
        )
    elif mode == "broad":
        paper_instruction = (
            f"\nThe user wants this answered using real academic research "
            f"papers, and did not specify how many — so write several varied "
            f"queries (covering different angles/subtopics) so that up to "
            f"{state['paper_count']} distinct genuine papers can be found. "
            f"Avoid blogs or marketing pages."
        )

    prompt = f"""You are a research assistant planning how to answer a question.

Question: {state['query']}

Break this into 3-5 focused, distinct web search queries that together
would gather enough evidence to answer it thoroughly (e.g. if it's a
comparison, plan queries covering each side AND queries covering the
direct comparison/trade-offs).

IMPORTANT: Use the exact proper nouns, product names, and technical terms
from the question verbatim — do not paraphrase, rename, or introduce
spelling variants of names (e.g. if the question says "LangChain", every
query must say "LangChain", not an alternate spelling).
{topic_lock_instruction}
{paper_instruction}

Respond with ONLY a JSON object of this exact shape, nothing else:
{{"queries": ["query one", "query two", "query three"]}}"""

    response = _invoke_with_retry(planning_llm, prompt)
    try:
        parsed = _extract_json(response.content)
        # response_format=json_object requires the top-level response to be
        # a JSON *object*, not a bare array — so the prompt now asks for
        # {"queries": [...]} rather than a raw array like before.
        queries = parsed["queries"] if isinstance(parsed, dict) else parsed
        assert isinstance(queries, list) and all(isinstance(q, str) for q in queries)
    except Exception:
        queries = [state["query"]]

    # Belt-and-suspenders: force the locked topic into every query even if
    # the LLM slipped, and seed a direct query for the specific-title case.
    queries = [_lock_topic(q, state["topic_entity"]) for q in queries]
    if mode == "specific" and state["paper_title"]:
        queries = [state["paper_title"]] + queries

    print("  Search plan:")
    for q in queries:
        print(f"    - {q}")

    return {**state, "sub_queries": queries, "pending_queries": queries}


def retrieve_node(state: AgentState) -> AgentState:
    mode = state["paper_query_mode"]

    # Cap how large the candidate pool grows to, per mode, so retrieval
    # doesn't run away — with a small buffer over the target so analyze
    # has room to pick the best ones.
    if mode == "specific":
        pool_cap = 10
    elif mode == "count":
        pool_cap = (state["paper_count"] or 3) * 3
    elif mode == "broad":
        pool_cap = state["paper_count"] or BROAD_MODE_TARGET
    else:
        pool_cap = None

    # Build the full list of independent search calls up front. These are
    # network I/O calls with no dependency on each other, so they're run
    # concurrently below instead of one-at-a-time — with up to 25 calls in
    # broad paper mode, sequential execution was the main source of
    # slowness.
    #
    # arxiv.org is special-cased to hit the real Arxiv API directly
    # (search_arxiv_structured) rather than a DuckDuckGo `site:arxiv.org`
    # search — DDG's coverage of arXiv depends on what it happens to have
    # indexed, while the Arxiv API is authoritative and typically returns
    # more/better matches for a given topic.
    #
    # Chroma is queried alongside web/arxiv (not only as a fallback) so
    # semantically relevant results from prior runs are available to
    # analyze_node in the very same pass — this is the RAG "long-term
    # memory" layer. It is skipped in paper_mode since that flow is
    # restricted to live academic-domain search only.
    tasks = []  # list of (label, kind, query_string, max_results)
    if state["paper_mode"]:
        per_domain_results = 4 if mode == "broad" else 2
        for q in state["pending_queries"]:
            for domain in ACADEMIC_DOMAINS:
                if domain == "arxiv.org":
                    label = f"arxiv API: {q}"
                    tasks.append((label, "arxiv", q, per_domain_results))
                else:
                    domain_query = f"site:{domain} {q}"
                    tasks.append((domain_query, "web", domain_query, per_domain_results))
    else:
        for q in state["pending_queries"]:
            tasks.append((q, "web", q, 4))
            tasks.append((f"chroma: {q}", "chroma", q, 5))

    print(f"\n[RETRIEVE] Running {len(tasks)} search(es) in parallel...")

    def _run_task(label, kind, query_string, max_results):
        try:
            if kind == "arxiv":
                return label, search_arxiv_structured(query_string, max_results=max_results), None
            if kind == "chroma":
                return label, query_chroma(query_string, n_results=max_results), None
            return label, web_search(query_string, max_results=max_results), None
        except Exception as e:
            return label, [], e

    task_results = []
    if tasks:
        max_workers = min(SEARCH_MAX_WORKERS, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_task, label, kind, q, n): label
                for (label, kind, q, n) in tasks
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    task_results.append(future.result(timeout=SEARCH_TIMEOUT_SECONDS))
                except FuturesTimeoutError:
                    print(f"  Searching: {label} -> timed out after {SEARCH_TIMEOUT_SECONDS}s, skipping")
                    task_results.append((label, [], "timeout"))

    # Dedup/filter sequentially (cheap, and avoids any shared-state races
    # from the concurrent calls above).
    sources = list(state["sources"])
    seen_keys = {_dedup_key(s) for s in sources}
    pool_cap_hit = False

    for label, results, err in task_results:
        if err and err != "timeout":
            print(f"  Searching: {label} -> error: {err}")
            continue
        added = 0
        for r in results:
            if pool_cap is not None and len(sources) >= pool_cap:
                pool_cap_hit = True
                break
            if not r.url:
                continue
            if state["paper_mode"] and not _looks_like_paper(r):
                continue
            key = _dedup_key(r)
            if key in seen_keys:
                continue
            sources.append(r)
            seen_keys.add(key)
            added += 1
        if err != "timeout":
            print(f"  Searching: {label} -> {len(results)} result(s), {added} new")

    if pool_cap_hit:
        print(f"  Pool cap ({pool_cap}) reached — extra results discarded.")

    if state["topic_entity"]:
        sources, n_dropped = _filter_acronym_collisions(sources, state["topic_entity"])
        if n_dropped:
            print(f"  [acronym filter] Dropped {n_dropped} source(s) using a different "
                  f"expansion of \"{state['topic_entity']}\" than the dominant one in this pool.")

    print(f"  Total unique sources so far: {len(sources)}")
    return {**state, "sources": sources, "pending_queries": []}


def analyze_node(state: AgentState) -> AgentState:
    print("\n[ANALYZE] Synthesizing evidence into a draft answer...")

    mode = state["paper_query_mode"]
    if not state["paper_mode"] and len(state["sources"]) > MAX_SYNTHESIS_SOURCES:
        synthesis_indices = _top_synthesis_indices(state["sources"], MAX_SYNTHESIS_SOURCES, query=state["query"])
        print(f"  Capping synthesis context: {len(state['sources'])} retrieved -> "
              f"top {len(synthesis_indices)} shown to LLM (reranked).")
    else:
        synthesis_indices = None  # show all sources (paper modes need the full pool)

    sources_block = _format_sources_block(state["sources"], include_indices=synthesis_indices)
    paper_instruction = ""
    if mode == "specific":
        paper_instruction = f"""
The user wants information specifically about ONE PAPER: "{state['paper_title']}".
Find the source(s) in the evidence below that are that exact paper (allow
for minor title wording differences) and answer using ONLY that paper's
content — do not summarize any other paper. If no source below is
actually that paper, say clearly that it wasn't found in the evidence
gathered, rather than substituting a different paper."""
    elif mode == "count":
        n = state["paper_count"]
        paper_instruction = f"""
The user specifically wants this answered using exactly {n} research
paper(s). Among the evidence below, identify the {n} source(s) that are
genuinely academic papers (from arXiv, Semantic Scholar, ResearchGate,
ACM, IEEE, or similar) and base your ENTIRE answer only on those {n}
source(s), ignoring all other evidence even if present. If fewer than {n}
genuine papers exist in the evidence, say so explicitly and use only the
ones that are real papers."""
    elif mode == "broad":
        n = state["paper_count"]
        paper_instruction = f"""
The user wants this answered using real academic research papers but did
not specify how many. Among the evidence below, use as many genuinely
academic papers (from arXiv, Semantic Scholar, ResearchGate, ACM, IEEE,
or similar) as are actually present and relevant, up to {n}. Do not pad
the count with non-academic sources, and do not invent papers that
aren't in the evidence — cite only what's genuinely there, even if that's
fewer than {n}."""

    prompt = f"""You are a research assistant. Using ONLY the evidence below,
write a clear, structured answer to the question. Cite sources inline
using their bracket numbers, e.g. [1], [2].

Start with one sentence that directly restates and answers the core of
the question, using the same key terms the question uses. Then expand
with supporting detail.

Answer the question as directly and specifically as possible. If a
source contains information that is only tangentially related to the
question (e.g. unrelated case studies, or background on a different but
similar topic), leave it out even if it's interesting — every sentence
should clearly serve the question being asked, not just be "about" the
same general subject.

EXCLUDE vendor/consulting marketing content even when it's technically
about the topic — e.g. a company's list of paid services, integrations
they sell, or client offerings ("our services include...", "we provide
custom development of..."). This describes a business, not the
technology itself, and should never be cited even if a source states it.

If the evidence doesn't fully support a claim, don't state it as fact.

Do NOT add your own "References" or "Sources" section at the end — only
use inline [n] citations within the text. A reference list will be
generated automatically afterward.

Do NOT add a closing summary or evaluative sentence (e.g. "this
technology has the potential to revolutionize...") unless that exact
judgment is directly stated in a cited source. End the answer on the
last substantive, source-backed point.
{paper_instruction}

Question: {state['query']}

Evidence (numbered sources):
{sources_block}

Write the draft answer now (with inline [n] citations, no trailing
references list):"""

    response = _invoke_with_retry(llm, prompt, fallback=fallback_llm)
    print("  Draft answer generated.")

    # Persist this run's gathered sources into the Chroma vector store so
    # future queries can hit them via semantic search instead of only
    # relying on live web/arxiv calls. Wrapped defensively so a storage
    # hiccup never breaks the answer pipeline itself.
    try:
        add_to_chroma(state["sources"])
    except Exception as e:
        print(f"  [chroma] warning: failed to store sources for future reuse: {e}")

    return {**state, "draft_answer": response.content}


def verify_node(state: AgentState) -> AgentState:
    """
    Claim-level verification. Rather than asking the LLM to pass/fail the
    draft as a whole (which tends to rubber-stamp "verified: true" even
    when RAGAS later finds low faithfulness), this splits the draft into
    individual claims and checks each one against the specific source it
    cites. `support_ratio` (fraction of claims that hold up) drives the
    retry decision, and unsupported claims are passed through to
    finalize_node so they can be stripped before the answer is scored.

    On a JSON parse failure, this retries ONLY the verification LLM call
    itself (cheap) up to VERIFY_PARSE_RETRIES times before giving up —
    NOT the full retrieve->analyze->verify loop, which would re-run
    retrieval and re-generate the entire draft just for another shot at
    valid JSON, and was the main cause of cascading Groq 429s.
    """
    print("\n[VERIFY] Checking draft against evidence (claim-level)...")

    # --- Cap what's shown to the verifier -------------------------------
    # Only the sources actually cited [n] in the draft need to be checked
    # against — that's all the claim-level check can ever reference.
    # Previously this passed the FULL retrieved pool (title + URL + full
    # snippet for every source, sometimes 50-60+) to verification_llm,
    # which runs on a small 6,000-TPM Groq model. That's what caused the
    # 413 "Request too large" errors once the pool grew past ~20-30
    # sources. Restricting to cited sources (with a reranked fallback if
    # the draft somehow has no citations) keeps this well within limits.
    cited_in_draft = sorted({int(n) - 1 for n in re.findall(r"\[(\d+)\]", state["draft_answer"])})
    cited_in_draft = [i for i in cited_in_draft if 0 <= i < len(state["sources"])]

    if cited_in_draft:
        verify_indices = cited_in_draft
    elif len(state["sources"]) > MAX_SYNTHESIS_SOURCES:
        verify_indices = _top_synthesis_indices(state["sources"], MAX_SYNTHESIS_SOURCES, query=state["query"])
    else:
        verify_indices = None  # small pool, show everything

    if verify_indices is not None:
        print(f"  Capping verify context: {len(state['sources'])} retrieved -> "
              f"{len(verify_indices)} shown to verifier.")

    sources_block = _format_sources_block(state["sources"], include_indices=verify_indices)

    topic_lock_instruction = ""
    if state["topic_entity"]:
        topic_lock_instruction = (
            f"\nIf you propose additional_search_queries, every one of them "
            f"MUST contain the exact term \"{state['topic_entity']}\" "
            f"verbatim — do not drift onto a related but different topic."
        )

    prompt = f"""You are a fact-checking reviewer. Break the DRAFT ANSWER into
individual factual claims (roughly one per sentence). For EACH claim,
check whether it is directly supported by the specific numbered source(s)
it cites in the evidence below. A claim is only "supported" if the cited
source's snippet actually contains that information — not just related
information, and not general topic knowledge.

When writing each "claim" value, paraphrase it in plain prose and avoid
using quotation marks or apostrophes inside the string — this output
must be valid JSON.

Question: {state['query']}

Evidence:
{sources_block}

Draft answer:
{state['draft_answer']}
{topic_lock_instruction}

Respond with ONLY JSON in this exact shape:
{{
  "claims": [
    {{"claim": "short paraphrase of the claim", "citation": 14, "supported": true or false}},
    ...
  ],
  "additional_search_queries": ["query if unsupported claims need better evidence", ...]
}}
Include every claim that has at least one [n] citation in the draft.
If a claim has no citation, mark "citation": null and "supported": false."""

    result = None
    parse_failed = False
    response = None
    for parse_attempt in range(VERIFY_PARSE_RETRIES + 1):
        response = _invoke_with_retry(verification_llm, prompt, fallback=fallback_llm)
        try:
            result = _extract_json(response.content)
            parse_failed = False
            break
        except Exception as e:
            parse_failed = True
            print(f"  [verify] parse attempt {parse_attempt + 1}/{VERIFY_PARSE_RETRIES + 1} "
                  f"failed to parse verifier JSON: {e}")

    if parse_failed:
        print(f"  [verify] raw response (first 500 chars): {response.content[:500]}")
        claims, gap_queries = [], []
    else:
        claims = result.get("claims", []) or []
        gap_queries = result.get("additional_search_queries", []) or []

    if parse_failed:
        # We genuinely don't know whether claims are supported — this is
        # a parsing failure, not a real 0% score. Route back to retrieval
        # (if iterations remain) but don't claim a specific support ratio.
        support_ratio = None
        verified = False
        unsupported = []
        notes = "Verifier response could not be parsed as JSON after retries; re-checking."
    elif claims:
        supported_count = sum(1 for c in claims if c.get("supported"))
        support_ratio = supported_count / len(claims)
        verified = support_ratio >= MIN_SUPPORT_RATIO
        unsupported = [c for c in claims if not c.get("supported")]
        notes = (
            f"{len(unsupported)}/{len(claims)} claim(s) unsupported: "
            + "; ".join(c.get("claim", "") for c in unsupported[:5])
            if unsupported else "All claims supported."
        )
    else:
        # Valid JSON, but the model genuinely found zero citable claims
        # (e.g. an empty or citation-free draft). Distinct from a parse
        # failure, and distinct from "all claims supported".
        support_ratio = 0.0
        verified = False
        unsupported = []
        notes = "No citable claims were found in the draft."

    # Belt-and-suspenders topic lock: even if the verifier LLM forgot the
    # instruction, force the locked topic into every follow-up query so
    # the retry loop can't drift onto a different subject.
    gap_queries = [_lock_topic(q, state["topic_entity"]) for q in gap_queries]

    print(f"  Support ratio: {support_ratio:.2f}" if support_ratio is not None else "  Support ratio: unknown (parse failure)")
    if unsupported:
        print(f"  Unsupported: {[c.get('claim','') for c in unsupported]}")

    return {
        **state,
        "verified": verified,
        "verification_notes": notes,
        "pending_queries": gap_queries,
        "support_ratio": support_ratio,
        "unsupported_claims": unsupported,
    }


def finalize_node(state: AgentState) -> AgentState:
    print("\n[FINALIZE] Assembling final answer with references...")

    draft = state["draft_answer"]

    # Strip any claim that failed claim-level verification before the
    # answer is scored by RAGAS, so unsupported statements never reach
    # the final output. Wrapped defensively — if this call fails for any
    # reason, fall back to the original draft rather than blocking.
    if state.get("unsupported_claims"):
        print(f"  Stripping {len(state['unsupported_claims'])} unsupported claim(s) before finalizing...")
        strip_prompt = f"""Remove ONLY the following unsupported claims from this
answer. Do not rewrite, rephrase, or reorder anything else. Keep all
other sentences and their [n] citations exactly as-is.

CRITICAL: Just delete the listed claims/sentences. Do NOT add any words
about the fact that you removed something — no "has been removed", no
"this sentence now reads", no bracketed notes, no editorial commentary
of any kind. The reader must never see any trace that an edit happened.

Example of what NOT to do:
  BAD:  "...is seen as a novel approach has been removed, this sentence
         now reads: It has been compared to..."
  GOOD: "...It has been compared to..."

Claims to remove:
{chr(10).join('- ' + c.get('claim', '') for c in state['unsupported_claims'])}

Original answer:
{draft}

Return only the edited answer, no preamble, no commentary about the edit."""

        # Telltale phrases indicating the model narrated the edit inline
        # instead of cleanly deleting text — these have shown up leaking
        # straight into user-facing output, so treat any occurrence as a
        # failed edit rather than trying to regex it back out afterward.
        _META_EDIT_SIGNALS = (
            "has been removed", "now reads", "this sentence now",
            "was removed", "has been deleted", "edited to remove",
            "no longer includes",
        )

        try:
            response = _invoke_with_retry(llm, strip_prompt, fallback=fallback_llm)
            candidate = response.content
            lowered = candidate.lower()
            if any(sig in lowered for sig in _META_EDIT_SIGNALS):
                print("  [strip] warning: model narrated the edit instead of cleanly "
                      "removing text — discarding the stripped version and keeping "
                      "the original draft (with unsupported claims still present).")
            else:
                draft = candidate
        except Exception as e:
            print(f"  [strip] warning: failed to strip unsupported claims, using original draft: {e}")

    cited_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", draft)})
    sources = state["sources"]

    refs_lines = []
    seen_ref_keys = set()
    for n in cited_numbers:
        idx = n - 1
        if 0 <= idx < len(sources):
            s = sources[idx]
            key = _dedup_key(s)
            if key in seen_ref_keys:
                # Same paper already listed under a different bracket
                # number (shouldn't normally happen since retrieval
                # dedupes up front, but guard against it here too).
                continue
            seen_ref_keys.add(key)
            refs_lines.append(f"[{n}] {s.title} — {s.url}")

    refs = "\n".join(refs_lines) if refs_lines else "(no sources were cited)"

    # Evaluate answer quality with RAGAS, using only the snippets of
    # sources actually cited in the (now stripped) draft — scoring
    # against uncited evidence would understate faithfulness.
    print("  Scoring answer quality with RAGAS...")
    cited_contexts = []
    seen_snippet_keys = set()
    for n in cited_numbers:
        idx = n - 1
        if 0 <= idx < len(sources):
            s = sources[idx]
            key = _dedup_key(s)
            if key in seen_snippet_keys:
                continue
            seen_snippet_keys.add(key)
            if s.snippet:
                cited_contexts.append(s.snippet)

    eval_scores = evaluate_answer(state["query"], draft, cited_contexts)

    eval_block = ""
    if eval_scores:
        label_map = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy"}
        lines = [f"- {label_map.get(k, k)}: {v}" for k, v in eval_scores.items()]
        eval_block = "\n\n---\nQuality Metrics (RAGAS):\n" + "\n".join(lines)
        print(f"  Scores: {eval_scores}")
    else:
        print("  Scores unavailable (no cited context, or evaluation failed).")

    final = f"{draft}\n\n---\nReferences:\n{refs}{eval_block}"
    return {**state, "draft_answer": draft, "final_answer": final, "eval_scores": eval_scores}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def should_loop_or_finish(state: AgentState) -> str:
    if state["verified"]:
        return "finalize"
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"  (Max iterations [{MAX_ITERATIONS}] reached — finalizing with current evidence.)")
        return "finalize"
    return "retrieve_more"


def bump_iteration(state: AgentState) -> AgentState:
    return {**state, "iteration": state["iteration"] + 1}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("verify", verify_node)
    graph.add_node("bump_iteration", bump_iteration)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "verify")

    graph.add_conditional_edges(
        "verify",
        should_loop_or_finish,
        {"retrieve_more": "bump_iteration", "finalize": "finalize"},
    )
    graph.add_edge("bump_iteration", "retrieve")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_research_agent(question: str) -> str:
    app = build_graph()
    result = app.invoke(_init_state(question), config={"recursion_limit": 50})
    return result["final_answer"]


def stream_research_agent(question: str):
    """
    Generator version for UIs: yields (node_name, partial_state) after each
    node finishes, so a caller (e.g. Streamlit) can render progress live
    instead of waiting for the whole pipeline to finish.
    """
    app = build_graph()
    for update in app.stream(
        _init_state(question), config={"recursion_limit": 50}, stream_mode="updates"
    ):
        for node_name, partial_state in update.items():
            yield node_name, partial_state