"""
Hybrid retrieval tools for the Research Scientist Agent.
Includes:
1. DuckDuckGo (DDGS) -> general web search
2. Arxiv        -> academic research papers (via the `arxiv` package
                    directly, not LangChain's ArxivQueryRun wrapper --
                    that class has moved between langchain / langchain_community
                    across versions and kept breaking imports; calling the
                    underlying `arxiv` library directly avoids that churn).

Note: newer versions of the `arxiv` package moved result iteration off of
`Search.results()` (removed) and onto `Client.results(search)` instead.
Both functions below use the `Client`-based API so they keep working on
current `arxiv` package versions.

Rate limiting
-------------
arXiv's API asks for roughly one request every 3 seconds per client.
agent.py's retrieve_node can fan out many concurrent arxiv calls in the
same ThreadPoolExecutor batch (e.g. broad paper mode x several queries),
so _arxiv_call_with_backoff below serializes every call to
_ARXIV_CLIENT.results() through a shared token bucket and retries with
backoff on HTTP 429, instead of letting each concurrent call hit the API
at once and fail immediately.

The token bucket's minimum interval is also *adaptive*: a 429 seen by
any thread widens the shared interval for every caller (not just the
thread that got throttled), and a clean success gradually relaxes it
back down toward the base interval. This matters specifically because
retrieve_node fans out concurrently -- without a shared signal, each
thread only backs off after *it personally* gets a 429, so several
threads can independently burn through their own retries in parallel
against an API that already asked everyone to slow down.
"""
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import List
from urllib.error import HTTPError

from ddgs import DDGS

try:
    import arxiv
    _ARXIV_AVAILABLE = True
    _ARXIV_CLIENT = arxiv.Client()
except ImportError:
    _ARXIV_AVAILABLE = False
    _ARXIV_CLIENT = None

ARXIV_MIN_INTERVAL_SECONDS = float(os.environ.get("RESEARCH_AGENT_ARXIV_MIN_INTERVAL", "3.2"))
ARXIV_MAX_RETRIES = int(os.environ.get("RESEARCH_AGENT_ARXIV_MAX_RETRIES", "4"))
# Ceiling on how far the adaptive interval is allowed to widen, so a run
# of 429s can't make arxiv calls effectively grind to a halt.
ARXIV_MAX_INTERVAL_SECONDS = float(os.environ.get("RESEARCH_AGENT_ARXIV_MAX_INTERVAL", "15.0"))


class _ArxivRateLimiter:
    """
    Shared token bucket so concurrent callers still hit arXiv at roughly
    one request per `min_interval`. `min_interval` is adaptive: it widens
    (up to `max_interval`) whenever any caller reports a 429, and eases
    back toward `base_interval` on reported successes, so the whole
    thread pool slows down together instead of only the threads that
    personally got throttled.
    """
    def __init__(self, min_interval: float = ARXIV_MIN_INTERVAL_SECONDS,
                 max_interval: float = ARXIV_MAX_INTERVAL_SECONDS):
        self.base_interval = min_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()

    def report_429(self):
        with self._lock:
            widened = min(self.min_interval * 1.5, self.max_interval)
            if widened > self.min_interval:
                print(f"    [arxiv] widening shared request interval to {widened:.1f}s after a 429")
            self.min_interval = widened

    def report_success(self):
        with self._lock:
            if self.min_interval > self.base_interval:
                self.min_interval = max(self.base_interval, self.min_interval * 0.9)


_arxiv_limiter = _ArxivRateLimiter()


def _arxiv_call_with_backoff(search: "arxiv.Search", max_retries: int = ARXIV_MAX_RETRIES):
    """
    Runs _ARXIV_CLIENT.results(search) through the shared rate limiter,
    retrying with exponential backoff + jitter on HTTP 429. This is the
    single choke point both search_arxiv() and search_arxiv_structured()
    go through, so the retry logic only needs to live in one place.

    A 429 also reports back to the shared limiter so every concurrent
    caller widens its wait, not just this one. A clean result reports
    success so the interval relaxes back down over time.

    Raises the last error if all retries are exhausted -- callers keep
    their own try/except around this to decide how to degrade (return ""
    or [] rather than raising further).
    """
    last_err = None
    for attempt in range(max_retries + 1):
        _arxiv_limiter.wait()
        try:
            results = list(_ARXIV_CLIENT.results(search))
            _arxiv_limiter.report_success()
            return results
        except HTTPError as e:
            last_err = e
            if e.code != 429:
                raise
            _arxiv_limiter.report_429()
            if attempt == max_retries:
                raise
            backoff = (2 ** (attempt + 1)) + random.uniform(0, 1.5)
            print(f"    [arxiv] 429 -- retrying in {backoff:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(backoff)
        except Exception as e:
            # Non-HTTP errors (timeouts, connection resets): one retry then raise.
            last_err = e
            if attempt == max_retries:
                raise
            time.sleep(2 + random.uniform(0, 1))
    raise last_err


# -------------------------------
# Data structure for web results
# -------------------------------
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_citation_line(self, idx: int) -> str:
        return f"[{idx}] {self.title} — {self.url}"


# -------------------------------
# DuckDuckGo Web Search
# -------------------------------
def web_search(query: str, max_results: int = 5) -> List[SearchResult]:
    results: List[SearchResult] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    SearchResult(
                        title=r.get("title", "Untitled"),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    )
                )
    except Exception as e:
        print(f"[Web search error for '{query}']: {e}")
    return results


# -------------------------------
# Arxiv Search (Research Papers)
# -------------------------------
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Returns summarized academic paper results from Arxiv, as a single
    formatted string (kept as text output to match the previous
    ArxivQueryRun.run() interface, in case anything downstream expects
    a string rather than structured results).
    """
    if not _ARXIV_AVAILABLE:
        return "Arxiv error: the 'arxiv' package is not installed. Run: pip install arxiv"
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        entries = []
        for result in _arxiv_call_with_backoff(search):
            authors = ", ".join(a.name for a in result.authors) or "Unknown authors"
            summary = (result.summary or "").strip().replace("\n", " ")
            if len(summary) > 400:
                summary = summary[:400].rstrip() + "..."
            entries.append(
                f"Title: {result.title}\n"
                f"Authors: {authors}\n"
                f"Published: {result.published.date() if result.published else 'unknown'}\n"
                f"Summary: {summary}\n"
                f"URL: {result.entry_id}"
            )
        return "\n\n".join(entries) if entries else "No results found."
    except Exception as e:
        return f"Arxiv error: {e}"


def search_arxiv_structured(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Same Arxiv search as search_arxiv(), but returns a List[SearchResult]
    instead of a formatted string -- for use with agent.py's paper-mode
    retrieval, which expects SearchResult objects (title/url/snippet) like
    web_search() returns, not a single text blob.
    """
    if not _ARXIV_AVAILABLE:
        print("[Arxiv error] the 'arxiv' package is not installed. Run: pip install arxiv")
        return []
    results: List[SearchResult] = []
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in _arxiv_call_with_backoff(search):
            summary = (result.summary or "").strip().replace("\n", " ")
            if len(summary) > 300:
                summary = summary[:300].rstrip() + "..."
            results.append(
                SearchResult(
                    title=result.title,
                    url=result.entry_id,
                    snippet=summary,
                )
            )
    except Exception as e:
        print(f"[Arxiv search error for '{query}']: {e}")
    return results