"""
Regression tests for agent.py.

Run with:
    pip install pytest
    pytest test_agent.py -v

These tests use fake/mock LangChain, LangGraph, and tools modules (no real
network or API calls, no GROQ_API_KEY needed) so they can run anywhere,
fast, and catch regressions like the ones we hit manually during
development:
    - long paper titles being truncated by regex length caps
    - topic drift across the verify retry loop
    - duplicate papers slipping into the reference list
    - non-paper content (blogs/guides) leaking into paper mode
    - arXiv queries accidentally going through DuckDuckGo instead of the
      real Arxiv API
"""
import sys
import types
import importlib
import pytest


# ---------------------------------------------------------------------------
# Fake dependencies, installed into sys.modules BEFORE agent.py is imported.
# ---------------------------------------------------------------------------

class FakeSearchResult:
    def __init__(self, title, url, snippet=""):
        self.title, self.url, self.snippet = title, url, snippet


class FakeResponse:
    def __init__(self, content):
        self.content = content


class ScriptedChatGroq:
    """
    A fake ChatGroq whose .invoke() behavior is driven by a dict of
    {substring_in_prompt: response_text_or_callable}, checked in order.
    Falls back to a generic default if nothing matches.
    """
    def __init__(self, *a, **k):
        self.model = k.get("model", "unknown")

    def invoke(self, prompt):
        script = _CURRENT_SCRIPT.get(self.model, _CURRENT_SCRIPT.get("*", {}))
        for marker, response in script.items():
            if marker in prompt:
                content = response(prompt) if callable(response) else response
                return FakeResponse(content)
        # sensible generic defaults
        if "Respond with ONLY a JSON array" in prompt:
            return FakeResponse('["generic query"]')
        if "Respond with ONLY JSON in this exact shape" in prompt:
            return FakeResponse('{"verified": true, "notes": "", "additional_search_queries": []}')
        return FakeResponse("Generic draft answer [1].")


_CURRENT_SCRIPT = {}  # test cases populate this before calling run_research_agent


END = "END"


class FakeStateGraph:
    def __init__(self, *a, **k):
        self.nodes, self.edges, self.conditional, self.entry = {}, [], None, None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, a, b):
        self.edges.append((a, b))

    def add_conditional_edges(self, src, cond_fn, mapping):
        self.conditional = (src, cond_fn, mapping)

    def compile(self):
        return FakeCompiledGraph(self)


class FakeCompiledGraph:
    def __init__(self, g):
        self.g = g

    def invoke(self, state, config=None):
        node = self.g.entry
        while node != END:
            state = self.g.nodes[node](state)
            if self.g.conditional and self.g.conditional[0] == node:
                _, cond_fn, mapping = self.g.conditional
                node = mapping[cond_fn(state)]
            else:
                nxt = [b for (a, b) in self.g.edges if a == node]
                node = nxt[0] if nxt else END
        return state


_WEB_SEARCH_CALLS = []
_ARXIV_CALLS = []
_FAKE_WEB_RESULTS = {}    # substring-in-query -> list[FakeSearchResult]
_FAKE_ARXIV_RESULTS = []  # list[FakeSearchResult], returned for any arxiv query


def _fake_web_search(query, max_results=4):
    _WEB_SEARCH_CALLS.append(query)
    for marker, results in _FAKE_WEB_RESULTS.items():
        if marker in query:
            return results[:max_results]
    return []


def _fake_search_arxiv_structured(query, max_results=5):
    _ARXIV_CALLS.append(query)
    return list(_FAKE_ARXIV_RESULTS)[:max_results]


@pytest.fixture(autouse=True)
def _reset_and_install_fakes(monkeypatch):
    """Runs before every test: resets call logs/scripts and (re)installs fakes."""
    _WEB_SEARCH_CALLS.clear()
    _ARXIV_CALLS.clear()
    _FAKE_WEB_RESULTS.clear()
    _FAKE_ARXIV_RESULTS.clear()
    _CURRENT_SCRIPT.clear()

    sys.modules['langchain_groq'] = types.SimpleNamespace(ChatGroq=ScriptedChatGroq)
    graph_mod = types.SimpleNamespace(StateGraph=FakeStateGraph, END=END)
    sys.modules['langgraph'] = types.SimpleNamespace(graph=graph_mod)
    sys.modules['langgraph.graph'] = graph_mod
    sys.modules['tools'] = types.SimpleNamespace(
        web_search=_fake_web_search,
        SearchResult=FakeSearchResult,
        search_arxiv_structured=_fake_search_arxiv_structured,
    )

    global agent
    if 'agent' in sys.modules:
        importlib.reload(sys.modules['agent'])
    import agent as _agent
    agent = _agent
    yield


# ---------------------------------------------------------------------------
# extract_paper_intent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected_mode", [
    ("Explain how LangChain agents work", "none"),
    ("Explain how LangChain agents work from 2 research papers", "count"),
    ("Compare LangChain vs LlamaIndex from three studies", "count"),
    ("Explain how LangChain agents work based on research papers", "broad"),
    ("What is retrieval augmented generation from a single paper", "count"),
    ('Tell me about the paper called "Attention Is All You Need"', "specific"),
    ("Summarize the 'ReAct: Synergizing Reasoning and Acting' paper", "specific"),
    # long title (104 chars) -- regression test for the length-cap bug
    (
        'Tell me about the paper called "SRAG: Structured Retrieval-Augmented '
        'Generation for Multi-Entity Question Answering over Wikipedia Graph"',
        "specific",
    ),
])
def test_extract_paper_intent_mode(question, expected_mode):
    mode, count, title = agent.extract_paper_intent(question)
    assert mode == expected_mode, f"expected mode={expected_mode!r}, got {mode!r} for: {question!r}"


def test_extract_paper_intent_broad_count_target():
    _, count, _ = agent.extract_paper_intent("Explain X based on research papers")
    assert count == agent.BROAD_MODE_TARGET


def test_extract_paper_intent_count_value():
    _, count, _ = agent.extract_paper_intent("Explain X from 3 research papers")
    assert count == 3


def test_extract_paper_intent_specific_title_not_truncated():
    long_title = (
        "SRAG: Structured Retrieval-Augmented Generation for Multi-Entity "
        "Question Answering over Wikipedia Graph"
    )
    assert len(long_title) > 100  # sanity check this is actually a long title
    _, _, title = agent.extract_paper_intent(f'Tell me about the paper called "{long_title}"')
    assert title == long_title


# ---------------------------------------------------------------------------
# extract_topic_entity
# ---------------------------------------------------------------------------

def test_topic_entity_prefers_quoted_term():
    assert agent.extract_topic_entity('Explain "LangGraph" internals') == "LangGraph"


def test_topic_entity_long_quoted_title_not_truncated():
    long_title = (
        "SRAG: Structured Retrieval-Augmented Generation for Multi-Entity "
        "Question Answering over Wikipedia Graph"
    )
    assert len(long_title) > 60  # sanity check: exceeds the old buggy cap
    topic = agent.extract_topic_entity(f'Summarize the "{long_title}" paper')
    assert topic == long_title


def test_topic_entity_strips_sentence_starting_verb():
    topic = agent.extract_topic_entity("Compare LangChain vs LlamaIndex")
    assert topic in ("LangChain", "LlamaIndex", "LlamaIndex vs LangChain")
    assert "Compare" not in (topic or "")


def test_topic_entity_no_capitalized_terms_returns_none():
    assert agent.extract_topic_entity("what is retrieval augmented generation") is None


# ---------------------------------------------------------------------------
# dedup / non-paper filter
# ---------------------------------------------------------------------------

def test_dedup_key_collapses_mirrored_titles():
    a = FakeSearchResult("Understanding LangChain Agents - arXiv", "https://arxiv.org/abs/1")
    b = FakeSearchResult("Understanding LangChain Agents | Semantic Scholar", "https://semanticscholar.org/paper/2")
    assert agent._dedup_key(a) == agent._dedup_key(b)


def test_dedup_key_distinct_papers_differ():
    a = FakeSearchResult("Paper A", "https://arxiv.org/abs/1")
    b = FakeSearchResult("Paper B", "https://arxiv.org/abs/2")
    assert agent._dedup_key(a) != agent._dedup_key(b)


@pytest.mark.parametrize("title,url,expected", [
    ("RAG Techniques Overview Blog", "https://semanticscholar.org/blog/rag-overview", False),
    ("What is RAG? A Beginner Guide", "https://atlan.com/know/what-is-rag/", False),
    ("An Introduction to Retrieval-Augmented Generation", "https://arxiv.org/abs/2601.00042", True),
    ("SRAG: Structured Retrieval-Augmented Generation", "https://arxiv.org/html/2503.01346v2", True),
    ("Top 10 RAG Tools in 2026", "https://dl.acm.org/doi/10.1145/999999", True),  # DOI overrides title signal
])
def test_looks_like_paper(title, url, expected):
    src = FakeSearchResult(title, url)
    assert agent._looks_like_paper(src) == expected


# ---------------------------------------------------------------------------
# rate-limit / retry-after parsing
# ---------------------------------------------------------------------------

def test_parse_retry_after_minutes_and_seconds():
    msg = "... Please try again in 11m50.208s."
    wait = agent._parse_retry_after(msg)
    assert wait is not None
    assert 700 < wait < 720


def test_parse_retry_after_seconds_only():
    assert agent._parse_retry_after("try again in 30s") == pytest.approx(31.0)


def test_parse_retry_after_no_match_returns_none():
    assert agent._parse_retry_after("some unrelated error") is None


def test_is_daily_quota_error_detection():
    assert agent._is_daily_quota_error("... on tokens per day (TPD): Limit 100000 ...")
    assert not agent._is_daily_quota_error("... requests per minute exceeded ...")


# ---------------------------------------------------------------------------
# End-to-end mocked runs: one per paper mode
# ---------------------------------------------------------------------------

def test_e2e_broad_mode_uses_real_arxiv_api_not_ddg():
    _CURRENT_SCRIPT["*"] = {
        "JSON array": '["LangChain agent architecture"]',
        "JSON in this exact shape": '{"verified": true, "notes": "", "additional_search_queries": []}',
    }
    _FAKE_WEB_RESULTS["semanticscholar.org"] = [
        FakeSearchResult("LangChain Mirror | Semantic Scholar", "https://semanticscholar.org/paper/x", "s")
    ]
    _FAKE_ARXIV_RESULTS.extend([
        FakeSearchResult("LangChain Agents: A Survey", "https://arxiv.org/abs/2401.00001", "real api result"),
    ])

    result = agent.run_research_agent("Explain how LangChain agents work based on research papers")

    assert all("arxiv.org" not in c for c in _WEB_SEARCH_CALLS), "DDG should never be queried for arxiv.org"
    assert len(_ARXIV_CALLS) >= 1, "the real arxiv API should have been called"
    assert isinstance(result, str) and len(result) > 0


def test_e2e_count_mode_cites_exact_arxiv_paper():
    _CURRENT_SCRIPT["*"] = {
        "JSON array": '["retrieval augmented generation"]',
        "JSON in this exact shape": '{"verified": true, "notes": "", "additional_search_queries": []}',
        "Write the draft answer now": "RAG combines retrieval and generation [1].",
    }
    _FAKE_ARXIV_RESULTS.append(
        FakeSearchResult("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                          "https://arxiv.org/abs/2005.11401", "the original RAG paper")
    )

    result = agent.run_research_agent("Explain RAG from 1 research paper")
    assert "arxiv.org/abs/2005.11401" in result


def test_e2e_specific_mode_ignores_unrelated_papers():
    target_title = "SRAG: Structured Retrieval-Augmented Generation over Wikipedia Graph"
    _CURRENT_SCRIPT["*"] = {
        "JSON array": f'["{target_title}"]',
        "JSON in this exact shape": '{"verified": true, "notes": "", "additional_search_queries": []}',
        "Write the draft answer now": f"The paper '{target_title}' proposes structured retrieval [1].",
    }
    _FAKE_ARXIV_RESULTS.extend([
        FakeSearchResult(target_title, "https://arxiv.org/abs/2503.01346", "the target paper"),
        FakeSearchResult("An unrelated RAG survey", "https://arxiv.org/abs/9999.99999", "should not be cited"),
    ])

    result = agent.run_research_agent(f'Tell me about the paper called "{target_title}"')
    assert "2503.01346" in result
    assert "9999.99999" not in result


def test_e2e_topic_lock_across_iterations():
    """
    Verify-loop gap queries must still contain the locked topic, even if
    the (fake) verifier LLM 'forgets' to include it.
    """
    calls_seen = {"count": 0}

    def verify_response(prompt):
        calls_seen["count"] += 1
        if calls_seen["count"] == 1:
            # first verify: not satisfied, propose a topic-less gap query
            return '{"verified": false, "notes": "need more evidence", "additional_search_queries": ["agent frameworks comparison"]}'
        return '{"verified": true, "notes": "", "additional_search_queries": []}'

    _CURRENT_SCRIPT["*"] = {
        "JSON array": '["LangChain overview"]',
        "Respond with ONLY JSON in this exact shape": verify_response,
    }

    result = agent.run_research_agent("Explain LangChain")
    # pending_queries after the first verify should have been topic-locked
    assert any("LangChain" in c for c in _WEB_SEARCH_CALLS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))