"""
Hybrid retrieval tools for the Research Scientist Agent.
Includes:
1. DuckDuckGo (DDGS) -> general web search
2. Arxiv        -> academic research papers (via the `arxiv` package
                    directly, not LangChain's ArxivQueryRun wrapper --
                    that class has moved between langchain / langchain_community
                    across versions and kept breaking imports; calling the
                    underlying `arxiv` library directly avoids that churn).
"""
from dataclasses import dataclass
from typing import List

from ddgs import DDGS

try:
    import arxiv
    _ARXIV_AVAILABLE = True
except ImportError:
    _ARXIV_AVAILABLE = False


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
        for result in search.results():
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
        for result in search.results():
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