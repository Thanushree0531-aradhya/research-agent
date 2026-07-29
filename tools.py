"""
Web search tool for the Research Scientist Agent.

Wraps DDGS (DuckDuckGo Search) so the agent can retrieve real, live
information from the web. No API key required for search itself.
"""

from dataclasses import dataclass
from typing import List
from ddgs import DDGS


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_citation_line(self, idx: int) -> str:
        return f"[{idx}] {self.title} — {self.url}"


def web_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Run a live web search and return structured results.
    Fails soft: returns an empty list instead of raising, so one bad
    query doesn't crash the whole research pipeline.
    """
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
        print(f"  [search error for '{query}']: {e}")
    return results