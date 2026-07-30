import chromadb
from chromadb.utils import embedding_functions
from tools import SearchResult

_client = chromadb.PersistentClient(path="./chroma_db")
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_collection = _client.get_or_create_collection(
    name="research_knowledge", embedding_function=_embed_fn
)


def query_chroma(query: str, n_results: int = 5) -> list[SearchResult]:
    res = _collection.query(query_texts=[query], n_results=n_results)
    out = []
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    for doc, meta in zip(docs[0], metas[0]):
        out.append(
            SearchResult(
                title=meta.get("title", ""),
                url=meta.get("url", ""),
                snippet=(doc or "")[:300],
            )
        )
    return out


def add_to_chroma(sources: list[SearchResult]):
    """
    Persist retrieved sources into the vector store so future queries can
    hit them via semantic search.

    Sources are keyed by URL. Two safeguards here:
      1. De-duplicate by URL before writing -- the same URL can appear
         more than once in a single `sources` list (e.g. surfaced by two
         different search queries with slightly different titles), and
         Chroma's `add`/`upsert` calls require unique IDs *within one
         call*, not just against what's already stored.
      2. Use `upsert` instead of `add` -- `add` raises if an ID already
         exists in the collection at all (i.e. from a previous run);
         `upsert` overwrites it instead, which is what we want since a
         newer snippet/title for the same URL is fine to replace the old
         one with.
    """
    if not sources:
        return

    deduped: dict[str, SearchResult] = {}
    for s in sources:
        if not s.url:
            continue
        deduped[s.url] = s  # last occurrence wins; fine for our purposes

    if not deduped:
        return

    ids = list(deduped.keys())
    documents = [s.snippet for s in deduped.values()]
    metadatas = [{"title": s.title, "url": s.url} for s in deduped.values()]

    _collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )