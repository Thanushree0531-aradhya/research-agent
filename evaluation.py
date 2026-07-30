"""
RAGAS-based answer quality evaluation.

Computes two reference-free metrics after each research run:

  * Faithfulness      -- how much of the answer is actually supported by
                          the cited source snippets (catches hallucination
                          that slipped past the agent's own verify_node).
  * Answer Relevancy  -- how well the answer actually addresses the
                          original question (catches on-topic-but-useless
                          answers, e.g. restating the question back).

Both metrics work without a ground-truth/reference answer, which matters
here since this agent answers arbitrary user questions with no pre-written
"correct" answer to compare against.

Uses the same Groq LLM and the same HuggingFace embedding model already
used elsewhere in this project (vector_store.py), wrapped for RAGAS via
its Langchain adapters, so no second embedding model needs to be
downloaded.

Rate limiting
-------------
Faithfulness decomposes the answer into individual statements and
verifies each one against the contexts with its own LLM call -- for a
typical answer that's easily 5-10+ Groq calls in a single evaluate()
run, on top of everything plan/retrieve/analyze/verify already used this
turn. Two things guard against that compounding the 429 problem seen
elsewhere in this pipeline:
  1. A RunConfig is passed to evaluate() with retry/backoff settings
     (RAGAS's defaults aren't tuned to Groq's limits at all), so a 429
     during scoring gets retried instead of silently dropping the
     metrics for that run.
  2. RunConfig.max_workers is capped low so RAGAS doesn't fire its
     per-statement checks concurrently right after verify_node already
     hammered the same model.

Schema compatibility
---------------------
RAGAS <0.2 expects dataset columns named question/answer/contexts.
RAGAS 0.2+ renamed these to user_input/response/retrieved_contexts.
_build_dataset tries the legacy names first (matching this project's
pinned version at time of writing) and falls back to the new names if
that raises a schema-related error, so this keeps working across a
version bump without silently returning {} on every call.
"""

import os
from typing import Dict, List, Optional

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

EVAL_MODEL_NAME = os.environ.get("RESEARCH_AGENT_EVAL_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# RAGAS's answer_relevancy metric defaults to strictness=3, which makes it
# request n=3 completions per call for self-consistency scoring. Groq's
# API rejects any n > 1 ("'n' : number must be at most 1"), which is what
# was silently dropping this metric from the final scores. Building our
# own AnswerRelevancy instance with strictness=1 keeps the metric's
# self-consistency behavior off (single sample) instead of sampling
# multiple completions, which Groq doesn't support.
answer_relevancy = AnswerRelevancy(strictness=1)

# Tuned for Groq: retry on transient failures (429s included) instead of
# RAGAS's default retry behavior, and keep concurrency low so scoring
# doesn't fire a burst of parallel calls right after the agent's own
# plan/verify steps already used the same rate-limited model.
EVAL_MAX_RETRIES = int(os.environ.get("RESEARCH_AGENT_EVAL_MAX_RETRIES", "4"))
EVAL_MAX_WAIT_SECONDS = int(os.environ.get("RESEARCH_AGENT_EVAL_MAX_WAIT_SECONDS", "60"))
EVAL_TIMEOUT_SECONDS = int(os.environ.get("RESEARCH_AGENT_EVAL_TIMEOUT_SECONDS", "180"))
EVAL_MAX_WORKERS = int(os.environ.get("RESEARCH_AGENT_EVAL_MAX_WORKERS", "2"))

_eval_llm = None
_eval_embeddings = None
_eval_run_config = None


def _get_eval_llm():
    global _eval_llm
    if _eval_llm is None:
        _eval_llm = LangchainLLMWrapper(ChatGroq(model=EVAL_MODEL_NAME, temperature=0))
    return _eval_llm


def _get_eval_embeddings():
    global _eval_embeddings
    if _eval_embeddings is None:
        _eval_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        )
    return _eval_embeddings


def _get_run_config():
    """
    RunConfig controls RAGAS's own internal retry/backoff/concurrency
    for the many LLM calls a single evaluate() pass can make. Built lazily
    once and reused, same pattern as the LLM/embeddings singletons above.
    """
    global _eval_run_config
    if _eval_run_config is None:
        _eval_run_config = RunConfig(
            timeout=EVAL_TIMEOUT_SECONDS,
            max_retries=EVAL_MAX_RETRIES,
            max_wait=EVAL_MAX_WAIT_SECONDS,
            max_workers=EVAL_MAX_WORKERS,
        )
    return _eval_run_config


def _build_dataset(question: str, answer: str, contexts: List[str]) -> Dataset:
    """
    Builds the single-row Dataset RAGAS scores. Tries the legacy
    question/answer/contexts column names first; if the installed RAGAS
    version has moved to user_input/response/retrieved_contexts instead,
    evaluate() will raise a KeyError/ValueError naming the missing
    column, and the caller falls back to the new schema rather than
    failing outright.
    """
    return Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
    )


def _build_dataset_new_schema(question: str, answer: str, contexts: List[str]) -> Dataset:
    """RAGAS 0.2+ column names, used as a fallback -- see _build_dataset."""
    return Dataset.from_dict(
        {
            "user_input": [question],
            "response": [answer],
            "retrieved_contexts": [contexts],
        }
    )


def _looks_like_schema_error(error: Exception) -> bool:
    """
    Distinguishes "wrong column names for this RAGAS version" from other
    failures (network errors, genuine 429 exhaustion, etc.) so we only
    retry with the alternate schema when it's actually a schema mismatch,
    rather than masking a real error behind a second failed attempt.
    """
    msg = str(error).lower()
    schema_hints = ("question", "answer", "contexts", "user_input", "response",
                     "retrieved_contexts", "column", "keyerror")
    return any(hint in msg for hint in schema_hints)


def evaluate_answer(question: str, answer: str, contexts: List[str]) -> Dict[str, Optional[float]]:
    """
    Scores a single (question, answer, contexts) triple with RAGAS.

    Returns a dict like {"faithfulness": 0.91, "answer_relevancy": 0.87}.
    Returns an empty dict (rather than raising) if evaluation can't run --
    e.g. no contexts were actually cited, or the RAGAS/LLM call itself
    fails even after RunConfig's retries -- so a scoring hiccup never
    breaks the main answer pipeline.
    """
    contexts = [c for c in contexts if c and c.strip()]
    if not contexts or not answer.strip():
        return {}

    run_config = _get_run_config()
    llm = _get_eval_llm()
    embeddings = _get_eval_embeddings()
    metrics = [faithfulness, answer_relevancy]

    dataset = _build_dataset(question, answer, contexts)
    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
        )
    except Exception as e:
        if not _looks_like_schema_error(e):
            print(f"  [RAGAS] evaluation failed, skipping quality scores: {e}")
            return {}
        # Legacy column names didn't match what this RAGAS version
        # expects -- retry once with the 0.2+ schema before giving up.
        try:
            dataset = _build_dataset_new_schema(question, answer, contexts)
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
            )
        except Exception as e2:
            print(f"  [RAGAS] evaluation failed under both column schemas, skipping quality scores: {e2}")
            return {}

    try:
        df = result.to_pandas()
        row = df.iloc[0]

        scores: Dict[str, Optional[float]] = {}
        for metric_name in ("faithfulness", "answer_relevancy"):
            if metric_name in row and row[metric_name] == row[metric_name]:  # filters NaN
                scores[metric_name] = round(float(row[metric_name]), 2)
        return scores
    except Exception as e:
        print(f"  [RAGAS] failed to extract scores from result, skipping quality scores: {e}")
        return {}