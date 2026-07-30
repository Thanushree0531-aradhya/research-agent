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
"""

import os
from typing import Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
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

_eval_llm = None
_eval_embeddings = None


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


def evaluate_answer(question: str, answer: str, contexts: List[str]) -> Dict[str, Optional[float]]:
    """
    Scores a single (question, answer, contexts) triple with RAGAS.

    Returns a dict like {"faithfulness": 0.91, "answer_relevancy": 0.87}.
    Returns an empty dict (rather than raising) if evaluation can't run --
    e.g. no contexts were actually cited, or the RAGAS/LLM call itself
    fails -- so a scoring hiccup never breaks the main answer pipeline.
    """
    contexts = [c for c in contexts if c and c.strip()]
    if not contexts or not answer.strip():
        return {}

    try:
        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=_get_eval_llm(),
            embeddings=_get_eval_embeddings(),
        )
        df = result.to_pandas()
        row = df.iloc[0]

        scores: Dict[str, Optional[float]] = {}
        for metric_name in ("faithfulness", "answer_relevancy"):
            if metric_name in row and row[metric_name] == row[metric_name]:  # filters NaN
                scores[metric_name] = round(float(row[metric_name]), 2)
        return scores
    except Exception as e:
        print(f"  [RAGAS] evaluation failed, skipping quality scores: {e}")
        return {}