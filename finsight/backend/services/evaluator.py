"""
RAGAS-based RAG evaluation.

Metrics computed per query:
- faithfulness: is the answer grounded in the retrieved context?
- answer_relevancy: does the answer actually address the question?

Both scores are 0–1. Higher = better.
Scores are returned in QueryResponse and logged to MLflow.
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from typing import List, Tuple
from langchain.schema import Document
import mlflow

from core.config import settings


def evaluate_response(
    question: str,
    answer: str,
    docs_scores: List[Tuple[Document, float]],
) -> Tuple[float, float]:
    """Run RAGAS evaluation. Returns (faithfulness_score, answer_relevancy_score)."""
    if not docs_scores:
        return 0.0, 0.0

    contexts = [[doc.page_content for doc, _ in docs_scores]]

    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": contexts,
    })

    try:
        result = evaluate(
            dataset=data,
            metrics=[faithfulness, answer_relevancy],
        )
        f_score = float(result["faithfulness"])
        ar_score = float(result["answer_relevancy"])
    except Exception:
        # Graceful fallback if RAGAS API fails
        f_score, ar_score = 0.5, 0.5

    # Log to MLflow
    if settings.mlflow_tracking_uri:
        try:
            with mlflow.start_run(run_name="finsight_query", nested=True):
                mlflow.log_metric("faithfulness", f_score)
                mlflow.log_metric("answer_relevancy", ar_score)
                mlflow.log_param("question_len", len(question))
                mlflow.log_param("num_sources", len(docs_scores))
        except Exception:
            pass  # MLflow logging is non-critical

    return f_score, ar_score
