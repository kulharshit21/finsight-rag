"""
RAGAS-based answer quality evaluation.

Two metrics are computed per query and shown to the user in real time:
  - faithfulness      (0–1): is every claim in the answer supported by the retrieved context?
  - answer_relevancy  (0–1): does the answer actually address the question asked?

Both are also logged to MLflow so quality can be tracked across many queries
and regressions detected when the prompt or retrieval strategy changes.
"""

from typing import List, Tuple

import mlflow
from datasets import Dataset
from langchain.schema import Document
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from core.config import settings


def evaluate_response(
    question: str,
    answer: str,
    docs_scores: List[Tuple[Document, float]],
) -> Tuple[float, float]:
    """
    Run RAGAS and return (faithfulness_score, answer_relevancy_score).
    Falls back to (0.5, 0.5) if RAGAS itself throws (e.g. API quota).
    """
    if not docs_scores:
        return 0.0, 0.0

    contexts = [[doc.page_content for doc, _ in docs_scores]]

    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": contexts,
    })

    try:
        result = evaluate(dataset=data, metrics=[faithfulness, answer_relevancy])
        f_score = float(result["faithfulness"])
        ar_score = float(result["answer_relevancy"])
    except Exception:
        f_score, ar_score = 0.5, 0.5

    # Non-critical: log to MLflow for historical tracking
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        with mlflow.start_run(run_name="finsight_query", nested=True):
            mlflow.log_metric("faithfulness", f_score)
            mlflow.log_metric("answer_relevancy", ar_score)
            mlflow.log_param("question_len", len(question))
            mlflow.log_param("num_sources", len(docs_scores))
    except Exception:
        pass

    return f_score, ar_score
