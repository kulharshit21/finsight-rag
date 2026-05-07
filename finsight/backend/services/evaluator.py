"""
RAGAS-based answer quality evaluation.

Imports are deferred to call time so a broken RAGAS/instructor install never
blocks server startup. If RAGAS fails for any reason the function falls back
to neutral (0.5, 0.5) scores — evaluation is non-critical.
"""

from typing import List, Tuple

from langchain_core.documents import Document

from core.config import settings


def evaluate_response(
    question: str,
    answer: str,
    docs_scores: List[Tuple[Document, float]],
) -> Tuple[float, float]:
    """
    Returns (faithfulness_score, answer_relevancy_score), both 0–1.
    Falls back to (0.5, 0.5) on any error.
    """
    if not docs_scores:
        return 0.0, 0.0

    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
        from datasets import Dataset

        contexts = [[doc.page_content for doc, _ in docs_scores]]
        data = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": contexts,
        })
        result = evaluate(dataset=data, metrics=[faithfulness, answer_relevancy])
        f_score = float(result["faithfulness"])
        ar_score = float(result["answer_relevancy"])
    except Exception:
        f_score, ar_score = 0.5, 0.5

    # Non-critical: log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        with mlflow.start_run(run_name="finsight_query", nested=True):
            mlflow.log_metric("faithfulness", f_score)
            mlflow.log_metric("answer_relevancy", ar_score)
            mlflow.log_param("question_len", len(question))
            mlflow.log_param("num_sources", len(docs_scores))
    except Exception:
        pass

    return f_score, ar_score
