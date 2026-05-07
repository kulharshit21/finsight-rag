from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Mistral
    mistral_api_key: str
    llm_model: str = "mistral-large-latest"
    embedding_model: str = "mistral-embed"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "financial_docs"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    min_relevance_score: float = 0.3

    # Evaluation (RAGAS)
    run_evaluation: bool = True

    # Fraud scoring
    fraud_model_path: Optional[str] = None
    fraud_threshold: float = 0.65

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    class Config:
        env_file = ".env"


settings = Settings()
