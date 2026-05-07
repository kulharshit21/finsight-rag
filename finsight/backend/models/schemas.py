from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_created: int
    status: str  # "ingested" | "already_exists" | "failed"
    ingested_at: datetime


class Source(BaseModel):
    page: int
    filename: str
    excerpt: str  # first 200 chars of the chunk


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    doc_ids: Optional[List[str]] = None  # filter to specific docs; None = all


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    faithfulness_score: float       # RAGAS faithfulness 0–1
    answer_relevancy_score: float   # RAGAS answer_relevancy 0–1
    fraud_risk_score: Optional[float]  # XGBoost 0–1, None if no model loaded
    fraud_flags: List[str]          # human-readable flags triggered
    latency_ms: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    ingested_at: str


class ListDocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int
