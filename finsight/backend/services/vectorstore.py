"""
ChromaDB wrapper.

Responsibilities:
- Dedup documents by doc_id (no re-ingestion)
- Upsert chunks with metadata
- Similarity search with optional doc_id filter
- List all ingested documents
- Delete document by doc_id
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Optional, Tuple
import chromadb

from core.config import settings
from services.ingestion import load_and_split


class VectorStoreService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.db = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    def ingest(self, file_bytes: bytes, filename: str) -> dict:
        chunks, doc_id = load_and_split(file_bytes, filename)

        # Dedup check
        existing = self.db.get(where={"doc_id": doc_id})
        if existing["ids"]:
            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunks_created": len(existing["ids"]),
                "status": "already_exists",
            }

        # Add to ChromaDB
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "status": "ingested",
        }

    def similarity_search(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """Returns (doc, relevance_score) pairs. Score 0–1, higher = more relevant."""
        k = k or settings.top_k_retrieval
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None

        results = self.db.similarity_search_with_relevance_scores(
            query, k=k, filter=where
        )
        # Filter out low-relevance chunks
        return [
            (doc, score)
            for doc, score in results
            if score >= settings.min_relevance_score
        ]

    def list_documents(self) -> List[dict]:
        """Return unique documents (by doc_id) stored in ChromaDB."""
        all_meta = self.db.get()["metadatas"]
        seen = {}
        for m in all_meta:
            if m and m.get("doc_id") not in seen:
                seen[m["doc_id"]] = {
                    "doc_id": m["doc_id"],
                    "filename": m.get("filename", "unknown"),
                    "chunk_count": 0,
                    "ingested_at": m.get("ingested_at", ""),
                }
            if m and m.get("doc_id") in seen:
                seen[m["doc_id"]]["chunk_count"] += 1
        return list(seen.values())

    def delete_document(self, doc_id: str) -> int:
        existing = self.db.get(where={"doc_id": doc_id})
        if not existing["ids"]:
            return 0
        self.db.delete(ids=existing["ids"])
        return len(existing["ids"])
