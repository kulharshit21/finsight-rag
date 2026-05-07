"""
PDF ingestion pipeline.

Flow:
  raw PDF bytes
    → PyPDFLoader (page-aware)
    → RecursiveCharacterTextSplitter (financial-tuned separators)
    → metadata injection (doc_id, filename, page, ingested_at)
    → OpenAI embeddings
    → ChromaDB persist
"""

import hashlib
import tempfile
import os
from datetime import datetime
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from core.config import settings


FINANCIAL_SEPARATORS = [
    "\n\n",       # paragraph breaks
    "\n",          # line breaks
    ". ",          # sentence end
    ", ",          # clause end
    " ",           # word
    "",            # char fallback
]


def compute_doc_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def load_and_split(file_bytes: bytes, filename: str) -> Tuple[List[Document], str]:
    """Load PDF and return LangChain Document chunks with rich metadata."""

    doc_id = compute_doc_id(file_bytes)
    ingested_at = datetime.utcnow().isoformat()

    # Write to temp file (PyPDFLoader needs a path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        raw_pages = loader.load()  # one Document per page
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=FINANCIAL_SEPARATORS,
    )

    chunks = splitter.split_documents(raw_pages)

    # Inject metadata into every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "ingested_at": ingested_at,
            # page already set by PyPDFLoader
        })

    return chunks, doc_id
