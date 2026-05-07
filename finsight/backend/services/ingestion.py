"""
PDF ingestion pipeline.

Financial documents contain dense tables, irregular whitespace, and runs of
numeric data. The splitter separators and chunk sizes below were tuned for
this specific structure — paragraph breaks first, then sentences, then words.

Flow:
  raw PDF bytes → SHA-256 dedup ID → PyPDFLoader (page-aware)
  → RecursiveCharacterTextSplitter → metadata injection
"""

import hashlib
import tempfile
import os
from datetime import datetime
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import settings


# Ordered from coarsest to finest — keeps table rows and paragraphs together
FINANCIAL_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]


def compute_doc_id(file_bytes: bytes) -> str:
    """16-char SHA-256 prefix — unique per file content, enables dedup."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def load_and_split(file_bytes: bytes, filename: str) -> Tuple[List[Document], str]:
    """
    Load a PDF and return (chunks, doc_id).
    Each chunk carries full provenance metadata so any answer can be cited.
    """
    doc_id = compute_doc_id(file_bytes)
    ingested_at = datetime.utcnow().isoformat()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()          # one Document per page
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=FINANCIAL_SEPARATORS,
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "ingested_at": ingested_at,
            # "page" key already present from PyPDFLoader
        })

    return chunks, doc_id
