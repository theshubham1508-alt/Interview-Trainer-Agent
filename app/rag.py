"""
RAG pipeline: document ingestion, chunking, embedding, and retrieval.
Uses ChromaDB as the local vector store and sentence-transformers for embeddings.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL,
    RAG_TOP_K,
    VECTOR_STORE_PATH,
)

logger = logging.getLogger(__name__)

# ── Embedding model (cached after first load) ─────────────────────────────────
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── Vector store (singleton) ──────────────────────────────────────────────────
_vector_store: Chroma | None = None


def get_vector_store(force_reload: bool = False) -> Chroma:
    global _vector_store
    if _vector_store is None or force_reload:
        _vector_store = Chroma(
            collection_name="interview_knowledge_base",
            embedding_function=get_embeddings(),
            persist_directory=VECTOR_STORE_PATH,
        )
    return _vector_store


# ── Document ingestion helpers ────────────────────────────────────────────────

def _load_json_interview_data(filepath: Path) -> List[Document]:
    """Load structured JSON interview Q&A data into Documents."""
    docs: List[Document] = []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for entry in data:
            content_parts = []
            if "role" in entry:
                content_parts.append(f"Role: {entry['role']}")
            if "category" in entry:
                content_parts.append(f"Category: {entry['category']}")
            if "question" in entry:
                content_parts.append(f"Question: {entry['question']}")
            if "answer" in entry:
                content_parts.append(f"Model Answer: {entry['answer']}")
            if "tips" in entry:
                content_parts.append(f"Tips: {entry['tips']}")

            content = "\n".join(content_parts)
            if content.strip():
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(filepath),
                            "role": entry.get("role", "general"),
                            "category": entry.get("category", "general"),
                            "type": "structured_qa",
                        },
                    )
                )
    return docs


def ingest_documents(documents_path: str = DOCUMENTS_PATH) -> int:
    """
    Ingest all documents from the documents directory into the vector store.
    Supports: PDF, TXT, JSON interview data files.
    Returns the number of chunks added.
    """
    docs_dir = Path(documents_path)
    all_docs: List[Document] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Load PDFs
    for pdf_file in docs_dir.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)
            logger.info("Loaded PDF: %s (%d chunks)", pdf_file.name, len(chunks))
        except Exception as e:
            logger.warning("Failed to load PDF %s: %s", pdf_file, e)

    # Load plain text files
    for txt_file in docs_dir.glob("**/*.txt"):
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)
            logger.info("Loaded TXT: %s (%d chunks)", txt_file.name, len(chunks))
        except Exception as e:
            logger.warning("Failed to load TXT %s: %s", txt_file, e)

    # Load structured JSON interview data
    for json_file in docs_dir.glob("**/*.json"):
        try:
            json_docs = _load_json_interview_data(json_file)
            all_docs.extend(json_docs)
            logger.info("Loaded JSON: %s (%d entries)", json_file.name, len(json_docs))
        except Exception as e:
            logger.warning("Failed to load JSON %s: %s", json_file, e)

    if all_docs:
        vs = get_vector_store()
        vs.add_documents(all_docs)
        vs.persist()
        logger.info("Ingested %d document chunks into vector store.", len(all_docs))

    return len(all_docs)


def retrieve_context(
    query: str,
    role: str | None = None,
    category: str | None = None,
    top_k: int = RAG_TOP_K,
) -> List[Document]:
    """
    Retrieve the most relevant documents for a query.
    Optionally filter by role or category metadata.
    """
    vs = get_vector_store()

    where_filter = {}
    if role:
        where_filter["role"] = role
    if category:
        where_filter["category"] = category

    try:
        if where_filter:
            results = vs.similarity_search(
                query, k=top_k, filter=where_filter
            )
        else:
            results = vs.similarity_search(query, k=top_k)
    except Exception:
        # Fallback without filter if metadata filtering fails
        results = vs.similarity_search(query, k=top_k)

    return results


def format_context(docs: List[Document]) -> str:
    """Concatenate retrieved documents into a single context string."""
    if not docs:
        return "No specific context available from the knowledge base."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
