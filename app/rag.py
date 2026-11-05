"""
Lightweight RAG pipeline.
Uses keyword-based retrieval to reduce RAM consumption.
No PyTorch, HuggingFace or ChromaDB required.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_PATH,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)

# Lightweight in-memory document storage
_documents: List[Document] = []


def tokenize(text: str) -> List[str]:
    """Convert text into searchable lowercase tokens."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def _load_json_interview_data(filepath: Path) -> List[Document]:
    """Load structured JSON interview data."""

    docs = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for entry in data:
            content_parts = []

            for key, label in [
                ("role", "Role"),
                ("category", "Category"),
                ("question", "Question"),
                ("answer", "Model Answer"),
                ("tips", "Tips"),
            ]:
                if key in entry:
                    content_parts.append(f"{label}: {entry[key]}")

            content = "\n".join(content_parts)

            if content.strip():
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": filepath.name,
                            "role": entry.get("role", "general"),
                            "category": entry.get("category", "general"),
                            "type": "structured_qa",
                        },
                    )
                )

    return docs


def ingest_documents(documents_path: str = DOCUMENTS_PATH) -> int:
    """Load and chunk documents into lightweight memory storage."""

    global _documents

    docs_dir = Path(documents_path)
    all_docs = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    if not docs_dir.exists():
        logger.warning("Documents directory not found: %s", docs_dir)
        _documents = []
        return 0

    # Load PDF documents
    for pdf_file in docs_dir.glob("**/*.pdf"):
        try:
            pages = PyPDFLoader(str(pdf_file)).load()
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)

            logger.info(
                "Loaded PDF: %s (%d chunks)",
                pdf_file.name,
                len(chunks),
            )

        except Exception as e:
            logger.warning("Failed to load PDF %s: %s", pdf_file, e)

    # Load text documents
    for txt_file in docs_dir.glob("**/*.txt"):
        try:
            pages = TextLoader(
                str(txt_file),
                encoding="utf-8",
            ).load()

            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)

            logger.info(
                "Loaded TXT: %s (%d chunks)",
                txt_file.name,
                len(chunks),
            )

        except Exception as e:
            logger.warning("Failed to load TXT %s: %s", txt_file, e)

    # Load JSON documents
    for json_file in docs_dir.glob("**/*.json"):
        try:
            json_docs = _load_json_interview_data(json_file)
            all_docs.extend(json_docs)

            logger.info(
                "Loaded JSON: %s (%d entries)",
                json_file.name,
                len(json_docs),
            )

        except Exception as e:
            logger.warning("Failed to load JSON %s: %s", json_file, e)

    _documents = all_docs

    logger.info(
        "Successfully loaded %d documents into lightweight RAG.",
        len(_documents),
    )

    return len(_documents)


def retrieve_context(
    query: str,
    role: str | None = None,
    category: str | None = None,
    top_k: int = RAG_TOP_K,
) -> List[Document]:
    """Retrieve relevant documents using keyword matching."""

    if not _documents:
        return []

    query_tokens = set(tokenize(query))

    if not query_tokens:
        return []

    scored_docs = []

    for doc in _documents:

        if role and doc.metadata.get("role") != role:
            continue

        if category and doc.metadata.get("category") != category:
            continue

        doc_tokens = tokenize(doc.page_content)

        if not doc_tokens:
            continue

        token_counts = Counter(doc_tokens)

        score = 0.0

        for token in query_tokens:
            if token in token_counts:
                score += 1 + math.log1p(token_counts[token])

        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        doc
        for score, doc in scored_docs[:top_k]
    ]


def format_context(docs: List[Document]) -> str:
    """Format retrieved documents for the LLM prompt."""

    if not docs:
        return "No specific context available from the knowledge base."

    return "\n\n---\n\n".join(
        doc.page_content for doc in docs
    )