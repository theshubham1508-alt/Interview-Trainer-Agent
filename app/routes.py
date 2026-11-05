"""
FastAPI router — all Interview Trainer Agent API endpoints.

Deployment notes:
- Endpoints that call the LLM / ingestion are plain `def` (not `async def`),
  so FastAPI runs them in a thread pool and the server stays responsive.
- Internal error details are logged, never sent to the client.
- Uploads are size-limited and filenames are sanitized.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.agent import get_agent
from app.config import DOCUMENTS_PATH, GRANITE_MODEL_ID
from app.rag import ingest_documents
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    ModelAnswerRequest,
    ModelAnswerResponse,
    QuestionSetRequest,
    QuestionSetResponse,
    ResumeAnalysisRequest,
    ResumeAnalysisResponse,
    SourceDoc,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Settings & helpers ────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".json"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

# Only one ingestion may run at a time (upload background task / manual rebuild).
_ingest_lock = threading.Lock()


def _sources(result: Dict[str, Any]) -> List[SourceDoc]:
    """Convert raw source dicts from the agent into SourceDoc models."""
    return [SourceDoc(**source) for source in result.get("sources", [])]


def _to_http_error(error: Exception, action: str) -> HTTPException:
    """
    Map an exception to a safe HTTPException.
    Full details go to the logs; the client only gets a generic message.
    """
    if isinstance(error, HTTPException):
        return error

    if isinstance(error, EnvironmentError):
        # Typically a missing API key / credentials / project id.
        logger.error("%s failed: configuration error: %s", action, error)
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not configured correctly. Please try again later.",
        )

    logger.exception("%s failed", action)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"{action} failed. Please try again.",
    )


def _safe_filename(raw_name: str | None) -> str:
    """Return a safe filename (no path parts, no odd characters)."""
    name = Path(raw_name or "").name
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).stem).strip("._")
    suffix = Path(name).suffix.lower()
    return f"{stem}{suffix}" if stem else ""


def _run_ingestion(path: str) -> int:
    """Run ingestion under the lock. Used by background tasks."""
    with _ingest_lock:
        try:
            count = ingest_documents(path)
            logger.info("Ingestion finished: %s chunks", count)
            return count
        except Exception:
            logger.exception("Background ingestion failed")
            return 0


# ── Health Check ──────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="healthy",
        model=GRANITE_MODEL_ID,
        vector_store="keyword-based retrieval",
    )


# ── Question Generation ───────────────────────────────────────────────────────

@router.post(
    "/generate-questions",
    response_model=QuestionSetResponse,
    tags=["Interview Prep"],
)
def generate_questions(req: QuestionSetRequest):
    try:
        agent = get_agent()

        result = agent.generate_question_set(
            name=req.name,
            role=req.role,
            experience_level=req.experience_level,
            background=req.background or "",
        )

        return QuestionSetResponse(
            name=result["name"],
            role=result["role"],
            experience_level=result["experience_level"],
            response=result["response"],
            sources=_sources(result),
        )

    except Exception as error:
        raise _to_http_error(error, "Question generation")


# ── Model Answer ──────────────────────────────────────────────────────────────

@router.post(
    "/model-answer",
    response_model=ModelAnswerResponse,
    tags=["Interview Prep"],
)
def get_model_answer(req: ModelAnswerRequest):
    try:
        agent = get_agent()

        result = agent.get_model_answer(
            question=req.question,
            name=req.name,
            role=req.role,
            experience_level=req.experience_level,
        )

        return ModelAnswerResponse(
            question=result["question"],
            role=result["role"],
            experience_level=result["experience_level"],
            response=result["response"],
            sources=_sources(result),
        )

    except Exception as error:
        raise _to_http_error(error, "Model answer generation")


# ── Resume Analysis ───────────────────────────────────────────────────────────

@router.post(
    "/analyze-resume",
    response_model=ResumeAnalysisResponse,
    tags=["Interview Prep"],
)
def analyze_resume(req: ResumeAnalysisRequest):
    try:
        agent = get_agent()

        result = agent.analyze_resume(
            name=req.name,
            role=req.role,
            experience_level=req.experience_level,
            resume_text=req.resume_text,
        )

        return ResumeAnalysisResponse(
            name=result["name"],
            role=result["role"],
            experience_level=result["experience_level"],
            response=result["response"],
            sources=_sources(result),
        )

    except Exception as error:
        raise _to_http_error(error, "Resume analysis")


# ── Chat ──────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Interview Prep"],
)
def chat(req: ChatRequest):
    try:
        agent = get_agent()

        # Keep only the latest messages so the prompt never exceeds the model's context window.
        recent_messages = (req.history or [])[-MAX_HISTORY_MESSAGES:]
        history = [
            {"role": message.role, "content": message.content}
            for message in recent_messages
        ]

        result = agent.chat(
            user_message=req.message,
            name=req.name,
            role=req.role,
            experience_level=req.experience_level,
            history=history,
        )

        return ChatResponse(
            user_message=result["user_message"],
            response=result["response"],
            sources=_sources(result),
        )

    except Exception as error:
        raise _to_http_error(error, "Chat")


# ── Document Upload ───────────────────────────────────────────────────────────

@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    tags=["Knowledge Base"],
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    filename = _safe_filename(file.filename)
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF, TXT and JSON files are allowed.",
        )

    # Read in chunks so oversized files are rejected without loading everything into memory.
    size = 0
    chunks: List[bytes] = []
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File is too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.",
            )
        chunks.append(chunk)

    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        documents_directory = Path(DOCUMENTS_PATH)
        documents_directory.mkdir(parents=True, exist_ok=True)
        (documents_directory / filename).write_bytes(b"".join(chunks))
    except OSError:
        logger.exception("Could not save uploaded file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save the uploaded file.",
        )

    background_tasks.add_task(_run_ingestion, str(DOCUMENTS_PATH))

    return IngestResponse(
        chunks_added=0,
        detail=f"File '{filename}' uploaded successfully. Ingestion running in background.",
    )


# ── Rebuild Knowledge Base ────────────────────────────────────────────────────

@router.post(
    "/ingest/rebuild",
    response_model=IngestResponse,
    tags=["Knowledge Base"],
)
def rebuild_index():
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ingestion is already running. Please try again in a moment.",
        )

    try:
        count = ingest_documents(DOCUMENTS_PATH)

        return IngestResponse(
            chunks_added=count,
            detail=f"Knowledge base rebuilt successfully with {count} documents.",
        )

    except Exception as error:
        raise _to_http_error(error, "Knowledge base rebuild")

    finally:
        _ingest_lock.release()