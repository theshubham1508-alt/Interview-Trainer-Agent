"""
FastAPI router — all Interview Trainer Agent API endpoints.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.agent import get_agent
from app.config import DOCUMENTS_PATH, GRANITE_MODEL_ID, VECTOR_STORE_PATH
from app.rag import ingest_documents, get_vector_store
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


# ── Health check ──────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and component status."""
    try:
        vs = get_vector_store()
        vs_status = "connected"
    except Exception:
        vs_status = "unavailable"

    return HealthResponse(
        status="healthy",
        model=GRANITE_MODEL_ID,
        vector_store=vs_status,
    )


# ── Question Set Generation ───────────────────────────────────────────────────

@router.post("/generate-questions", response_model=QuestionSetResponse, tags=["Interview Prep"])
async def generate_questions(req: QuestionSetRequest):
    """
    Generate a tailored interview question set (technical, behavioral, HR)
    plus a preparation strategy for the given user profile.
    """
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
            sources=[SourceDoc(**s) for s in result.get("sources", [])],
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Question generation failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ── Model Answer ──────────────────────────────────────────────────────────────

@router.post("/model-answer", response_model=ModelAnswerResponse, tags=["Interview Prep"])
async def get_model_answer(req: ModelAnswerRequest):
    """
    Provide a model answer, key talking points, common mistakes,
    and improvement tips for a specific interview question.
    """
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
            sources=[SourceDoc(**s) for s in result.get("sources", [])],
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Model answer generation failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ── Resume Analysis ───────────────────────────────────────────────────────────

@router.post("/analyze-resume", response_model=ResumeAnalysisResponse, tags=["Interview Prep"])
async def analyze_resume(req: ResumeAnalysisRequest):
    """
    Analyze a pasted resume and produce interview readiness feedback,
    predicted questions, skill gap analysis, and a 30-day prep plan.
    """
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
            sources=[SourceDoc(**s) for s in result.get("sources", [])],
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Resume analysis failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ── Chat ──────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Interview Prep"])
async def chat(req: ChatRequest):
    """
    Free-form follow-up conversation with the Interview Coach.
    Maintains session history for contextual responses.
    """
    try:
        agent = get_agent()
        history = [{"role": m.role, "content": m.content} for m in (req.history or [])]
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
            sources=[SourceDoc(**s) for s in result.get("sources", [])],
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ── Document Upload & Ingestion ───────────────────────────────────────────────

@router.post("/ingest/upload", response_model=IngestResponse, tags=["Knowledge Base"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF, TXT, or JSON interview document to the knowledge base.
    The file is saved and ingested into the vector store in the background.
    """
    allowed_ext = {".pdf", ".txt", ".json"}
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_ext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(allowed_ext)}",
        )

    save_path = Path(DOCUMENTS_PATH) / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    background_tasks.add_task(ingest_documents, DOCUMENTS_PATH)

    return IngestResponse(
        chunks_added=0,
        detail=f"File '{file.filename}' uploaded successfully. Ingestion running in background.",
    )


@router.post("/ingest/rebuild", response_model=IngestResponse, tags=["Knowledge Base"])
async def rebuild_index():
    """
    Re-ingest all documents from the documents directory and rebuild the vector store index.
    """
    try:
        count = ingest_documents(DOCUMENTS_PATH)
        return IngestResponse(
            chunks_added=count,
            detail=f"Knowledge base rebuilt successfully with {count} chunks.",
        )
    except Exception as e:
        logger.exception("Index rebuild failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
