"""
FastAPI application entry point for Interview Trainer Agent.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import APP_HOST, APP_PORT, DEBUG, DOCUMENTS_PATH
from app.rag import ingest_documents
from app.routes import router

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot tasks: pre-ingest any seed documents in the documents folder."""
    logger.info("Interview Trainer Agent starting up...")
    docs_path = Path(DOCUMENTS_PATH)

    # Auto-ingest seed documents on startup if they exist
    doc_count = sum(1 for _ in docs_path.glob("**/*") if _.is_file())
    if doc_count > 0:
        logger.info("Found %d files in documents directory — ingesting...", doc_count)
        try:
            chunks = ingest_documents(DOCUMENTS_PATH)
            logger.info("Startup ingestion complete: %d chunks indexed.", chunks)
        except Exception as e:
            logger.warning("Startup ingestion failed (non-fatal): %s", e)
    else:
        logger.info("No documents found in %s — vector store will populate on first upload.", DOCUMENTS_PATH)

    yield
    logger.info("Interview Trainer Agent shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Interview Trainer Agent",
    description=(
        "An AI-powered Interview Trainer Agent built with IBM Granite (watsonx.ai) and RAG. "
        "Generates tailored interview questions, model answers, and preparation strategies "
        "for any role and experience level."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")

# ── Static files (frontend) ───────────────────────────────────────────────────
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(static_dir / "index.html"))


# ── Dev server entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )
