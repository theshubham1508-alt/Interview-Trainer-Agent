"""
Configuration management for Interview Trainer Agent.
Loads settings from environment variables / .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── IBM watsonx.ai ────────────────────────────────────────────────────────────
WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# IBM Granite model – default to granite-13b-chat-v2 (IBM Cloud Lite compatible)
GRANITE_MODEL_ID: str = os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-chat-v2")

# ── Application ───────────────────────────────────────────────────────────────
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

# ── Storage paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_PATH: str = os.getenv(
    "VECTOR_STORE_PATH", str(BASE_DIR / "data" / "vector_store")
)
DOCUMENTS_PATH: str = os.getenv(
    "DOCUMENTS_PATH", str(BASE_DIR / "data" / "documents")
)

# Ensure directories exist
Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
Path(DOCUMENTS_PATH).mkdir(parents=True, exist_ok=True)

# ── IBM Granite generation parameters ────────────────────────────────────────
GRANITE_PARAMS = {
    "decoding_method": "greedy",
    "max_new_tokens": 1024,
    "min_new_tokens": 50,
    "stop_sequences": ["<|endoftext|>"],
    "repetition_penalty": 1.1,
    "temperature": 0.7,
}

# ── Embedding model (local, no API key required) ──────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── RAG parameters ────────────────────────────────────────────────────────────
RAG_TOP_K: int = 5          # number of retrieved chunks
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100
