"""
IBM Granite LLM wrapper via IBM watsonx.ai.
Provides a LangChain-compatible interface for text generation.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_ibm import WatsonxLLM

from app.config import (
    GRANITE_MODEL_ID,
    GRANITE_PARAMS,
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
)


@lru_cache(maxsize=1)
def get_granite_llm() -> WatsonxLLM:
    """
    Return a cached LangChain WatsonxLLM instance backed by IBM Granite.
    Raises a clear error if credentials are missing.
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        raise EnvironmentError(
            "IBM watsonx.ai credentials are not set. "
            "Please configure WATSONX_API_KEY and WATSONX_PROJECT_ID "
            "in your .env file."
        )

    llm = WatsonxLLM(
        model_id=GRANITE_MODEL_ID,
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        params=GRANITE_PARAMS,
    )
    return llm


def generate_text(prompt: str, **override_params: Any) -> str:
    """
    Generate text from IBM Granite given a prompt.
    Optional override_params can adjust generation settings per call.
    """
    llm = get_granite_llm()
    if override_params:
        # Clone with updated params for this specific call
        merged = {**GRANITE_PARAMS, **override_params}
        llm_temp = WatsonxLLM(
            model_id=GRANITE_MODEL_ID,
            url=WATSONX_URL,
            project_id=WATSONX_PROJECT_ID,
            apikey=WATSONX_API_KEY,
            params=merged,
        )
        return llm_temp.invoke(prompt)
    return llm.invoke(prompt)
