"""
Core agent logic for Interview Trainer Agent.
Orchestrates RAG retrieval and IBM Granite LLM generation.
"""
from __future__ import annotations

import logging
from typing import Any

from app.llm import generate_text
from app.prompts import (
    FOLLOWUP_CHAT_PROMPT,
    MODEL_ANSWER_PROMPT,
    QUESTION_GENERATION_PROMPT,
    RESUME_ANALYSIS_PROMPT,
    SYSTEM_PERSONA,
)
from app.rag import format_context, retrieve_context

logger = logging.getLogger(__name__)


class InterviewTrainerAgent:
    """
    Main agent that ties together RAG retrieval and IBM Granite generation
    to deliver a full interview preparation experience.
    """

    def __init__(self) -> None:
        logger.info("Interview Trainer Agent initialized.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_question_set(
        self,
        name: str,
        role: str,
        experience_level: str,
        background: str = "",
    ) -> dict[str, Any]:
        """
        Generate a tailored set of interview questions and preparation strategy.
        Returns structured dict with the LLM response and retrieved context sources.
        """
        query = f"{experience_level} {role} interview questions technical behavioral HR"
        docs = retrieve_context(query, role=None)  # broad retrieval first
        context = format_context(docs)

        prompt = QUESTION_GENERATION_PROMPT.format(
            system_persona=SYSTEM_PERSONA,
            context=context,
            name=name,
            role=role,
            experience_level=experience_level,
            background=background or f"{experience_level} professional targeting {role} positions",
        )

        response = generate_text(prompt)

        return {
            "name": name,
            "role": role,
            "experience_level": experience_level,
            "response": response,
            "sources": [
                {"source": d.metadata.get("source", "knowledge base"), "excerpt": d.page_content[:150]}
                for d in docs
            ],
        }

    def get_model_answer(
        self,
        question: str,
        name: str,
        role: str,
        experience_level: str,
    ) -> dict[str, Any]:
        """
        Generate a model answer, key points, mistakes to avoid, and tips
        for a specific interview question.
        """
        docs = retrieve_context(question, top_k=3)
        context = format_context(docs)

        prompt = MODEL_ANSWER_PROMPT.format(
            system_persona=SYSTEM_PERSONA,
            context=context,
            name=name,
            role=role,
            experience_level=experience_level,
            question=question,
        )

        response = generate_text(prompt)

        return {
            "question": question,
            "role": role,
            "experience_level": experience_level,
            "response": response,
            "sources": [
                {"source": d.metadata.get("source", "knowledge base"), "excerpt": d.page_content[:150]}
                for d in docs
            ],
        }

    def analyze_resume(
        self,
        name: str,
        role: str,
        experience_level: str,
        resume_text: str,
    ) -> dict[str, Any]:
        """
        Analyze a resume/background and produce interview readiness feedback,
        predicted questions, and a 30-day prep plan.
        """
        query = f"{role} interview preparation resume skills {experience_level}"
        docs = retrieve_context(query, top_k=5)
        context = format_context(docs)

        prompt = RESUME_ANALYSIS_PROMPT.format(
            system_persona=SYSTEM_PERSONA,
            context=context,
            name=name,
            role=role,
            experience_level=experience_level,
            resume_text=resume_text,
        )

        response = generate_text(prompt)

        return {
            "name": name,
            "role": role,
            "experience_level": experience_level,
            "response": response,
            "sources": [
                {"source": d.metadata.get("source", "knowledge base"), "excerpt": d.page_content[:150]}
                for d in docs
            ],
        }

    def chat(
        self,
        user_message: str,
        name: str,
        role: str,
        experience_level: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Handle a free-form follow-up conversation turn.
        Maintains lightweight conversation history in the prompt.
        """
        docs = retrieve_context(user_message, top_k=3)
        context = format_context(docs)

        # Format history as readable turns
        history_text = ""
        if history:
            for turn in history[-6:]:  # keep last 6 turns to stay within token budget
                role_label = "User" if turn.get("role") == "user" else "Coach"
                history_text += f"{role_label}: {turn.get('content', '')}\n"

        prompt = FOLLOWUP_CHAT_PROMPT.format(
            system_persona=SYSTEM_PERSONA,
            context=context,
            name=name,
            role=role,
            experience_level=experience_level,
            history=history_text or "No prior conversation.",
            user_message=user_message,
        )

        response = generate_text(prompt)

        return {
            "user_message": user_message,
            "response": response,
            "sources": [
                {"source": d.metadata.get("source", "knowledge base"), "excerpt": d.page_content[:150]}
                for d in docs
            ],
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
_agent: InterviewTrainerAgent | None = None


def get_agent() -> InterviewTrainerAgent:
    global _agent
    if _agent is None:
        _agent = InterviewTrainerAgent()
    return _agent
