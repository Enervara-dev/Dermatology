"""
Streaming Gemini-backed answer generator used by the GraphRAG pipeline.
"""

from __future__ import annotations

import time
from typing import Generator

from graphrag.config.settings import settings
from graphrag.domain.vocabulary import DEFAULT_ANSWER_GOAL
from graphrag.llm.gemini_client import (
    DEFAULT_MODEL,
    generate_stream,
    get_client,
)
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLM:
    def __init__(self):
        # Fail fast if the API key is missing.
        get_client()
        self._model = settings.ANSWER_MODEL or DEFAULT_MODEL

    def generate_from_messages(self, messages: list[dict[str, str]]):
        logger.info("[3/3] Sending memory-aware structured context to LLM Engine...")
        system_instruction, user_prompt = _split_messages(messages)
        yield from self._stream(system_instruction=system_instruction, user_prompt=user_prompt)

    def generate_response(
        self,
        query_text: str,
        vector_context: str,
        graph_context: str,
        memory_context: str = "",
        conversation_history: str = "",
        query_type: str = "unknown",
        goal: str = DEFAULT_ANSWER_GOAL,
        risk_level: str = "none",
        needs_followup: bool = True,
        memory_only: bool = False,
        has_findings: bool = False,
        terminal: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Streaming Gemini answer that yields validated block dicts.
        """
        from graphrag.domain.answer_prompt import compose_system_prompt
        from graphrag.domain.clinical_policy import closure_directive

        logger.info("[3/3] Sending structured context to LLM Engine...")

        has_name = "Patient name:" in memory_context
        system_prompt = compose_system_prompt(
            query_type=query_type,
            risk_level=risk_level,
            has_name=has_name,
            terminal=terminal,
        )

        # ── Stage-4 interception ──────────────────────────────────────────────
        constraint = closure_directive(
            intent=query_type,
            needs_followup=needs_followup,
            memory_only=memory_only,
            has_findings=has_findings,
        )
        if constraint:
            system_prompt = f"{system_prompt}\n\n{constraint}"
            logger.info(
                "🧭 Stage-4 closure constraint injected (intent=%s, needs_followup=%s, memory_only=%s).",
                query_type, needs_followup, memory_only,
            )

        user_prompt = f"""
USER QUESTION: {query_text}

=== STRUCTURED CLINICAL MEMORY ===
{memory_context}

=== RECENT CONVERSATION ===
{conversation_history}

=== RETRIEVED MEDICAL CONTEXT ===
{vector_context}

=== GRAPH RELATIONS ===
{graph_context}
"""

        yield from self._stream(
            system_instruction=system_prompt,
            user_prompt=user_prompt,
            terminal=terminal
        )

    def _stream(
        self,
        *,
        system_instruction: str | None,
        user_prompt: str,
        terminal: bool = False
    ) -> Generator[dict, None, None]:
        try:
            t_start = time.monotonic()

            logger.info("\n" + "=" * 80)
            logger.info("AI RESPONSE (NDJSON STREAM)")
            logger.info("=" * 80 + "\n")

            t_first_visible: float | None = None

            token_stream = generate_stream(
                model=self._model,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
            )

            def token_generator():
                nonlocal t_first_visible
                for piece in token_stream:
                    if t_first_visible is None:
                        t_first_visible = time.monotonic()
                        logger.info(
                            f"⏱️  Time-to-first-visible-token: "
                            f"{(t_first_visible - t_start) * 1000:.0f}ms"
                        )
                    # We can print pieces as they arrive (unvalidated, for diagnostics/progress)
                    print(piece, end="", flush=True)
                    yield piece

            from graphrag.validators.answer_validator import iter_blocks

            for block in iter_blocks(token_generator(), terminal=terminal):
                yield block.model_dump()

            t_end = time.monotonic()
            logger.info(
                f"\n⏱️  Stream complete in {(t_end - t_start) * 1000:.0f}ms "
            )
            print("\n\n" + "=" * 80 + "\n")

        except Exception as e:
            logger.error(f"\nLLM Error: {e}")
            raise


def _split_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    """
    Collapse an OpenAI-style messages array into (system_instruction, user_prompt)
    that Gemini's generate_content API expects.
    """
    system_parts: list[str] = []
    body_parts: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            body_parts.append(f"Assistant: {content}")
        else:
            body_parts.append(f"User: {content}")
    system_instruction = "\n\n".join(p for p in system_parts if p).strip() or None
    user_prompt = "\n\n".join(p for p in body_parts if p).strip()
    return system_instruction, user_prompt
