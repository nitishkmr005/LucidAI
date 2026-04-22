from time import perf_counter
from typing import AsyncGenerator

from loguru import logger
from ollama import AsyncClient

from app.utils.module_logging import log_module_io
from config.settings import get_settings


async def stream_llm_response(
    transcript: str,
    conversation_history: list[dict[str, str]] | None = None,
    document_context: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream the LLM response token-by-token using Ollama.

    Builds the message list as: system prompt → optional document context →
    conversation history → current user message.

    Args:
        transcript: The user's transcribed speech to send as the current user message.
        conversation_history: Optional list of previous {"role": "user"/"assistant",
                              "content": "..."} dicts for multi-turn context.
        document_context: Optional document-aware system context injected as a
                          second system message when a document is loaded.

    Returns:
        An async generator that yields string tokens as they stream from the model.

    Library:
        ollama (AsyncClient) — streams chat completions from a local Ollama instance.
    """
    settings = get_settings()
    client = AsyncClient(host=settings.ollama_host)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": settings.llm_system_prompt},
    ]
    if document_context:
        messages.append({"role": "system", "content": document_context})
    messages.extend(conversation_history or [])
    messages.append({"role": "user", "content": transcript})

    logger.info(
        "event=llm_start model={} host={} history_turns={} input_preview={!r}",
        settings.llm_model,
        settings.ollama_host,
        len(conversation_history or []) // 2,
        transcript[:60],
    )

    full_output = ""
    started_at = perf_counter()
    stream = await client.chat(
        model=settings.llm_model,
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        token: str = chunk.message.content
        if token:
            full_output += token
            yield token

    log_module_io(
        module="llm",
        latency_ms=round((perf_counter() - started_at) * 1000, 2),
        input_payload={"messages": messages},
        output_payload={"text": full_output},
    )


async def complete_llm_response(
    transcript: str,
    conversation_history: list[dict[str, str]] | None = None,
    document_context: str | None = None,
) -> str:
    settings = get_settings()
    client = AsyncClient(host=settings.ollama_host)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": settings.llm_system_prompt},
    ]
    if document_context:
        messages.append({"role": "system", "content": document_context})
    messages.extend(conversation_history or [])
    messages.append({"role": "user", "content": transcript})

    logger.info(
        "event=llm_complete_start model={} host={} history_turns={} input_preview={!r}",
        settings.llm_model,
        settings.ollama_host,
        len(conversation_history or []) // 2,
        transcript[:60],
    )

    started_at = perf_counter()
    response = await client.chat(
        model=settings.llm_model,
        messages=messages,
        stream=False,
    )
    content = response.message.content or ""
    log_module_io(
        module="llm",
        latency_ms=round((perf_counter() - started_at) * 1000, 2),
        input_payload={"messages": messages},
        output_payload={"text": content},
    )
    return content
