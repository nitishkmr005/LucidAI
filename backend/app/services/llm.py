from time import perf_counter
from typing import AsyncGenerator

from loguru import logger

from app.utils.module_logging import log_module_io
from config.settings import get_settings


def _build_messages(
    transcript: str,
    conversation_history: list[dict[str, str]] | None,
    document_context: str | None,
) -> list[dict[str, str]]:
    settings = get_settings()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": settings.llm_system_prompt},
    ]
    if document_context:
        messages.append({"role": "system", "content": document_context})
    messages.extend(conversation_history or [])
    messages.append({"role": "user", "content": transcript})
    return messages


async def _stream_ollama(
    messages: list[dict[str, str]],
) -> AsyncGenerator[str, None]:
    from ollama import AsyncClient
    settings = get_settings()
    client = AsyncClient(host=settings.ollama_host)
    stream = await client.chat(model=settings.llm_model, messages=messages, stream=True)
    async for chunk in stream:
        token: str = chunk.message.content
        if token:
            yield token


async def _complete_ollama(messages: list[dict[str, str]]) -> str:
    from ollama import AsyncClient
    settings = get_settings()
    client = AsyncClient(host=settings.ollama_host)
    response = await client.chat(model=settings.llm_model, messages=messages, stream=False)
    return response.message.content or ""


async def _stream_openai(
    messages: list[dict[str, str]],
) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    stream = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,  # type: ignore[arg-type]
        stream=True,
        max_tokens=settings.llm_max_tokens,
    )
    async for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


async def _complete_openai(messages: list[dict[str, str]]) -> str:
    from openai import AsyncOpenAI
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,  # type: ignore[arg-type]
        stream=False,
        max_tokens=settings.llm_max_tokens,
    )
    return response.choices[0].message.content or ""


async def stream_llm_response(
    transcript: str,
    conversation_history: list[dict[str, str]] | None = None,
    document_context: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream LLM response token-by-token. Provider is selected via settings.llm_provider.

    Args:
        transcript: The user's transcribed speech.
        conversation_history: Previous turn dicts for multi-turn context.
        document_context: Optional second system message with document excerpts.

    Returns:
        Async generator yielding string tokens.
    """
    settings = get_settings()
    messages = _build_messages(transcript, conversation_history, document_context)

    logger.info(
        "event=llm_start provider={} model={} history_turns={} input_preview={!r}",
        settings.llm_provider,
        settings.openai_model if settings.llm_provider == "openai" else settings.llm_model,
        len(conversation_history or []) // 2,
        transcript[:60],
    )

    full_output = ""
    started_at = perf_counter()

    if settings.llm_provider == "openai":
        gen = _stream_openai(messages)
    else:
        gen = _stream_ollama(messages)

    async for token in gen:
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
    """
    Non-streaming LLM completion. Provider is selected via settings.llm_provider.

    Args:
        transcript: The user's transcribed speech.
        conversation_history: Previous turn dicts for multi-turn context.
        document_context: Optional second system message with document excerpts.

    Returns:
        Full response string.
    """
    settings = get_settings()
    messages = _build_messages(transcript, conversation_history, document_context)

    logger.info(
        "event=llm_complete_start provider={} model={} history_turns={} input_preview={!r}",
        settings.llm_provider,
        settings.openai_model if settings.llm_provider == "openai" else settings.llm_model,
        len(conversation_history or []) // 2,
        transcript[:60],
    )

    started_at = perf_counter()

    if settings.llm_provider == "openai":
        content = await _complete_openai(messages)
    else:
        content = await _complete_ollama(messages)

    log_module_io(
        module="llm",
        latency_ms=round((perf_counter() - started_at) * 1000, 2),
        input_payload={"messages": messages},
        output_payload={"text": content},
    )
    return content
