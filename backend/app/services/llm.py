from typing import AsyncGenerator

from loguru import logger
from ollama import AsyncClient

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

    stream = await client.chat(
        model=settings.llm_model,
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        token: str = chunk.message.content
        if token:
            yield token
