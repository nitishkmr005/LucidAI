from __future__ import annotations

import asyncio

from loguru import logger


async def web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Async DuckDuckGo search returning [{title, snippet, url}].

    Runs the synchronous DDGS client in an executor to avoid blocking.
    Returns an empty list on any failure (timeout, rate limit, no network).
    """
    def _sync_search() -> list[dict]:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]

    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _sync_search),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        logger.warning("event=web_search_timeout query={!r}", query)
        return []
    except Exception as exc:
        logger.warning("event=web_search_error query={!r} error={}", query, exc)
        return []
