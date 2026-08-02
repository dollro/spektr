"""Shared utilities for the ingestion package."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


def run_async[T](coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
    """Run an async coroutine from a sync context and return its result.

    If no event loop is running, uses ``asyncio.run``.
    If called from within a running loop, offloads to a background thread.

    Args:
        coro: The coroutine to run.
        timeout: Optional timeout in seconds. On expiry, pending tasks
            are cancelled gracefully before raising TimeoutError.
    """

    async def _with_timeout(c: Coroutine[Any, Any, T]) -> T:
        if timeout is None:
            return await c
        return await asyncio.wait_for(c, timeout=timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_with_timeout(coro))
    # Already inside an event loop — run in a dedicated thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _with_timeout(coro)).result()
