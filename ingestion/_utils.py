"""Shared utilities for the ingestion package."""

from __future__ import annotations

import asyncio
from typing import TypeVar

T = TypeVar("T")


def run_async(coro: T) -> T:  # type: ignore[type-var]
    """Run an async coroutine from a sync context.

    If no event loop is running, uses ``asyncio.run``.
    If called from within a running loop (e.g. inside a CocoIndex op
    dispatched by an async framework), offloads to a background thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    # Already inside an event loop — run in a dedicated thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]
