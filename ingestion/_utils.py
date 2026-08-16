"""Shared utilities for the ingestion package."""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


def cap_image_size(image_bytes: bytes, max_px: int) -> bytes:
    """Downscale an image so its longest side is at most *max_px*.

    `IMAGE_EMBED_MAX_PX` was documented in `.env.example` for a long time with
    no `Settings` field to receive it, so `extra="ignore"` swallowed the key
    and nothing ever resized. Pages render at 150 DPI
    (`ingestion/file_processor.py`), which puts an A4 page on the wire at
    roughly 1240x1754 — billed area on every provider, and ~28k estimated
    tokens per page on Jina's tile meter against a default TPM of 100k.

    Returns the input unchanged when *max_px* is 0 (cap disabled), when the
    image already fits, or when it cannot be decoded. Failing open is
    deliberate: this is a cost optimisation, and refusing to embed a page
    because PIL could not parse it would turn a saving into an outage.
    """
    if max_px <= 0:
        return image_bytes
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        if max(img.size) <= max_px:
            return image_bytes
        img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        # PNG keeps text and line art legible at small sizes, which is what
        # a downscaled document page mostly is. JPEG artefacts around glyph
        # edges are exactly the detail a visual embedder is reading.
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        logger.warning("Could not downscale image; embedding at full size", exc_info=True)
        return image_bytes


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
