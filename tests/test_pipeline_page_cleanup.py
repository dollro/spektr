"""Failure cleanup in `_process_pages`.

Regression: the old cleanup closed *every* coroutine on failure, including
ones a still-running task owned. A sibling queued behind `Semaphore(2)` then
woke up and awaited an already-closed coroutine, so the real error was
replaced by `RuntimeError: cannot reuse already awaited coroutine` and the
orphaned tasks resurfaced as "Task exception was never retrieved" during
interpreter shutdown. Both made real ingest failures near-undiagnosable.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any
from unittest.mock import patch

import pytest

from ingestion.pipeline import _process_pages


class _BoomError(Exception):
    """Distinctive error that must survive cleanup unchanged."""


def _page_tasks(text: list[Any], image: list[Any]) -> Any:
    class _PT:
        pass

    pt = _PT()
    pt.text = text  # type: ignore[attr-defined]
    pt.image = image  # type: ignore[attr-defined]
    return pt


async def _run(text_coros: list[Any], image_coros: list[Any]) -> None:
    """Drive _process_pages with pre-built coroutines for a single page."""
    with patch(
        "ingestion.pipeline._build_page_tasks",
        return_value=_page_tasks(text_coros, image_coros),
    ):
        await _process_pages(
            filename="f.pdf",
            pages=["page"],
            dl_chunks=None,
            mime="application/pdf",
            now="2026-01-01",
            dense=object(),
            multivec=None,
            embedder=object(),  # type: ignore[arg-type]
            graph_engine=None,
        )


async def test_real_error_propagates() -> None:
    """The first real failure must reach the caller unchanged."""

    async def fails() -> None:
        raise _BoomError("the actual problem")

    async def slow() -> None:
        await asyncio.sleep(0.05)

    # More coroutines than the semaphore allows, so several are still queued
    # when the first one blows up — the exact shape that used to corrupt.
    coros = [fails()] + [slow() for _ in range(8)]

    with pytest.raises(_BoomError, match="the actual problem"):
        await _run(coros, [])


async def test_no_tasks_survive_the_failure() -> None:
    """The discriminating regression test for this module.

    An orphaned task is the root cause: it outlives the file it belongs to,
    then wakes onto a coroutine the cleanup already closed and raises
    `cannot reuse already awaited coroutine` into the void. Verified to fail
    against the pre-fix implementation; the other tests here guard the new
    behaviour but also pass against the old one.
    """
    started = 0

    async def fails() -> None:
        raise _BoomError("boom")

    async def counts() -> None:
        nonlocal started
        started += 1
        await asyncio.sleep(0.05)

    before = len(asyncio.all_tasks())
    with pytest.raises(_BoomError):
        await _run([fails()] + [counts() for _ in range(6)], [])

    # Give any orphan a chance to show itself.
    await asyncio.sleep(0.1)
    assert len(asyncio.all_tasks()) <= before, "a page task outlived the failure"


async def test_unstarted_coroutines_do_not_warn() -> None:
    """Cancelled-while-queued coroutines must still be closed."""

    async def fails() -> None:
        raise _BoomError("boom")

    async def never_runs() -> None:  # pragma: no cover - must not execute
        await asyncio.sleep(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_BoomError):
            await _run([fails()] + [never_runs() for _ in range(6)], [])
        await asyncio.sleep(0.05)

    never_awaited = [w for w in caught if "never awaited" in str(w.message)]
    assert not never_awaited, f"unclosed coroutines: {[str(w.message) for w in never_awaited]}"


async def test_pending_image_coroutines_are_closed() -> None:
    """Images run serially; the ones after a failure were never handed off."""

    async def fails() -> None:
        raise _BoomError("image boom")

    async def pending() -> None:  # pragma: no cover - must not execute
        await asyncio.sleep(0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_BoomError, match="image boom"):
            await _run([], [fails(), pending(), pending()])

    never_awaited = [w for w in caught if "never awaited" in str(w.message)]
    assert not never_awaited


async def test_success_path_runs_everything() -> None:
    """The fix must not change behaviour when nothing fails."""
    ran: list[int] = []

    async def work(i: int) -> None:
        await asyncio.sleep(0)
        ran.append(i)

    await _run([work(i) for i in range(5)], [work(100 + i) for i in range(2)])
    assert sorted(ran) == [0, 1, 2, 3, 4, 100, 101]
