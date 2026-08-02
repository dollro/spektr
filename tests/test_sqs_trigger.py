"""Tests for the SQS-as-trigger daemon.

CocoIndex v1's amazon_s3 connector is scan-only, so live S3 ingestion is driven
by this loop instead. The contract worth pinning down:

- a startup sweep always runs, so changes made while the daemon was down are
  recovered (SQS retention caps at 14 days — older events cannot be replayed);
- an event burst is coalesced into one update;
- messages are deleted only after an update that reported zero errors, so a
  failed run replays rather than silently dropping the event;
- the interval sweep fires even with an empty queue.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from ingestion import sqs_trigger


class FakeSqs:
    """Minimal aiobotocore SQS stand-in driven by a scripted receive queue."""

    def __init__(self, batches: list[list[dict[str, Any]]]) -> None:
        self._batches = list(batches)
        self.deleted: list[list[dict[str, Any]]] = []

    async def receive_message(self, **kwargs: Any) -> dict[str, Any]:
        if not self._batches:
            raise _StopLoop
        batch = self._batches.pop(0)
        return {"Messages": batch} if batch else {}

    async def delete_message_batch(self, **kwargs: Any) -> dict[str, Any]:
        self.deleted.append(kwargs["Entries"])
        return {}


class _StopLoop(BaseException):
    """Raised by the fake to break out of the daemon's infinite loop.

    Deliberately a BaseException: ``_receive`` swallows ``Exception`` so that a
    transient SQS failure falls back to the interval sweep, and an ordinary
    test-control exception would be caught by that same handler.
    """


def _msg(i: int) -> dict[str, Any]:
    return {"MessageId": str(i), "ReceiptHandle": f"rh-{i}"}


async def _drive(
    batches: list[list[dict[str, Any]]],
    *,
    errors: list[int],
    interval_hours: float = 24.0,
    debounce: float = 0.0,
) -> tuple[FakeSqs, list[str]]:
    """Run the loop against a fake SQS until the scripted batches run out."""
    sqs = FakeSqs(batches)
    calls: list[str] = []
    error_seq = list(errors)

    async def fake_run_update(app: Any, **kwargs: Any) -> int:
        calls.append("update")
        return error_seq.pop(0) if error_seq else 0

    with (
        patch("ingestion.runner.run_update", new=fake_run_update),
        patch("ingestion.sqs_trigger.settings") as mock_settings,
        patch.object(sqs_trigger, "_open_sqs_client", return_value=_FakeCM(sqs)),
    ):
        mock_settings.s3_full_scan_interval_hours = interval_hours
        mock_settings.s3_sqs_queue_url = "https://sqs.test/q"
        mock_settings.s3_sqs_debounce_seconds = debounce
        with pytest.raises(_StopLoop):
            await sqs_trigger.run_sqs_triggered(object())
    return sqs, calls


class _FakeCM:
    def __init__(self, obj: Any) -> None:
        self._obj = obj

    async def __aenter__(self) -> Any:
        return self._obj

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class TestStartupSweep:
    async def test_startup_sweep_runs_before_polling(self) -> None:
        sqs, calls = await _drive([], errors=[0])
        # The startup sweep happens, then the first receive raises _StopLoop.
        assert calls == ["update"]
        assert sqs.deleted == []


class TestEventHandling:
    async def test_event_triggers_update_and_deletes_messages(self) -> None:
        sqs, calls = await _drive([[_msg(1)], []], errors=[0, 0])
        # startup + one event-driven update
        assert calls == ["update", "update"]
        assert len(sqs.deleted) == 1
        assert [e["ReceiptHandle"] for e in sqs.deleted[0]] == ["rh-1"]

    async def test_burst_is_coalesced_into_one_update(self) -> None:
        # First receive returns two messages, the drain returns two more,
        # then an empty drain ends the coalescing.
        sqs, calls = await _drive(
            [[_msg(1), _msg(2)], [_msg(3), _msg(4)], []],
            errors=[0, 0],
        )
        assert calls == ["update", "update"]
        assert len(sqs.deleted) == 1
        assert len(sqs.deleted[0]) == 4

    async def test_messages_kept_when_update_reports_errors(self) -> None:
        """A failed run must replay the event, not drop it."""
        sqs, calls = await _drive([[_msg(1)], []], errors=[0, 3])
        assert calls == ["update", "update"]
        assert sqs.deleted == []


class TestIntervalSweep:
    async def test_empty_queue_does_not_trigger_update(self) -> None:
        sqs, calls = await _drive([[], []], errors=[0])
        # Only the startup sweep; the empty polls are not due yet.
        assert calls == ["update"]

    async def test_interval_elapsed_triggers_update(self) -> None:
        sqs, calls = await _drive([[]], errors=[0, 0], interval_hours=0.0)
        assert calls == ["update", "update"]
        assert sqs.deleted == []


class TestDeleteBatching:
    async def test_deletes_are_chunked_to_the_api_limit(self) -> None:
        sqs = FakeSqs([])
        await sqs_trigger._delete_messages(
            sqs, "https://sqs.test/q", [_msg(i) for i in range(23)]
        )
        assert [len(batch) for batch in sqs.deleted] == [10, 10, 3]

    async def test_drain_is_bounded_on_a_continuously_fed_queue(self) -> None:
        """An unbounded drain would never hand control back to the update."""

        class NeverEmpty:
            async def receive_message(self, **kwargs: Any) -> dict[str, Any]:
                return {"Messages": [_msg(1)] * 10}

        drained = await sqs_trigger._drain(NeverEmpty(), "https://sqs.test/q")
        assert len(drained) <= sqs_trigger._MAX_DRAIN_MESSAGES + 10

    async def test_delete_failure_is_swallowed(self) -> None:
        class Failing(FakeSqs):
            async def delete_message_batch(self, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("throttled")

        # Must not raise: the message is simply redelivered later.
        await sqs_trigger._delete_messages(Failing([]), "https://sqs.test/q", [_msg(1)])


class TestReceiveResilience:
    async def test_receive_failure_returns_empty(self) -> None:
        class Failing:
            async def receive_message(self, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("network")

        assert await sqs_trigger._receive(Failing(), "https://sqs.test/q", wait=0) == []


class TestIntervalOnlyFallback:
    async def test_no_queue_url_falls_back_to_interval_sweeps(self) -> None:
        calls: list[str] = []

        async def fake_run_update(app: Any, **kwargs: Any) -> int:
            calls.append("update")
            if len(calls) >= 3:
                raise _StopLoop
            return 0

        async def fake_sleep(_seconds: float) -> None:
            return None

        with (
            patch("ingestion.runner.run_update", new=fake_run_update),
            patch("ingestion.sqs_trigger.settings") as mock_settings,
            patch.object(asyncio, "sleep", new=fake_sleep),
        ):
            mock_settings.s3_full_scan_interval_hours = 24.0
            mock_settings.s3_sqs_queue_url = ""
            with pytest.raises(_StopLoop):
                await sqs_trigger.run_sqs_triggered(object())

        assert calls == ["update", "update", "update"]
