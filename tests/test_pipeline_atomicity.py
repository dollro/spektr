"""Tests for ingest_file failure tracking and poison-pill behaviour.

Covers the contract:
- A single failure re-raises so CocoIndex retries.
- Repeated failures increment a persistent counter.
- At max_retries, the exception is swallowed + logged CRITICAL so
  the rest of the batch proceeds (CocoIndex marks file processed).
- A successful ingest resets the counter.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion._failure_tracker import FailureTracker


@pytest.fixture
def tracker(tmp_path: Path) -> FailureTracker:
    return FailureTracker(db_path=tmp_path / "failures.db")


class TestFailureTracker:
    def test_first_failure_returns_one(self, tracker: FailureTracker) -> None:
        assert tracker.record_failure("a.pdf", "boom") == 1

    def test_repeated_failures_increment(self, tracker: FailureTracker) -> None:
        assert tracker.record_failure("a.pdf") == 1
        assert tracker.record_failure("a.pdf") == 2
        assert tracker.record_failure("a.pdf") == 3

    def test_independent_files(self, tracker: FailureTracker) -> None:
        tracker.record_failure("a.pdf")
        tracker.record_failure("a.pdf")
        tracker.record_failure("b.pdf")
        assert tracker.fail_count("a.pdf") == 2
        assert tracker.fail_count("b.pdf") == 1

    def test_reset_clears_count(self, tracker: FailureTracker) -> None:
        tracker.record_failure("a.pdf")
        tracker.record_failure("a.pdf")
        tracker.reset("a.pdf")
        assert tracker.fail_count("a.pdf") == 0

    def test_should_poison(self, tracker: FailureTracker) -> None:
        tracker.record_failure("a.pdf")
        tracker.record_failure("a.pdf")
        assert tracker.should_poison("a.pdf", max_retries=3) is False
        tracker.record_failure("a.pdf")
        assert tracker.should_poison("a.pdf", max_retries=3) is True

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db = tmp_path / "failures.db"
        FailureTracker(db_path=db).record_failure("a.pdf")
        FailureTracker(db_path=db).record_failure("a.pdf")
        assert FailureTracker(db_path=db).fail_count("a.pdf") == 2


class TestIngestFileAtomicity:
    """Pipeline wiring: re-raise until poison, then swallow."""

    @pytest.fixture
    def isolated_tracker(self, tmp_path: Path) -> FailureTracker:
        t = FailureTracker(db_path=tmp_path / "failures.db")
        with patch("ingestion.pipeline.get_tracker", return_value=t):
            yield t

    def _run_ingest_with_failure(self, exc: Exception) -> Exception | str:
        """Call ingest_file with a file guaranteed to fail at the run_async step."""
        from ingestion.pipeline import ingest_file

        mock_settings = MagicMock()
        mock_settings.pipeline_timeout = 30
        mock_settings.pipeline_max_retries = 3
        mock_settings.graph_enabled = False
        mock_settings.image_embed_strategy = "smart"

        with (
            patch("ingestion.pipeline.settings", mock_settings),
            patch("ingestion.pipeline.run_async", side_effect=exc),
            patch("ingestion.pipeline._get_qdrant_client", return_value=MagicMock()),
            patch("ingestion.pipeline.create_embedder", return_value=MagicMock()),
        ):
            try:
                return ingest_file(b"hello world", "poison.txt")
            except Exception as raised:  # noqa: BLE001
                return raised

    def test_re_raises_under_threshold(
        self, isolated_tracker: FailureTracker
    ) -> None:
        for _ in range(2):
            result = self._run_ingest_with_failure(RuntimeError("boom"))
            assert isinstance(result, RuntimeError)
        assert isolated_tracker.fail_count("poison.txt") == 2

    def test_swallows_at_threshold(
        self, isolated_tracker: FailureTracker, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        caplog.set_level(logging.CRITICAL, logger="ingestion.pipeline")
        for _ in range(2):
            self._run_ingest_with_failure(RuntimeError("boom"))

        result = self._run_ingest_with_failure(RuntimeError("final boom"))
        assert result == "poison.txt"
        assert isolated_tracker.fail_count("poison.txt") == 3
        assert any("POISON PILL" in rec.message for rec in caplog.records)

    def test_timeout_is_tracked_same_as_exception(
        self, isolated_tracker: FailureTracker
    ) -> None:
        result = self._run_ingest_with_failure(TimeoutError("slow"))
        assert isinstance(result, TimeoutError)
        assert isolated_tracker.fail_count("poison.txt") == 1
