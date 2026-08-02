"""Tests for the CocoIndex v1 graph-cleanup target.

Replaces the v0 ``test_target_connector.py`` / ``test_pipeline_delete.py``.
The Qdrant half of that connector is gone — points are declared on the native
CocoIndex Qdrant target and reconciled per point id — so what is left to test is
the graph half: does a source file disappearing produce exactly one delete
action, and does that action reach the right engine?
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import cocoindex as coco
import pytest

from ingestion.graph_target import (
    GraphSourceHandler,
    GraphSourceState,
    remove_graph_data,
)


@pytest.fixture
def handler() -> GraphSourceHandler:
    return GraphSourceHandler()


def _state(fingerprint: str = "fp-1") -> GraphSourceState:
    return GraphSourceState(source_key="report.pdf", content_fingerprint=fingerprint)


class TestReconcileExistence:
    def test_unchanged_fingerprint_is_a_noop(self, handler: GraphSourceHandler) -> None:
        """A file re-declared with the same content produces no action."""
        out = handler.reconcile("report.pdf", _state("fp-1"), ["fp-1"], False)
        assert out is None

    def test_changed_fingerprint_emits_upsert(self, handler: GraphSourceHandler) -> None:
        out = handler.reconcile("report.pdf", _state("fp-2"), ["fp-1"], False)
        assert out is not None
        assert out.action.delete is False
        assert out.action.source_key == "report.pdf"
        assert out.tracking_record == "fp-2"

    def test_first_ingest_emits_upsert(self, handler: GraphSourceHandler) -> None:
        """A never-tracked key arrives with prev_may_be_missing=True."""
        out = handler.reconcile("report.pdf", _state("fp-1"), [], True)
        assert out is not None
        assert out.action.delete is False
        assert out.tracking_record == "fp-1"

    def test_possibly_missing_forces_reconcile(self, handler: GraphSourceHandler) -> None:
        """An uncertain prior state must not be optimised away."""
        out = handler.reconcile("report.pdf", _state("fp-1"), ["fp-1"], True)
        assert out is not None

    def test_known_converged_empty_state_is_a_noop(
        self, handler: GraphSourceHandler
    ) -> None:
        """No prior records *and* certainty about it means already converged.

        Matches the built-in Qdrant connector's ``_PointHandler.reconcile``
        exactly; the engine signals a genuinely new key with
        ``prev_may_be_missing=True``, not with this combination.
        """
        assert handler.reconcile("report.pdf", _state("fp-1"), [], False) is None


class TestReconcileDeletion:
    def test_tracked_key_emits_delete(self, handler: GraphSourceHandler) -> None:
        out = handler.reconcile("report.pdf", coco.NON_EXISTENCE, ["fp-1"], False)
        assert out is not None
        assert out.action.delete is True
        assert out.action.source_key == "report.pdf"
        assert coco.is_non_existence(out.tracking_record)

    def test_untracked_key_is_a_noop(self, handler: GraphSourceHandler) -> None:
        """Nothing of ours ever existed for this key, so nothing to clean up."""
        out = handler.reconcile("never-seen.pdf", coco.NON_EXISTENCE, [], False)
        assert out is None

    def test_possibly_missing_still_deletes(self, handler: GraphSourceHandler) -> None:
        out = handler.reconcile("report.pdf", coco.NON_EXISTENCE, [], True)
        assert out is not None
        assert out.action.delete is True


class TestApplyActions:
    async def test_upsert_actions_do_nothing(self, handler: GraphSourceHandler) -> None:
        """Graph writes already happened as a side effect during processing."""
        from ingestion.graph_target import _GraphAction

        with patch("ingestion.graph_target.remove_graph_data") as mock_remove:
            await handler._apply_actions(None, [_GraphAction("report.pdf", delete=False)])
        mock_remove.assert_not_called()

    async def test_deletes_run_once_per_key(self, handler: GraphSourceHandler) -> None:
        from ingestion.graph_target import _GraphAction

        with (
            patch("ingestion.graph_target.remove_graph_data", new=AsyncMock()) as mock_remove,
            patch("ingestion.graph_target.settings") as mock_settings,
        ):
            mock_settings.graph_enabled = False
            await handler._apply_actions(
                None,
                [
                    _GraphAction("a.pdf", delete=True),
                    _GraphAction("keep.pdf", delete=False),
                    _GraphAction("b.pdf", delete=True),
                ],
            )
        assert [c.args[0] for c in mock_remove.call_args_list] == ["a.pdf", "b.pdf"]

    async def test_graphiti_client_closed_once_per_batch(
        self, handler: GraphSourceHandler
    ) -> None:
        """v0 closed the shared client after *every* delete, mid-run."""
        from ingestion.graph_target import _GraphAction

        with (
            patch("ingestion.graph_target.remove_graph_data", new=AsyncMock()),
            patch("ingestion.graph_target.settings") as mock_settings,
            patch("ingestion.graphiti_client.close_graphiti", new=AsyncMock()) as mock_close,
        ):
            mock_settings.graph_enabled = True
            mock_settings.graph_engine = "graphiti"
            await handler._apply_actions(
                None,
                [_GraphAction("a.pdf", delete=True), _GraphAction("b.pdf", delete=True)],
            )
        assert mock_close.await_count == 1


class TestRemoveGraphData:
    async def test_routes_to_graphiti_by_default(self) -> None:
        with (
            patch("ingestion.graph_target.settings") as mock_settings,
            patch(
                "ingestion.graph_target._remove_graphiti_episodes", new=AsyncMock()
            ) as mock_graphiti,
            patch(
                "ingestion.graph_target._remove_gliner_entities", new=AsyncMock()
            ) as mock_gliner,
        ):
            mock_settings.graph_enabled = True
            mock_settings.graph_engine = "graphiti"
            await remove_graph_data("report.pdf")
        mock_graphiti.assert_awaited_once_with("report.pdf")
        mock_gliner.assert_not_awaited()

    async def test_routes_to_gliner_when_selected(self) -> None:
        with (
            patch("ingestion.graph_target.settings") as mock_settings,
            patch(
                "ingestion.graph_target._remove_graphiti_episodes", new=AsyncMock()
            ) as mock_graphiti,
            patch(
                "ingestion.graph_target._remove_gliner_entities", new=AsyncMock()
            ) as mock_gliner,
        ):
            mock_settings.graph_enabled = True
            mock_settings.graph_engine = "gliner"
            await remove_graph_data("report.pdf")
        mock_gliner.assert_awaited_once_with("report.pdf")
        mock_graphiti.assert_not_awaited()

    async def test_skipped_when_graph_disabled(self) -> None:
        with (
            patch("ingestion.graph_target.settings") as mock_settings,
            patch(
                "ingestion.graph_target._remove_graphiti_episodes", new=AsyncMock()
            ) as mock_graphiti,
        ):
            mock_settings.graph_enabled = False
            await remove_graph_data("report.pdf")
        mock_graphiti.assert_not_awaited()

    async def test_engine_failure_is_swallowed(self) -> None:
        """A cleanup failure must not abort the rest of the reconcile batch."""
        with (
            patch("ingestion.graph_target.settings") as mock_settings,
            patch(
                "ingestion.graph_target._remove_graphiti_episodes",
                new=AsyncMock(side_effect=RuntimeError("neo4j down")),
            ),
        ):
            mock_settings.graph_enabled = True
            mock_settings.graph_engine = "graphiti"
            await remove_graph_data("report.pdf")  # must not raise
