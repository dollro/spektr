"""Integration test: verify file deletion cleans up Qdrant.

Requires Docker services (Qdrant). Run with:
    uv run pytest tests/test_integration_delete.py -m integration
"""

from __future__ import annotations

import pytest

from ingestion.target_connector import RagTarget, RagTargetConnector


@pytest.mark.integration
class TestDeletionIntegration:
    def test_delete_nonexistent_is_idempotent(self) -> None:
        """Deleting a file that was never ingested should not error."""
        spec = RagTarget(qdrant_url="http://localhost:6333")
        # Should not raise
        RagTargetConnector.mutate((spec, {"nonexistent.pdf": None}))
