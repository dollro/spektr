"""Unit tests for scripts.backup + scripts.restore — no live services."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.backup as backup
import scripts.restore as restore


@pytest.fixture
def tmp_backup_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path / "backups"


class TestMakeDest:
    def test_creates_dated_dir(self, tmp_backup_root: Path) -> None:
        dest = backup._make_dest(tmp_backup_root, timestamp="20260419-000000")
        assert dest.exists()
        assert dest.name == "20260419-000000"


class TestBackupQdrant:
    def test_skips_when_no_collections_present(self, tmp_backup_root: Path) -> None:
        dest = backup._make_dest(tmp_backup_root, timestamp="ts")
        with patch.object(backup, "_qdrant_collections", return_value=[]):
            result = backup.backup_qdrant(dest)
        assert result["status"] == "nothing-to-do"
        assert result["collections"] == []

    def test_snapshots_each_present_collection(self, tmp_backup_root: Path) -> None:
        dest = backup._make_dest(tmp_backup_root, timestamp="ts")
        # Mock Qdrant API: one collection present, snapshot + download succeed
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = False
        fake_client.post.return_value.json.return_value = {
            "result": {"name": "snap-1.snapshot"}
        }
        fake_client.post.return_value.raise_for_status = MagicMock()
        stream_ctx = MagicMock()
        stream_ctx.__enter__.return_value = stream_ctx
        stream_ctx.__exit__.return_value = False
        stream_ctx.raise_for_status = MagicMock()
        stream_ctx.iter_bytes.return_value = [b"snapshot-bytes"]
        fake_client.stream.return_value = stream_ctx

        with (
            patch("scripts.backup.httpx.Client", return_value=fake_client),
            patch.object(backup, "_qdrant_collections", return_value=["documents_dense"]),
        ):
            result = backup.backup_qdrant(dest)

        assert result["status"] == "ok"
        assert len(result["collections"]) == 1
        assert result["collections"][0]["collection"] == "documents_dense"
        # Snapshot file was written
        snap_path = dest / "qdrant" / "documents_dense_snap-1.snapshot"
        assert snap_path.exists()
        assert snap_path.read_bytes() == b"snapshot-bytes"


class TestBackupPostgres:
    def test_runs_pg_dump_and_records_size(self, tmp_backup_root: Path) -> None:
        dest = backup._make_dest(tmp_backup_root, timestamp="ts")

        def fake_run(cmd: list[str], stdout=None, stderr=None, check=False):  # type: ignore[no-untyped-def]
            # Simulate pg_dump writing to stdout
            if stdout is not None and hasattr(stdout, "write"):
                stdout.write(b"PGDMP-fake-bytes")
            return MagicMock(returncode=0, stderr=b"")

        with patch.object(subprocess, "run", side_effect=fake_run):
            result = backup.backup_postgres(dest)

        assert result["file"] == "cocoindex.dump"
        assert result["bytes"] > 0
        dump = dest / "postgres" / "cocoindex.dump"
        assert dump.read_bytes().startswith(b"PGDMP")


class TestManifest:
    def test_writes_valid_json(self, tmp_backup_root: Path) -> None:
        dest = backup._make_dest(tmp_backup_root, timestamp="ts")
        path = backup._write_manifest(dest, {"foo": "bar"})
        data = json.loads(path.read_text())
        assert data["entries"] == {"foo": "bar"}
        assert data["timestamp"] == "ts"
        assert "qdrant_url" in data


class TestRetention:
    def test_prunes_old_dirs(
        self, tmp_backup_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old = tmp_backup_root / "old"
        new = tmp_backup_root / "new"
        old.mkdir(parents=True)
        new.mkdir()
        # Backdate `old` by 10 days
        import os

        old_time = __import__("time").time() - 10 * 86400
        os.utime(old, (old_time, old_time))
        removed = backup._prune_older_than(tmp_backup_root, days=5)
        assert removed == 1
        assert not old.exists()
        assert new.exists()


class TestRestore:
    def test_refuses_without_safety_flag(self, tmp_backup_root: Path) -> None:
        src = tmp_backup_root / "ts"
        src.mkdir(parents=True)
        (src / "manifest.json").write_text(json.dumps({"entries": {}}))
        rc = restore.main(["--from", str(src), "--target", "qdrant"])
        assert rc == 3  # safety-flag missing

    def test_fails_on_missing_manifest(self, tmp_backup_root: Path) -> None:
        src = tmp_backup_root / "none"
        rc = restore.main(
            ["--from", str(src), "--yes-i-know-this-wipes-things"]
        )
        assert rc == 2

    def test_qdrant_restore_uploads_snapshot(self, tmp_backup_root: Path) -> None:
        src = tmp_backup_root / "ts"
        (src / "qdrant").mkdir(parents=True)
        snap = src / "qdrant" / "documents_dense_snap-1.snapshot"
        snap.write_bytes(b"snap")
        manifest = {
            "entries": {
                "qdrant": {
                    "collections": [
                        {
                            "collection": "documents_dense",
                            "file": "documents_dense_snap-1.snapshot",
                            "bytes": 4,
                        }
                    ]
                }
            }
        }
        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = False
        fake_client.post.return_value.status_code = 200
        with patch("scripts.restore.httpx.Client", return_value=fake_client):
            restore.restore_qdrant(src, manifest)
        # DELETE then POST upload
        assert fake_client.delete.called
        assert fake_client.post.called
