from __future__ import annotations

from pathlib import Path

from services.sharepoint_sync.local_index import LocalIndex


def test_upsert_and_lookup(tmp_path: Path) -> None:
    idx = LocalIndex(tmp_path / "index.sqlite")
    idx.upsert("ITEM1", "a/b.pdf", etag="e1")
    assert idx.get_path("ITEM1") == "a/b.pdf"
    assert idx.get_etag("ITEM1") == "e1"


def test_upsert_replaces_path_on_rename(tmp_path: Path) -> None:
    idx = LocalIndex(tmp_path / "index.sqlite")
    idx.upsert("ITEM1", "old/name.pdf", etag="e1")
    idx.upsert("ITEM1", "new/name.pdf", etag="e2")
    assert idx.get_path("ITEM1") == "new/name.pdf"
    assert idx.get_etag("ITEM1") == "e2"


def test_delete_removes_row(tmp_path: Path) -> None:
    idx = LocalIndex(tmp_path / "index.sqlite")
    idx.upsert("ITEM1", "a.pdf", etag="e1")
    idx.delete("ITEM1")
    assert idx.get_path("ITEM1") is None


def test_iter_under_returns_descendants_only(tmp_path: Path) -> None:
    idx = LocalIndex(tmp_path / "index.sqlite")
    idx.upsert("A", "folder/a.pdf", etag="x")
    idx.upsert("B", "folder/sub/b.pdf", etag="x")
    idx.upsert("C", "other/c.pdf", etag="x")
    idx.upsert("D", "folderX/d.pdf", etag="x")  # substring trap
    descendants = sorted(idx.iter_under("folder"))
    assert descendants == [("A", "folder/a.pdf"), ("B", "folder/sub/b.pdf")]


def test_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "index.sqlite"
    LocalIndex(db).upsert("X", "x.pdf", etag="e")
    assert LocalIndex(db).get_path("X") == "x.pdf"


def test_get_path_unknown_id_returns_none(tmp_path: Path) -> None:
    idx = LocalIndex(tmp_path / "index.sqlite")
    assert idx.get_path("nope") is None
    assert idx.get_etag("nope") is None
