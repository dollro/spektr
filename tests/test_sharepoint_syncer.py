from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from services.sharepoint_sync.delta_state import DeltaState
from services.sharepoint_sync.local_index import LocalIndex
from services.sharepoint_sync.models import DeltaItem
from services.sharepoint_sync.syncer import Syncer


class FakeGraph:
    def __init__(self, batch: list[DeltaItem], final_token: str) -> None:
        self._batch = batch
        self.next_delta_link: str | None = None
        self._final_token = final_token
        self.downloads: list[tuple[str, Path]] = []
        self.last_initial_url: str | None = None

    async def iter_delta(self, initial_url: str | None = None) -> AsyncIterator[DeltaItem]:
        self.last_initial_url = initial_url
        for item in self._batch:
            yield item
        self.next_delta_link = self._final_token

    async def download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f"BYTES:{url}".encode())
        self.downloads.append((url, dest))


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, DeltaState, LocalIndex]:
    mirror = tmp_path / "documents" / "sharepoint"
    state = DeltaState(tmp_path / "delta.json")
    index = LocalIndex(tmp_path / "index.sqlite")
    return mirror, state, index


def _make_item(
    *,
    item_id: str = "I1",
    name: str = "doc.pdf",
    parent_path: str = "/drive/root:/Specs",
    is_folder: bool = False,
    is_deleted: bool = False,
    etag: str = "e1",
    download_url: str | None = "https://dl/doc.pdf",
) -> DeltaItem:
    return DeltaItem(
        item_id=item_id,
        name=name,
        parent_path=parent_path,
        is_folder=is_folder,
        is_deleted=is_deleted,
        etag=etag,
        download_url=download_url,
    )


async def test_new_file_in_scope_is_downloaded(
    workspace: tuple[Path, DeltaState, LocalIndex],
) -> None:
    mirror, state, index = workspace
    fake = FakeGraph([_make_item()], final_token="t1")
    syncer = Syncer(
        mirror_root=mirror,
        root_folder_path="/Specs",
        graph=fake,
        state=state,
        index=index,
    )
    await syncer.run_once()
    assert (mirror / "doc.pdf").read_bytes() == b"BYTES:https://dl/doc.pdf"
    assert index.get_path("I1") == "doc.pdf"
    assert state.load() == "t1"


async def test_deleted_file_removes_local_copy(
    workspace: tuple[Path, DeltaState, LocalIndex],
) -> None:
    mirror, state, index = workspace
    mirror.mkdir(parents=True)
    (mirror / "doc.pdf").write_bytes(b"old")
    index.upsert("I1", "doc.pdf", etag="e1")
    fake = FakeGraph(
        [_make_item(is_deleted=True, etag="", download_url=None)], final_token="t2"
    )
    syncer = Syncer(
        mirror_root=mirror,
        root_folder_path="/Specs",
        graph=fake,
        state=state,
        index=index,
    )
    await syncer.run_once()
    assert not (mirror / "doc.pdf").exists()
    assert index.get_path("I1") is None


async def test_move_out_of_scope_removes_local_copy(
    workspace: tuple[Path, DeltaState, LocalIndex],
) -> None:
    mirror, state, index = workspace
    mirror.mkdir(parents=True)
    (mirror / "doc.pdf").write_bytes(b"old")
    index.upsert("I1", "doc.pdf", etag="e1")
    fake = FakeGraph(
        [_make_item(parent_path="/drive/root:/Marketing", etag="e2")],
        final_token="t3",
    )
    syncer = Syncer(
        mirror_root=mirror,
        root_folder_path="/Specs",
        graph=fake,
        state=state,
        index=index,
    )
    await syncer.run_once()
    assert not (mirror / "doc.pdf").exists()
    assert index.get_path("I1") is None
    assert fake.downloads == []


async def test_rename_within_scope_moves_file(
    workspace: tuple[Path, DeltaState, LocalIndex],
) -> None:
    mirror, state, index = workspace
    mirror.mkdir(parents=True)
    (mirror / "old.pdf").write_bytes(b"old")
    index.upsert("I1", "old.pdf", etag="e1")
    fake = FakeGraph(
        [_make_item(name="new.pdf", etag="e2", download_url="https://dl/new.pdf")],
        final_token="t4",
    )
    syncer = Syncer(
        mirror_root=mirror,
        root_folder_path="/Specs",
        graph=fake,
        state=state,
        index=index,
    )
    await syncer.run_once()
    assert not (mirror / "old.pdf").exists()
    assert (mirror / "new.pdf").exists()
    assert index.get_path("I1") == "new.pdf"


async def test_unchanged_etag_is_skipped(
    workspace: tuple[Path, DeltaState, LocalIndex],
) -> None:
    mirror, state, index = workspace
    mirror.mkdir(parents=True)
    (mirror / "doc.pdf").write_bytes(b"current")
    index.upsert("I1", "doc.pdf", etag="e1")
    fake = FakeGraph([_make_item(etag="e1")], final_token="t5")
    syncer = Syncer(
        mirror_root=mirror,
        root_folder_path="/Specs",
        graph=fake,
        state=state,
        index=index,
    )
    await syncer.run_once()
    assert (mirror / "doc.pdf").read_bytes() == b"current"
    assert fake.downloads == []


async def test_folder_deletion_removes_descendants(
    workspace: tuple[Path, DeltaState, LocalIndex],
) -> None:
    mirror, state, index = workspace
    mirror.mkdir(parents=True)
    (mirror / "subdir").mkdir()
    (mirror / "subdir" / "a.pdf").write_bytes(b"a")
    (mirror / "subdir" / "b.pdf").write_bytes(b"b")
    index.upsert("A", "subdir/a.pdf", etag="ea")
    index.upsert("B", "subdir/b.pdf", etag="eb")
    index.upsert("F", "subdir", etag="ef")
    fake = FakeGraph(
        [
            _make_item(
                item_id="F",
                name="subdir",
                is_folder=True,
                is_deleted=True,
                etag="",
                download_url=None,
            )
        ],
        final_token="t6",
    )
    syncer = Syncer(
        mirror_root=mirror,
        root_folder_path="/Specs",
        graph=fake,
        state=state,
        index=index,
    )
    await syncer.run_once()
    assert not (mirror / "subdir" / "a.pdf").exists()
    assert not (mirror / "subdir" / "b.pdf").exists()
    assert index.get_path("A") is None
    assert index.get_path("B") is None
    assert index.get_path("F") is None


async def test_resume_passes_stored_token_as_initial_url(
    workspace: tuple[Path, DeltaState, LocalIndex],
) -> None:
    mirror, state, index = workspace
    state.save("https://graph.microsoft.com/.../delta?token=resume")
    fake = FakeGraph([], final_token="t7")
    syncer = Syncer(
        mirror_root=mirror,
        root_folder_path="/Specs",
        graph=fake,
        state=state,
        index=index,
    )
    await syncer.run_once()
    assert fake.last_initial_url == "https://graph.microsoft.com/.../delta?token=resume"
    assert state.load() == "t7"


async def test_out_of_scope_unindexed_file_is_ignored(
    workspace: tuple[Path, DeltaState, LocalIndex],
) -> None:
    mirror, state, index = workspace
    fake = FakeGraph(
        [_make_item(item_id="NEW", parent_path="/drive/root:/Marketing")],
        final_token="t8",
    )
    syncer = Syncer(
        mirror_root=mirror,
        root_folder_path="/Specs",
        graph=fake,
        state=state,
        index=index,
    )
    await syncer.run_once()
    assert fake.downloads == []
    assert index.get_path("NEW") is None
