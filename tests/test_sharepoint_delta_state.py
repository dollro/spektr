from __future__ import annotations

from pathlib import Path

from services.sharepoint_sync.delta_state import DeltaState


def test_initial_token_is_none(tmp_path: Path) -> None:
    state = DeltaState(tmp_path / "delta.json")
    assert state.load() is None


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    state = DeltaState(tmp_path / "delta.json")
    state.save("https://graph.microsoft.com/.../delta?token=abc")
    assert state.load() == "https://graph.microsoft.com/.../delta?token=abc"


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    state = DeltaState(tmp_path / "nested" / "deep" / "delta.json")
    state.save("token")
    assert (tmp_path / "nested" / "deep" / "delta.json").exists()


def test_atomic_write_no_partial_files(tmp_path: Path) -> None:
    state = DeltaState(tmp_path / "delta.json")
    state.save("v1")
    state.save("v2")
    assert state.load() == "v2"
    assert not list(tmp_path.glob("*.tmp"))
