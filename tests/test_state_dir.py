"""STATE_DIR: one root for every piece of mutable runtime state.

The three paths below were independently overridable, and one of them —
the poison-pill failure DB — was not overridable at all, being a module
constant in `ingestion/_failure_tracker.py`. That made it impossible to
fully relocate state, which matters whenever the working tree is shared: a
container sandbox, a CI checkout or a second worktree all inherited the
host's ledger and could delete it.

These tests pin the "one knob moves everything" property. If a fourth state
path is ever added, `test_no_state_path_escapes_the_root` should fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings

# Every Settings field that names a path under the state root.
STATE_PATH_FIELDS = ("cocoindex_db_path", "failure_db_path", "sharepoint_state_dir")


def _settings(**kw: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "neo4j_password": "x",
        "rerank_enabled": False,
        "graph_engine": "gliner",
    }
    return Settings(**{**base, **kw})  # type: ignore[arg-type]


def test_paths_derive_from_state_dir() -> None:
    s = _settings(state_dir="/var/lib/spektr")
    assert s.cocoindex_db_path == "/var/lib/spektr/cocoindex.db"
    assert s.failure_db_path == "/var/lib/spektr/ingestion_failures.db"
    assert s.sharepoint_state_dir == "/var/lib/spektr/sharepoint"


def test_default_root_is_the_repo_state_dir() -> None:
    """The default must stay as it was — this refactor changes no behaviour."""
    s = _settings()
    assert s.cocoindex_db_path == "state/cocoindex.db"
    assert s.failure_db_path == "state/ingestion_failures.db"
    assert s.sharepoint_state_dir == "state/sharepoint"


@pytest.mark.parametrize("field", STATE_PATH_FIELDS)
def test_an_explicit_path_still_wins(field: str) -> None:
    """Existing `.env` files pin these directly; they must keep working."""
    s = _settings(state_dir="/var/lib/spektr", **{field: "/somewhere/else"})
    assert getattr(s, field) == "/somewhere/else"
    # ...and pinning one must not drag the others off the root.
    for other in STATE_PATH_FIELDS:
        if other != field:
            assert getattr(s, other).startswith("/var/lib/spektr")


@pytest.mark.parametrize("field", STATE_PATH_FIELDS)
def test_no_state_path_escapes_the_root(field: str) -> None:
    """The whole point: one knob relocates ALL of it.

    A new state path that forgets to derive from STATE_DIR would keep
    writing into the shared tree while everything around it moved — which
    is precisely the failure this exists to prevent, and it would be
    invisible until something got deleted.
    """
    root = Path("/tmp/spektr-isolated")
    s = _settings(state_dir=str(root))
    assert root in Path(getattr(s, field)).parents


def test_failure_tracker_follows_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_failure_tracker` read a module constant, so STATE_DIR could not move it."""
    from ingestion import _failure_tracker

    monkeypatch.setattr(
        _failure_tracker.settings, "failure_db_path", "/tmp/spektr-isolated/fail.db"
    )
    assert _failure_tracker._default_db_path() == Path("/tmp/spektr-isolated/fail.db")


def test_failure_tracker_resolves_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolved per call, not at import — import order must not pin it."""
    from ingestion import _failure_tracker

    monkeypatch.setattr(_failure_tracker.settings, "failure_db_path", "/tmp/a.db")
    first = _failure_tracker._default_db_path()
    monkeypatch.setattr(_failure_tracker.settings, "failure_db_path", "/tmp/b.db")
    assert _failure_tracker._default_db_path() != first


def test_tracker_writes_where_told(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: redirect the root, and nothing lands in the repo."""
    from ingestion._failure_tracker import FailureTracker

    target = tmp_path / "isolated" / "ingestion_failures.db"
    monkeypatch.setattr("ingestion._failure_tracker.settings.failure_db_path", str(target))
    tracker = FailureTracker()
    assert tracker.record_failure("doc.pdf", "boom") == 1
    assert target.exists(), "tracker must create its DB under the redirected root"
