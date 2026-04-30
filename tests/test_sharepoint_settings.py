from __future__ import annotations

import pytest

from config.settings import Settings

_REQUIRED_VARS = (
    "SHAREPOINT_TENANT_ID",
    "SHAREPOINT_CLIENT_ID",
    "SHAREPOINT_CLIENT_SECRET",
    "SHAREPOINT_SITE_ID",
    "SHAREPOINT_DRIVE_ID",
    "SHAREPOINT_ROOT_FOLDER_PATH",
)


def _clear_sharepoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _REQUIRED_VARS:
        monkeypatch.delenv(key, raising=False)
    for key in (
        "SHAREPOINT_LOCAL_SUBDIR",
        "SHAREPOINT_SYNC_INTERVAL_SECONDS",
        "SHAREPOINT_STATE_DIR",
    ):
        monkeypatch.delenv(key, raising=False)


def test_sharepoint_defaults_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sharepoint_env(monkeypatch)
    s = Settings(neo4j_password="x")
    assert s.sharepoint_tenant_id == ""
    assert s.sharepoint_client_id == ""
    assert s.sharepoint_client_secret == ""
    assert s.sharepoint_site_id == ""
    assert s.sharepoint_drive_id == ""
    assert s.sharepoint_root_folder_path == ""
    assert s.sharepoint_local_subdir == "sharepoint"
    assert s.sharepoint_sync_interval_seconds == 180
    assert s.sharepoint_state_dir == "state/sharepoint"
    assert s.sharepoint_enabled is False


def test_sharepoint_enabled_when_all_required_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHAREPOINT_TENANT_ID", "tenant")
    monkeypatch.setenv("SHAREPOINT_CLIENT_ID", "client")
    monkeypatch.setenv("SHAREPOINT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "site")
    monkeypatch.setenv("SHAREPOINT_DRIVE_ID", "drive")
    monkeypatch.setenv("SHAREPOINT_ROOT_FOLDER_PATH", "/Engineering/Specs")
    s = Settings(neo4j_password="x")
    assert s.sharepoint_enabled is True


def test_sharepoint_disabled_when_any_required_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sharepoint_env(monkeypatch)
    monkeypatch.setenv("SHAREPOINT_TENANT_ID", "tenant")
    s = Settings(neo4j_password="x")
    assert s.sharepoint_enabled is False
