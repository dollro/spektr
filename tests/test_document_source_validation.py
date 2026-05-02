"""Validation rules for the DOCUMENT_SOURCE setting.

Each value (`local`, `s3`, `sharepoint`) carries a contract with the
ingestion pipeline. The settings validator must reject misconfigurations
loudly so they don't manifest as silent ingestion failures in prod.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import Settings


def _clear_source_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "DOCUMENT_SOURCE",
        "S3_BUCKET_NAME",
        "S3_SQS_QUEUE_URL",
        "SHAREPOINT_TENANT_ID",
        "SHAREPOINT_CLIENT_ID",
        "SHAREPOINT_CLIENT_SECRET",
        "SHAREPOINT_SITE_ID",
        "SHAREPOINT_DRIVE_ID",
        "SHAREPOINT_ROOT_FOLDER_PATH",
    ):
        monkeypatch.delenv(key, raising=False)


def _set_sharepoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHAREPOINT_TENANT_ID", "t")
    monkeypatch.setenv("SHAREPOINT_CLIENT_ID", "c")
    monkeypatch.setenv("SHAREPOINT_CLIENT_SECRET", "s")
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "site")
    monkeypatch.setenv("SHAREPOINT_DRIVE_ID", "d")
    monkeypatch.setenv("SHAREPOINT_ROOT_FOLDER_PATH", "/Specs")


def test_document_source_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_source_env(monkeypatch)
    s = Settings(_env_file=None, neo4j_password="x")  # type: ignore[call-arg]
    assert s.document_source == "local"


def test_document_source_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_env(monkeypatch)
    monkeypatch.setenv("DOCUMENT_SOURCE", "ftp")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, neo4j_password="x")  # type: ignore[call-arg]


def test_s3_source_requires_bucket_and_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_source_env(monkeypatch)
    monkeypatch.setenv("DOCUMENT_SOURCE", "s3")
    with pytest.raises(ValidationError, match="S3_BUCKET_NAME"):
        Settings(_env_file=None, neo4j_password="x")  # type: ignore[call-arg]


def test_s3_source_passes_when_both_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_source_env(monkeypatch)
    monkeypatch.setenv("DOCUMENT_SOURCE", "s3")
    monkeypatch.setenv("S3_BUCKET_NAME", "bucket")
    monkeypatch.setenv("S3_SQS_QUEUE_URL", "https://sqs/q")
    s = Settings(_env_file=None, neo4j_password="x")  # type: ignore[call-arg]
    assert s.document_source == "s3"


def test_sharepoint_source_requires_all_sharepoint_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_env(monkeypatch)
    monkeypatch.setenv("DOCUMENT_SOURCE", "sharepoint")
    monkeypatch.setenv("SHAREPOINT_TENANT_ID", "t")  # only one of six
    with pytest.raises(ValidationError, match="SHAREPOINT_"):
        Settings(_env_file=None, neo4j_password="x")  # type: ignore[call-arg]


def test_sharepoint_source_passes_when_all_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_source_env(monkeypatch)
    monkeypatch.setenv("DOCUMENT_SOURCE", "sharepoint")
    _set_sharepoint(monkeypatch)
    s = Settings(_env_file=None, neo4j_password="x")  # type: ignore[call-arg]
    assert s.document_source == "sharepoint"
    assert s.sharepoint_enabled is True


def test_local_source_accepts_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """DOCUMENT_SOURCE=local must not require S3 or SharePoint vars."""
    _clear_source_env(monkeypatch)
    monkeypatch.setenv("DOCUMENT_SOURCE", "local")
    s = Settings(_env_file=None, neo4j_password="x")  # type: ignore[call-arg]
    assert s.document_source == "local"


def test_local_source_with_sharepoint_vars_does_not_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vestigial SharePoint vars under DOCUMENT_SOURCE=local are tolerated.

    The runtime guard in services/sharepoint_sync/main.py refuses to start
    the syncer in this case, which is the load-bearing safety net.
    """
    _clear_source_env(monkeypatch)
    monkeypatch.setenv("DOCUMENT_SOURCE", "local")
    _set_sharepoint(monkeypatch)
    s = Settings(_env_file=None, neo4j_password="x")  # type: ignore[call-arg]
    assert s.document_source == "local"
    assert s.sharepoint_enabled is True  # vars are present, just not the active source
