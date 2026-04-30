"""Runtime guards in the sharepoint-sync entrypoint.

The syncer must refuse to run when DOCUMENT_SOURCE is anything other than
'sharepoint', because in those cases its mirror dir would not be the source
of CocoIndex's ingestion — files would be downloaded to disk and never
ingested.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services.sharepoint_sync import main


async def test_run_loop_exits_when_document_source_is_local() -> None:
    fake_settings = type(
        "S",
        (),
        {
            "document_source": "local",
            "sharepoint_enabled": True,
        },
    )()
    with patch.object(main, "settings", fake_settings):
        with pytest.raises(SystemExit) as excinfo:
            await main.run_loop(once=True)
    assert excinfo.value.code == 2


async def test_run_loop_exits_when_document_source_is_s3() -> None:
    fake_settings = type(
        "S",
        (),
        {
            "document_source": "s3",
            "sharepoint_enabled": True,
        },
    )()
    with patch.object(main, "settings", fake_settings):
        with pytest.raises(SystemExit) as excinfo:
            await main.run_loop(once=True)
    assert excinfo.value.code == 2


async def test_run_loop_exits_when_sharepoint_not_enabled() -> None:
    fake_settings = type(
        "S",
        (),
        {
            "document_source": "sharepoint",
            "sharepoint_enabled": False,
        },
    )()
    with patch.object(main, "settings", fake_settings):
        with pytest.raises(SystemExit) as excinfo:
            await main.run_loop(once=True)
    assert excinfo.value.code == 2
