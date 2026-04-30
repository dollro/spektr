from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Callable
from pathlib import Path

from azure.identity.aio import ClientSecretCredential

from config.logging import setup_logging
from config.observability import setup_observability
from config.settings import settings

from .delta_state import DeltaState
from .graph_client import GraphClient
from .local_index import LocalIndex
from .syncer import Syncer

log = logging.getLogger(__name__)

_GRAPH_SCOPE = "https://graph.microsoft.com/.default"


class _TokenCache:
    """Holds the most recent Graph access token; refreshed before each cycle."""

    def __init__(self, credential: ClientSecretCredential) -> None:
        self._credential = credential
        self._token: str = ""

    async def refresh(self) -> None:
        result = await self._credential.get_token(_GRAPH_SCOPE)
        self._token = result.token

    def get(self) -> str:
        return self._token


def _build_credential() -> ClientSecretCredential:
    return ClientSecretCredential(
        tenant_id=settings.sharepoint_tenant_id,
        client_id=settings.sharepoint_client_id,
        client_secret=settings.sharepoint_client_secret,
    )


def _build_syncer(token_provider: Callable[[], str]) -> Syncer:
    state_dir = Path(settings.sharepoint_state_dir)
    mirror_root = Path(settings.local_documents_path) / settings.sharepoint_local_subdir
    graph = GraphClient(drive_id=settings.sharepoint_drive_id, token_provider=token_provider)
    return Syncer(
        mirror_root=mirror_root,
        root_folder_path=settings.sharepoint_root_folder_path,
        graph=graph,
        state=DeltaState(state_dir / "delta.json"),
        index=LocalIndex(state_dir / "index.sqlite"),
    )


async def run_loop(*, once: bool = False) -> None:
    if not settings.sharepoint_enabled:
        log.error("SharePoint sync requested but settings.sharepoint_enabled is False")
        raise SystemExit(2)

    setup_observability()
    setup_logging()

    credential = _build_credential()
    cache = _TokenCache(credential)
    await cache.refresh()
    syncer = _build_syncer(cache.get)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    try:
        while True:
            try:
                await cache.refresh()
                await syncer.run_once()
            except Exception:
                log.exception("sharepoint sync cycle failed; will retry next interval")
            if once or stop.is_set():
                return
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=settings.sharepoint_sync_interval_seconds,
                )
            except TimeoutError:
                pass
    finally:
        await credential.close()
