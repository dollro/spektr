from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx

from .models import DeltaItem

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class GraphClient:
    """Minimal async Microsoft Graph client for SharePoint drive delta + download."""

    def __init__(
        self,
        *,
        drive_id: str,
        token_provider: Callable[[], str],
        timeout: float = 60.0,
    ) -> None:
        self._drive_id = drive_id
        self._token = token_provider
        self._timeout = timeout
        self.next_delta_link: str | None = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token()}"}

    async def iter_delta(
        self, initial_url: str | None = None
    ) -> AsyncIterator[DeltaItem]:
        items = await self._fetch_all_delta_items(initial_url)
        for item in items:
            yield item

    async def _fetch_all_delta_items(
        self, initial_url: str | None
    ) -> list[DeltaItem]:
        url: str | None = (
            initial_url or f"{_GRAPH_BASE}/drives/{self._drive_id}/root/delta"
        )
        items: list[DeltaItem] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while url:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                payload = resp.json()
                for raw in payload.get("value", []):
                    items.append(_to_delta_item(raw))
                if "@odata.deltaLink" in payload:
                    self.next_delta_link = payload["@odata.deltaLink"]
                    return items
                url = payload.get("@odata.nextLink")
        return items

    async def download(self, download_url: str, dest_path: Path) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Graph pre-signed URLs reject Authorization headers — use a clean client.
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("GET", download_url) as resp:
                resp.raise_for_status()
                with dest_path.open("wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)


def _to_delta_item(raw: dict[str, Any]) -> DeltaItem:
    parent = (raw.get("parentReference") or {}).get("path", "")
    deleted = bool(raw.get("deleted"))
    is_folder = "folder" in raw
    etag = raw.get("cTag") or raw.get("eTag") or ""
    return DeltaItem(
        item_id=raw["id"],
        name=raw.get("name", ""),
        parent_path=parent,
        is_folder=is_folder,
        is_deleted=deleted,
        etag=etag,
        download_url=raw.get("@microsoft.graph.downloadUrl"),
    )
