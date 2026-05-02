from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeltaItem:
    """Normalized projection of a Microsoft Graph delta entry that we care about."""

    item_id: str
    name: str
    parent_path: str  # e.g. "/drive/root:/Engineering/Specs"
    is_folder: bool
    is_deleted: bool
    etag: str  # cTag preferred (changes on content edit), falls back to eTag
    download_url: str | None  # @microsoft.graph.downloadUrl; None for folders/deletes
