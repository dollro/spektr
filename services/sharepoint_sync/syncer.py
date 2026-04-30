from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .delta_state import DeltaState
from .local_index import LocalIndex
from .models import DeltaItem
from .scope import is_in_scope, to_relative_path

log = logging.getLogger(__name__)


class GraphLike(Protocol):
    next_delta_link: str | None

    def iter_delta(
        self, initial_url: str | None = None
    ) -> AsyncIterator[DeltaItem]: ...

    async def download(self, url: str, dest: Path) -> None: ...


@dataclass
class Syncer:
    mirror_root: Path
    root_folder_path: str
    graph: GraphLike
    state: DeltaState
    index: LocalIndex

    async def run_once(self) -> None:
        initial = self.state.load()
        async for item in self.graph.iter_delta(initial_url=initial):
            await self._apply(item)
        if self.graph.next_delta_link:
            self.state.save(self.graph.next_delta_link)

    async def _apply(self, item: DeltaItem) -> None:
        previous_local = self.index.get_path(item.item_id)

        if item.is_deleted:
            if item.is_folder and previous_local:
                self._remove_descendants(previous_local)
            if previous_local:
                self._remove_local(previous_local)
            self.index.delete(item.item_id)
            return

        if item.is_folder:
            return  # folders are implicit; their files arrive individually

        in_scope = is_in_scope(item.parent_path, item.name, self.root_folder_path)
        if not in_scope:
            if previous_local:
                self._remove_local(previous_local)
                self.index.delete(item.item_id)
            return

        rel = to_relative_path(item.parent_path, item.name, self.root_folder_path)
        if previous_local == rel and self.index.get_etag(item.item_id) == item.etag:
            return  # no-op

        if previous_local and previous_local != rel:
            self._remove_local(previous_local)

        if item.download_url is None:
            log.warning("in-scope item %s has no download url; skipping", item.item_id)
            return
        await self.graph.download(item.download_url, self.mirror_root / rel)
        self.index.upsert(item.item_id, rel, etag=item.etag)

    def _remove_descendants(self, folder_rel: str) -> None:
        for child_id, child_path in list(self.index.iter_under(folder_rel)):
            self._remove_local(child_path)
            self.index.delete(child_id)

    def _remove_local(self, rel: str) -> None:
        target = self.mirror_root / rel
        if target.is_file():
            target.unlink()
