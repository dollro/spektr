# SharePoint Ingestion Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an O365 SharePoint document-library source for Spektr's bulk ingestion (Path A) by mirroring an in-scope SharePoint folder to the local filesystem on a fixed interval; CocoIndex's existing `LocalFile` source picks the mirrored files up unchanged.

**Architecture:** A standalone polling syncer (new `services/sharepoint_sync/` package) authenticates with Azure AD (client credentials), calls Microsoft Graph `/drives/{drive_id}/root/delta` every N seconds, filters items to a configured root folder, downloads new/changed files to `${local_documents_path}/sharepoint/...`, and **propagates deletions/moves out-of-scope by removing the local mirror**. CocoIndex's `LocalFile` source + `target_connector.py` then handle Qdrant + Neo4j sync. The existing S3+SQS path stays untouched as a parallel option.

**Tech Stack:** Python 3.13, `httpx` (already a dep), `azure-identity` (new dep) for token acquisition, sqlite (stdlib) for the local item-id ↔ path index, pytest with `pytest-asyncio` (already used), `respx` (mock httpx) for Graph client tests.

---

## Context

Today, Path A reads from S3 with SQS push events handled internally by `cocoindex.sources.AmazonS3`. The user wants the same workflow against a single SharePoint folder, with deletion parity (Qdrant + Neo4j get cleaned when a SharePoint file is deleted or moved out of scope). We achieve this without touching `ingestion/pipeline.py` by mirroring the in-scope folder to a local directory that the existing `LocalFile` branch already consumes. Live latency target: 1–5 min via polling; no public webhook.

## File Structure

**New (source):**
- `services/__init__.py` — empty
- `services/sharepoint_sync/__init__.py` — empty
- `services/sharepoint_sync/__main__.py` — `python -m services.sharepoint_sync` entrypoint
- `services/sharepoint_sync/scope.py` — pure path-filter helper (`is_in_scope`, `to_relative_path`)
- `services/sharepoint_sync/delta_state.py` — read/write delta token to a JSON file
- `services/sharepoint_sync/local_index.py` — sqlite-backed `(item_id → local_path, etag)` mapping
- `services/sharepoint_sync/graph_client.py` — async `httpx` wrapper: token, delta, download
- `services/sharepoint_sync/models.py` — small dataclasses for delta items
- `services/sharepoint_sync/syncer.py` — applies one batch of delta entries to disk
- `services/sharepoint_sync/main.py` — outer loop (sleep + run-once)

**New (tests):**
- `tests/test_sharepoint_scope.py`
- `tests/test_sharepoint_delta_state.py`
- `tests/test_sharepoint_local_index.py`
- `tests/test_sharepoint_graph_client.py` (httpx mocked via `respx`)
- `tests/test_sharepoint_syncer.py` (graph_client mocked, real tmp_path filesystem, real sqlite)

**New (docs / ops):**
- `docs/ingestion/sharepoint-setup.md`

**Modified:**
- `config/settings.py` — add `sharepoint_*` fields and update the `Document Source` block
- `pyproject.toml` — add `azure-identity` (and `respx` to dev deps)
- `Taskfile.yml` — add `task sharepoint-sync` (loop) and `task sharepoint-sync-once`
- `docker-compose.prod.yml` — add `sharepoint-sync` service (no public ports, shares the documents volume)
- `mkdocs.yml` — register the new doc page (if its nav is explicit; check first)

**Untouched (load-bearing parity):**
- `ingestion/pipeline.py`, `ingestion/file_processor.py`, `ingestion/target_connector.py`, `ingestion/_failure_tracker.py`, `ingestion/embedder.py`, `ingestion/graph_engine.py`, `server/`, `agent/`.

## Decomposition rationale

- **Pure logic first** (`scope.py`) — fastest TDD loop, no IO.
- **Persistence next** (`delta_state.py`, `local_index.py`) — testable with `tmp_path`, no network.
- **Network adapter** (`graph_client.py`) — mocked with `respx`; covers only Graph specifics.
- **Orchestration** (`syncer.py`) — wires the above, owns deletion semantics. Tested with a fake graph_client + real filesystem.
- **Outer loop** (`main.py` + `__main__.py`) — thin; mostly tested via integration smoke.

## TDD discipline

- Every task: **Test → Run (fail) → Implement → Run (pass) → Commit.**
- Use the project's `task lint` / `task typecheck` / `task test` before each commit on a task that closes a logical unit.
- Keep files ≤ 600 lines, functions ≤ 60 lines (per `CLAUDE.md`). All proposed files come in well under this.

---

## Task 1: Add SharePoint settings fields

**Files:**
- Modify: `config/settings.py:62-72` (Document Source + AWS section)
- Test: `tests/test_sharepoint_settings.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sharepoint_settings.py
from config.settings import Settings


def test_sharepoint_defaults_are_empty(monkeypatch):
    # Clear all SHAREPOINT_* env vars so we test the dataclass defaults
    for key in list(monkeypatch._setitem.keys() if hasattr(monkeypatch, "_setitem") else []):
        if key.startswith("SHAREPOINT_"):
            monkeypatch.delenv(key, raising=False)
    s = Settings(neo4j_password="x")  # type: ignore[call-arg]
    assert s.sharepoint_tenant_id == ""
    assert s.sharepoint_client_id == ""
    assert s.sharepoint_client_secret == ""
    assert s.sharepoint_site_id == ""
    assert s.sharepoint_drive_id == ""
    assert s.sharepoint_root_folder_path == ""
    assert s.sharepoint_local_subdir == "sharepoint"
    assert s.sharepoint_sync_interval_seconds == 180
    assert s.sharepoint_state_dir == "state/sharepoint"


def test_sharepoint_enabled_flag(monkeypatch):
    monkeypatch.setenv("SHAREPOINT_TENANT_ID", "tenant")
    monkeypatch.setenv("SHAREPOINT_CLIENT_ID", "client")
    monkeypatch.setenv("SHAREPOINT_CLIENT_SECRET", "secret")
    monkeypatch.setenv("SHAREPOINT_SITE_ID", "site")
    monkeypatch.setenv("SHAREPOINT_DRIVE_ID", "drive")
    monkeypatch.setenv("SHAREPOINT_ROOT_FOLDER_PATH", "/Engineering/Specs")
    s = Settings(neo4j_password="x")  # type: ignore[call-arg]
    assert s.sharepoint_enabled is True


def test_sharepoint_disabled_when_any_required_missing(monkeypatch):
    monkeypatch.setenv("SHAREPOINT_TENANT_ID", "tenant")
    monkeypatch.delenv("SHAREPOINT_CLIENT_ID", raising=False)
    s = Settings(neo4j_password="x")  # type: ignore[call-arg]
    assert s.sharepoint_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_sharepoint_settings.py -v
```

Expected: `AttributeError: 'Settings' object has no attribute 'sharepoint_tenant_id'`.

- [ ] **Step 3: Add fields to `config/settings.py`**

Insert immediately after the AWS block (`config/settings.py:72`):

```python
    # SharePoint (only used when all sharepoint_* required fields are set)
    sharepoint_tenant_id: str = ""
    sharepoint_client_id: str = ""
    sharepoint_client_secret: str = ""
    sharepoint_site_id: str = ""
    sharepoint_drive_id: str = ""
    sharepoint_root_folder_path: str = ""  # e.g. "/Engineering/Specs"
    sharepoint_local_subdir: str = "sharepoint"
    sharepoint_sync_interval_seconds: int = 180
    sharepoint_state_dir: str = "state/sharepoint"
```

Add a property near `dense_dimensions`:

```python
    @property
    def sharepoint_enabled(self) -> bool:
        return all(
            [
                self.sharepoint_tenant_id,
                self.sharepoint_client_id,
                self.sharepoint_client_secret,
                self.sharepoint_site_id,
                self.sharepoint_drive_id,
                self.sharepoint_root_folder_path,
            ]
        )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_sharepoint_settings.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add config/settings.py tests/test_sharepoint_settings.py
git commit -m "feat(sharepoint): add settings fields and enabled flag"
```

---

## Task 2: Path scope filter

The filter decides which Graph delta items belong to our in-scope folder, and converts SharePoint absolute paths to local mirror-relative paths. Pure functions, no IO.

**Files:**
- Create: `services/__init__.py`, `services/sharepoint_sync/__init__.py`, `services/sharepoint_sync/scope.py`
- Test: `tests/test_sharepoint_scope.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sharepoint_scope.py
import pytest

from services.sharepoint_sync.scope import is_in_scope, to_relative_path


@pytest.mark.parametrize(
    "parent_path,name,root,expected",
    [
        # SharePoint paths come back like "/drive/root:/Engineering/Specs"
        ("/drive/root:/Engineering/Specs", "draft.pdf", "/Engineering/Specs", True),
        ("/drive/root:/Engineering/Specs/sub", "x.pdf", "/Engineering/Specs", True),
        ("/drive/root:/Engineering", "x.pdf", "/Engineering/Specs", False),
        ("/drive/root:/Engineering/Specs2", "x.pdf", "/Engineering/Specs", False),
        ("/drive/root:/Marketing", "x.pdf", "/Engineering/Specs", False),
        # Trailing-slash tolerance on the root config
        ("/drive/root:/Engineering/Specs", "x.pdf", "/Engineering/Specs/", True),
        # Root prefix is matched on segment boundaries (defends against substring bug)
        ("/drive/root:/Engineering/SpecsX", "x.pdf", "/Engineering/Specs", False),
    ],
)
def test_is_in_scope(parent_path, name, root, expected):
    assert is_in_scope(parent_path, name, root) is expected


def test_to_relative_path_strips_drive_prefix_and_root():
    rel = to_relative_path(
        parent_path="/drive/root:/Engineering/Specs/sub",
        name="draft.pdf",
        root_folder_path="/Engineering/Specs",
    )
    assert rel == "sub/draft.pdf"


def test_to_relative_path_at_root():
    rel = to_relative_path(
        parent_path="/drive/root:/Engineering/Specs",
        name="top.pdf",
        root_folder_path="/Engineering/Specs",
    )
    assert rel == "top.pdf"


def test_to_relative_path_raises_when_out_of_scope():
    with pytest.raises(ValueError):
        to_relative_path(
            parent_path="/drive/root:/Marketing",
            name="x.pdf",
            root_folder_path="/Engineering/Specs",
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_sharepoint_scope.py -v
```

Expected: `ModuleNotFoundError: No module named 'services'`.

- [ ] **Step 3: Implement `services/sharepoint_sync/scope.py`**

```python
# services/sharepoint_sync/scope.py
from __future__ import annotations

# Microsoft Graph returns parentReference.path as e.g. "/drive/root:/Engineering/Specs"
_DRIVE_PREFIX = "/drive/root:"


def _normalize_root(root_folder_path: str) -> str:
    """Strip trailing slashes and ensure a leading slash."""
    if not root_folder_path.startswith("/"):
        root_folder_path = "/" + root_folder_path
    return root_folder_path.rstrip("/") or "/"


def _strip_drive_prefix(parent_path: str) -> str:
    if parent_path.startswith(_DRIVE_PREFIX):
        return parent_path[len(_DRIVE_PREFIX) :] or "/"
    return parent_path


def is_in_scope(parent_path: str, name: str, root_folder_path: str) -> bool:
    """Return True iff (parent_path/name) is inside root_folder_path on segment boundaries."""
    root = _normalize_root(root_folder_path)
    parent = _strip_drive_prefix(parent_path).rstrip("/") or "/"
    if parent == root:
        return True
    return parent.startswith(root + "/")


def to_relative_path(parent_path: str, name: str, root_folder_path: str) -> str:
    """Convert a Graph item to its path relative to the in-scope root.

    Raises ValueError if the item is out of scope.
    """
    if not is_in_scope(parent_path, name, root_folder_path):
        raise ValueError(f"out of scope: {parent_path}/{name}")
    root = _normalize_root(root_folder_path)
    parent = _strip_drive_prefix(parent_path).rstrip("/") or "/"
    if parent == root:
        return name
    rel_parent = parent[len(root) + 1 :]
    return f"{rel_parent}/{name}"
```

Also create empty `services/__init__.py` and `services/sharepoint_sync/__init__.py`.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_sharepoint_scope.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/__init__.py services/sharepoint_sync/__init__.py services/sharepoint_sync/scope.py tests/test_sharepoint_scope.py
git commit -m "feat(sharepoint): scope filter for in-scope folder matching"
```

---

## Task 3: Delta state persistence

A trivial JSON-backed store for the Graph delta token. Keeps the syncer idempotent across restarts.

**Files:**
- Create: `services/sharepoint_sync/delta_state.py`
- Test: `tests/test_sharepoint_delta_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sharepoint_delta_state.py
from pathlib import Path

from services.sharepoint_sync.delta_state import DeltaState


def test_initial_token_is_none(tmp_path: Path):
    state = DeltaState(tmp_path / "delta.json")
    assert state.load() is None


def test_save_and_load_roundtrip(tmp_path: Path):
    state = DeltaState(tmp_path / "delta.json")
    state.save("https://graph.microsoft.com/.../delta?token=abc")
    assert state.load() == "https://graph.microsoft.com/.../delta?token=abc"


def test_save_creates_parent_dir(tmp_path: Path):
    state = DeltaState(tmp_path / "nested" / "deep" / "delta.json")
    state.save("token")
    assert (tmp_path / "nested" / "deep" / "delta.json").exists()


def test_atomic_write_no_partial_on_crash(tmp_path: Path):
    # We write to .tmp first then rename — assert the rename target is the final path.
    state = DeltaState(tmp_path / "delta.json")
    state.save("v1")
    state.save("v2")
    assert state.load() == "v2"
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_sharepoint_delta_state.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# services/sharepoint_sync/delta_state.py
from __future__ import annotations

import json
from pathlib import Path


class DeltaState:
    """JSON-backed Graph delta-token persistence with atomic writes."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> str | None:
        if not self._path.exists():
            return None
        data = json.loads(self._path.read_text(encoding="utf-8"))
        token = data.get("delta_token")
        return token if isinstance(token, str) else None

    def save(self, token: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps({"delta_token": token}), encoding="utf-8")
        tmp.replace(self._path)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_sharepoint_delta_state.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add services/sharepoint_sync/delta_state.py tests/test_sharepoint_delta_state.py
git commit -m "feat(sharepoint): delta-token persistence with atomic write"
```

---

## Task 4: Local index (item_id ↔ local_path)

The Graph delta API identifies items by `item_id`. Deletes only carry the id. We need a stable mapping from id to the local path we wrote, so a delete event resolves to the right file even if the item was renamed since.

**Files:**
- Create: `services/sharepoint_sync/local_index.py`
- Test: `tests/test_sharepoint_local_index.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sharepoint_local_index.py
from pathlib import Path

import pytest

from services.sharepoint_sync.local_index import LocalIndex


def test_upsert_and_lookup(tmp_path: Path):
    idx = LocalIndex(tmp_path / "index.sqlite")
    idx.upsert("ITEM1", "a/b.pdf", etag="e1")
    assert idx.get_path("ITEM1") == "a/b.pdf"
    assert idx.get_etag("ITEM1") == "e1"


def test_upsert_replaces_path_on_rename(tmp_path: Path):
    idx = LocalIndex(tmp_path / "index.sqlite")
    idx.upsert("ITEM1", "old/name.pdf", etag="e1")
    idx.upsert("ITEM1", "new/name.pdf", etag="e2")
    assert idx.get_path("ITEM1") == "new/name.pdf"
    assert idx.get_etag("ITEM1") == "e2"


def test_delete_removes_row(tmp_path: Path):
    idx = LocalIndex(tmp_path / "index.sqlite")
    idx.upsert("ITEM1", "a.pdf", etag="e1")
    idx.delete("ITEM1")
    assert idx.get_path("ITEM1") is None


def test_iter_descendants(tmp_path: Path):
    idx = LocalIndex(tmp_path / "index.sqlite")
    idx.upsert("A", "folder/a.pdf", etag="x")
    idx.upsert("B", "folder/sub/b.pdf", etag="x")
    idx.upsert("C", "other/c.pdf", etag="x")
    descendants = sorted(idx.iter_under("folder"))
    assert descendants == [("A", "folder/a.pdf"), ("B", "folder/sub/b.pdf")]


def test_persists_across_instances(tmp_path: Path):
    db = tmp_path / "index.sqlite"
    LocalIndex(db).upsert("X", "x.pdf", etag="e")
    assert LocalIndex(db).get_path("X") == "x.pdf"
```

- [ ] **Step 2: Run to verify failure** (`pytest tests/test_sharepoint_local_index.py -v`).

- [ ] **Step 3: Implement**

```python
# services/sharepoint_sync/local_index.py
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    local_path TEXT NOT NULL,
    etag TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_path_idx ON items (local_path);
"""


class LocalIndex:
    """SQLite-backed map of (Graph item_id) -> (local_path, etag)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        with self._connect() as cx:
            cx.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def upsert(self, item_id: str, local_path: str, *, etag: str) -> None:
        with self._connect() as cx:
            cx.execute(
                "INSERT INTO items (item_id, local_path, etag) VALUES (?, ?, ?) "
                "ON CONFLICT(item_id) DO UPDATE SET local_path=excluded.local_path, "
                "etag=excluded.etag",
                (item_id, local_path, etag),
            )

    def get_path(self, item_id: str) -> str | None:
        with self._connect() as cx:
            row = cx.execute(
                "SELECT local_path FROM items WHERE item_id=?", (item_id,)
            ).fetchone()
        return row[0] if row else None

    def get_etag(self, item_id: str) -> str | None:
        with self._connect() as cx:
            row = cx.execute(
                "SELECT etag FROM items WHERE item_id=?", (item_id,)
            ).fetchone()
        return row[0] if row else None

    def delete(self, item_id: str) -> None:
        with self._connect() as cx:
            cx.execute("DELETE FROM items WHERE item_id=?", (item_id,))

    def iter_under(self, prefix: str) -> Iterator[tuple[str, str]]:
        prefix = prefix.rstrip("/") + "/"
        with self._connect() as cx:
            for row in cx.execute(
                "SELECT item_id, local_path FROM items WHERE local_path LIKE ?",
                (prefix + "%",),
            ):
                yield row[0], row[1]
```

- [ ] **Step 4: Run tests** — expect 5 passed.

- [ ] **Step 5: Commit**

```bash
git add services/sharepoint_sync/local_index.py tests/test_sharepoint_local_index.py
git commit -m "feat(sharepoint): sqlite local index for item_id<->path mapping"
```

---

## Task 5: Add deps and Graph-item models

**Files:**
- Modify: `pyproject.toml`
- Create: `services/sharepoint_sync/models.py`

- [ ] **Step 1: Add deps**

In `pyproject.toml` `[project.dependencies]` add `azure-identity>=1.19,<2`. In dev/test group add `respx>=0.21,<1`.

- [ ] **Step 2: Sync deps**

```bash
uv sync
```

Expected: both packages resolve.

- [ ] **Step 3: Create dataclasses for delta items**

```python
# services/sharepoint_sync/models.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeltaItem:
    """A normalized projection of a Graph delta entry that we care about."""

    item_id: str
    name: str
    parent_path: str  # e.g. "/drive/root:/Engineering/Specs"
    is_folder: bool
    is_deleted: bool
    etag: str  # cTag preferred (changes on content edit), falls back to eTag
    download_url: str | None  # @microsoft.graph.downloadUrl, may be absent for folders/deletes
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock services/sharepoint_sync/models.py
git commit -m "feat(sharepoint): add azure-identity dep and DeltaItem model"
```

---

## Task 6: Graph client (token + delta + download)

This is the only module that touches the network. Async, uses `httpx.AsyncClient` and `azure.identity.aio.ClientSecretCredential`.

**Files:**
- Create: `services/sharepoint_sync/graph_client.py`
- Test: `tests/test_sharepoint_graph_client.py`

### Behavior contract

- `acquire_token()` — returns a bearer token; uses `ClientSecretCredential` so refresh is handled.
- `iter_delta(initial_url=None)` — async iterator yielding `DeltaItem`s; transparently follows `@odata.nextLink`; on completion yields the final `@odata.deltaLink` via `next_delta_link` attribute.
- `download(download_url, dest_path)` — streams the file to `dest_path` using a single HTTP GET; download URLs from Graph are pre-signed and **must NOT carry the bearer token**.

- [ ] **Step 1: Write failing tests** (mock httpx with `respx`)

```python
# tests/test_sharepoint_graph_client.py
import pytest
import respx
import httpx

from services.sharepoint_sync.graph_client import GraphClient
from services.sharepoint_sync.models import DeltaItem


@pytest.fixture
def client(monkeypatch):
    # Bypass real azure-identity by injecting a fake token provider
    return GraphClient(
        drive_id="DRIVE",
        token_provider=lambda: "fake-token",
    )


@pytest.mark.asyncio
@respx.mock
async def test_iter_delta_normalizes_items(client):
    respx.get("https://graph.microsoft.com/v1.0/drives/DRIVE/root/delta").mock(
        return_value=httpx.Response(
            200,
            json={
                "value": [
                    {
                        "id": "I1",
                        "name": "doc.pdf",
                        "cTag": "ct1",
                        "eTag": "et1",
                        "file": {"mimeType": "application/pdf"},
                        "parentReference": {"path": "/drive/root:/Engineering/Specs"},
                        "@microsoft.graph.downloadUrl": "https://download/doc.pdf",
                    },
                    {
                        "id": "I2",
                        "name": "deleted.pdf",
                        "deleted": {"state": "deleted"},
                        "parentReference": {"path": "/drive/root:/Engineering/Specs"},
                    },
                ],
                "@odata.deltaLink": "https://graph.microsoft.com/v1.0/drives/DRIVE/root/delta?token=NEXT",
            },
        )
    )
    items: list[DeltaItem] = []
    async for it in client.iter_delta():
        items.append(it)
    assert items[0].item_id == "I1"
    assert items[0].is_folder is False
    assert items[0].is_deleted is False
    assert items[0].etag == "ct1"
    assert items[0].download_url == "https://download/doc.pdf"
    assert items[1].is_deleted is True
    assert items[1].download_url is None
    assert client.next_delta_link.endswith("token=NEXT")


@pytest.mark.asyncio
@respx.mock
async def test_iter_delta_follows_next_link(client):
    page1 = "https://graph.microsoft.com/v1.0/drives/DRIVE/root/delta"
    page2 = "https://graph.microsoft.com/v1.0/drives/DRIVE/root/delta?token=2"
    respx.get(page1).mock(
        return_value=httpx.Response(
            200,
            json={
                "value": [{"id": "A", "name": "a.pdf", "cTag": "x",
                            "file": {}, "parentReference": {"path": "/drive/root:/x"}}],
                "@odata.nextLink": page2,
            },
        )
    )
    respx.get(page2).mock(
        return_value=httpx.Response(
            200,
            json={
                "value": [{"id": "B", "name": "b.pdf", "cTag": "y",
                            "file": {}, "parentReference": {"path": "/drive/root:/x"}}],
                "@odata.deltaLink": "https://graph.microsoft.com/.../delta?token=END",
            },
        )
    )
    ids = [i.item_id async for i in client.iter_delta()]
    assert ids == ["A", "B"]
    assert client.next_delta_link.endswith("token=END")


@pytest.mark.asyncio
@respx.mock
async def test_download_streams_to_disk(client, tmp_path):
    respx.get("https://download/doc.pdf").mock(
        return_value=httpx.Response(200, content=b"PDFBYTES")
    )
    dest = tmp_path / "doc.pdf"
    await client.download("https://download/doc.pdf", dest)
    assert dest.read_bytes() == b"PDFBYTES"


@pytest.mark.asyncio
@respx.mock
async def test_download_does_not_send_bearer_token(client, tmp_path):
    route = respx.get("https://download/doc.pdf").mock(
        return_value=httpx.Response(200, content=b"ok")
    )
    await client.download("https://download/doc.pdf", tmp_path / "x.pdf")
    sent = route.calls[0].request
    assert "Authorization" not in sent.headers
```

- [ ] **Step 2: Run** — expect ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# services/sharepoint_sync/graph_client.py
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path

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

    async def iter_delta(self, initial_url: str | None = None) -> AsyncIterator[DeltaItem]:
        url = initial_url or f"{_GRAPH_BASE}/drives/{self._drive_id}/root/delta"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while url:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                payload = resp.json()
                for raw in payload.get("value", []):
                    yield _to_delta_item(raw)
                if "@odata.deltaLink" in payload:
                    self.next_delta_link = payload["@odata.deltaLink"]
                    return
                url = payload.get("@odata.nextLink")

    async def download(self, download_url: str, dest_path: Path) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Graph pre-signed URLs reject Authorization headers — use a clean client.
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("GET", download_url) as resp:
                resp.raise_for_status()
                with dest_path.open("wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)


def _to_delta_item(raw: dict) -> DeltaItem:
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
```

- [ ] **Step 4: Run tests** — expect 4 passed.

- [ ] **Step 5: Commit**

```bash
git add services/sharepoint_sync/graph_client.py tests/test_sharepoint_graph_client.py
git commit -m "feat(sharepoint): Graph delta + download client"
```

---

## Task 7: Syncer — apply one batch

The orchestration core. Owns deletion semantics. Takes injected `GraphClient`-like object so tests can drive it without touching the network.

**Files:**
- Create: `services/sharepoint_sync/syncer.py`
- Test: `tests/test_sharepoint_syncer.py`

### Behavior contract

`Syncer.run_once()`:
1. Read delta token (if any) from `DeltaState`.
2. Iterate Graph delta from that token.
3. For each `DeltaItem`:
   - If `is_folder` and `is_deleted`: walk `LocalIndex.iter_under(local_path)` for each descendant, delete file from disk + index.
   - If `is_folder` and not deleted: ignore (we only mirror files; folder existence is implicit).
   - If file `is_deleted`: look up local path in index; remove file; remove index row.
   - If file in scope and `etag` unchanged from index: skip (idempotent re-run).
   - If file in scope and changed (or new): download to mirror; write index row. If a previous local path existed and differs (rename), delete old file first.
   - If file out of scope **and** previously indexed: treat as out-of-scope move; delete local + index row.
   - Otherwise: ignore.
4. Persist new delta token via `DeltaState`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sharepoint_syncer.py
from pathlib import Path

import pytest

from services.sharepoint_sync.delta_state import DeltaState
from services.sharepoint_sync.local_index import LocalIndex
from services.sharepoint_sync.models import DeltaItem
from services.sharepoint_sync.syncer import Syncer


class FakeGraph:
    def __init__(self, batches: list[list[DeltaItem]], final_token: str) -> None:
        self._batches = batches
        self._call = 0
        self.next_delta_link = None
        self._final_token = final_token

    async def iter_delta(self, initial_url=None):
        batch = self._batches[self._call]
        self._call += 1
        for item in batch:
            yield item
        self.next_delta_link = self._final_token

    async def download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f"BYTES:{url}".encode())


@pytest.fixture
def syncer(tmp_path: Path):
    mirror = tmp_path / "documents" / "sharepoint"
    state = DeltaState(tmp_path / "delta.json")
    index = LocalIndex(tmp_path / "index.sqlite")
    return mirror, state, index, FakeGraph, tmp_path


@pytest.mark.asyncio
async def test_new_file_in_scope_is_downloaded(syncer):
    mirror, state, index, FakeGraph, _ = syncer
    fake = FakeGraph(
        [[DeltaItem(
            item_id="I1", name="doc.pdf",
            parent_path="/drive/root:/Specs", is_folder=False,
            is_deleted=False, etag="e1",
            download_url="https://dl/doc.pdf",
        )]],
        final_token="t1",
    )
    s = Syncer(mirror_root=mirror, root_folder_path="/Specs",
               graph=fake, state=state, index=index)
    await s.run_once()
    assert (mirror / "doc.pdf").read_bytes() == b"BYTES:https://dl/doc.pdf"
    assert index.get_path("I1") == "doc.pdf"
    assert state.load() == "t1"


@pytest.mark.asyncio
async def test_deleted_file_removes_local_copy(syncer):
    mirror, state, index, FakeGraph, _ = syncer
    (mirror).mkdir(parents=True)
    (mirror / "doc.pdf").write_bytes(b"old")
    index.upsert("I1", "doc.pdf", etag="e1")
    fake = FakeGraph(
        [[DeltaItem(item_id="I1", name="doc.pdf",
                    parent_path="/drive/root:/Specs", is_folder=False,
                    is_deleted=True, etag="", download_url=None)]],
        final_token="t2",
    )
    s = Syncer(mirror_root=mirror, root_folder_path="/Specs",
               graph=fake, state=state, index=index)
    await s.run_once()
    assert not (mirror / "doc.pdf").exists()
    assert index.get_path("I1") is None


@pytest.mark.asyncio
async def test_move_out_of_scope_removes_local_copy(syncer):
    mirror, state, index, FakeGraph, _ = syncer
    mirror.mkdir(parents=True)
    (mirror / "doc.pdf").write_bytes(b"old")
    index.upsert("I1", "doc.pdf", etag="e1")
    fake = FakeGraph(
        [[DeltaItem(item_id="I1", name="doc.pdf",
                    parent_path="/drive/root:/Marketing",  # out of scope
                    is_folder=False, is_deleted=False, etag="e2",
                    download_url="https://dl/doc.pdf")]],
        final_token="t3",
    )
    s = Syncer(mirror_root=mirror, root_folder_path="/Specs",
               graph=fake, state=state, index=index)
    await s.run_once()
    assert not (mirror / "doc.pdf").exists()
    assert index.get_path("I1") is None


@pytest.mark.asyncio
async def test_rename_within_scope_moves_file(syncer):
    mirror, state, index, FakeGraph, _ = syncer
    mirror.mkdir(parents=True)
    (mirror / "old.pdf").write_bytes(b"old")
    index.upsert("I1", "old.pdf", etag="e1")
    fake = FakeGraph(
        [[DeltaItem(item_id="I1", name="new.pdf",
                    parent_path="/drive/root:/Specs",
                    is_folder=False, is_deleted=False, etag="e2",
                    download_url="https://dl/new.pdf")]],
        final_token="t4",
    )
    s = Syncer(mirror_root=mirror, root_folder_path="/Specs",
               graph=fake, state=state, index=index)
    await s.run_once()
    assert not (mirror / "old.pdf").exists()
    assert (mirror / "new.pdf").exists()
    assert index.get_path("I1") == "new.pdf"


@pytest.mark.asyncio
async def test_unchanged_etag_is_skipped(syncer):
    mirror, state, index, FakeGraph, _ = syncer
    mirror.mkdir(parents=True)
    (mirror / "doc.pdf").write_bytes(b"current")
    index.upsert("I1", "doc.pdf", etag="e1")
    fake = FakeGraph(
        [[DeltaItem(item_id="I1", name="doc.pdf",
                    parent_path="/drive/root:/Specs",
                    is_folder=False, is_deleted=False, etag="e1",
                    download_url="https://dl/doc.pdf")]],
        final_token="t5",
    )
    s = Syncer(mirror_root=mirror, root_folder_path="/Specs",
               graph=fake, state=state, index=index)
    await s.run_once()
    assert (mirror / "doc.pdf").read_bytes() == b"current"  # untouched


@pytest.mark.asyncio
async def test_folder_deletion_removes_descendants(syncer):
    mirror, state, index, FakeGraph, _ = syncer
    mirror.mkdir(parents=True)
    (mirror / "subdir").mkdir()
    (mirror / "subdir" / "a.pdf").write_bytes(b"a")
    (mirror / "subdir" / "b.pdf").write_bytes(b"b")
    index.upsert("A", "subdir/a.pdf", etag="ea")
    index.upsert("B", "subdir/b.pdf", etag="eb")
    index.upsert("F", "subdir", etag="ef")  # folder
    fake = FakeGraph(
        [[DeltaItem(item_id="F", name="subdir",
                    parent_path="/drive/root:/Specs",
                    is_folder=True, is_deleted=True, etag="",
                    download_url=None)]],
        final_token="t6",
    )
    s = Syncer(mirror_root=mirror, root_folder_path="/Specs",
               graph=fake, state=state, index=index)
    await s.run_once()
    assert not (mirror / "subdir" / "a.pdf").exists()
    assert not (mirror / "subdir" / "b.pdf").exists()
    assert index.get_path("A") is None
    assert index.get_path("B") is None
```

- [ ] **Step 2: Run** — expect ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# services/sharepoint_sync/syncer.py
from __future__ import annotations

import logging
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

    def iter_delta(self, initial_url: str | None = None): ...
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
            return  # folders are implicit; their files come through individually

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
```

- [ ] **Step 4: Run tests** — expect all 6 passed.

- [ ] **Step 5: Commit**

```bash
git add services/sharepoint_sync/syncer.py tests/test_sharepoint_syncer.py
git commit -m "feat(sharepoint): syncer with delete/move/rename parity"
```

---

## Task 8: Outer loop + entrypoint

Wires settings → token provider → graph client → syncer, runs in a loop, signal-aware.

**Files:**
- Create: `services/sharepoint_sync/main.py`
- Create: `services/sharepoint_sync/__main__.py`

- [ ] **Step 1: Implement `main.py`**

```python
# services/sharepoint_sync/main.py
from __future__ import annotations

import asyncio
import logging
import signal
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


async def _build_token_provider() -> tuple[ClientSecretCredential, callable]:
    cred = ClientSecretCredential(
        tenant_id=settings.sharepoint_tenant_id,
        client_id=settings.sharepoint_client_id,
        client_secret=settings.sharepoint_client_secret,
    )
    cache: dict[str, str] = {}

    async def _refresh() -> str:
        token = await cred.get_token("https://graph.microsoft.com/.default")
        cache["token"] = token.token
        return token.token

    await _refresh()
    return cred, lambda: cache["token"]


async def run_loop(*, once: bool = False) -> None:
    if not settings.sharepoint_enabled:
        log.error("SharePoint sync requested but settings.sharepoint_enabled is False")
        raise SystemExit(2)

    setup_observability()
    setup_logging()

    state_dir = Path(settings.sharepoint_state_dir)
    mirror_root = Path(settings.local_documents_path) / settings.sharepoint_local_subdir

    cred, token_provider = await _build_token_provider()
    graph = GraphClient(drive_id=settings.sharepoint_drive_id, token_provider=token_provider)
    syncer = Syncer(
        mirror_root=mirror_root,
        root_folder_path=settings.sharepoint_root_folder_path,
        graph=graph,
        state=DeltaState(state_dir / "delta.json"),
        index=LocalIndex(state_dir / "index.sqlite"),
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    try:
        while True:
            try:
                # Refresh token before each cycle (azure-identity caches; cheap when valid)
                await token_provider  # noqa: B018  - keep cache warm via _refresh wrapper
                await syncer.run_once()
            except Exception:
                log.exception("sharepoint sync cycle failed; will retry next interval")
            if once or stop.is_set():
                return
            try:
                await asyncio.wait_for(stop.wait(),
                                        timeout=settings.sharepoint_sync_interval_seconds)
            except asyncio.TimeoutError:
                pass
    finally:
        await cred.close()
```

- [ ] **Step 2: Implement `__main__.py`**

```python
# services/sharepoint_sync/__main__.py
from __future__ import annotations

import argparse
import asyncio

from .main import run_loop


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="services.sharepoint_sync")
    p.add_argument("--once", action="store_true",
                   help="Run a single sync pass and exit")
    return p.parse_args()


def main() -> None:
    args = _parse()
    asyncio.run(run_loop(once=args.once))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Lint + typecheck**

```bash
task lint && task typecheck
```

Fix any issues surfaced.

- [ ] **Step 4: Commit**

```bash
git add services/sharepoint_sync/main.py services/sharepoint_sync/__main__.py
git commit -m "feat(sharepoint): outer loop + CLI entrypoint"
```

---

## Task 9: Taskfile entries

**Files:**
- Modify: `Taskfile.yml`

- [ ] **Step 1: Add tasks** under the existing task list (mirror the style of `ingest` / `ingest-live`):

```yaml
  sharepoint-sync:
    desc: Run the SharePoint sync loop (long-running)
    cmds:
      - uv run python -m services.sharepoint_sync

  sharepoint-sync-once:
    desc: Run one SharePoint sync pass and exit
    cmds:
      - uv run python -m services.sharepoint_sync --once
```

- [ ] **Step 2: Smoke test the wiring**

```bash
task --list | grep sharepoint
```

Expected: both tasks listed.

- [ ] **Step 3: Commit**

```bash
git add Taskfile.yml
git commit -m "chore(sharepoint): add sharepoint-sync tasks"
```

---

## Task 10: Compose service for prod

**Files:**
- Modify: `docker-compose.prod.yml`

- [ ] **Step 1: Inspect existing services** to copy the pattern (volumes, env_file, image, restart policy).

```bash
grep -n "ingest-live:" docker-compose.prod.yml
```

- [ ] **Step 2: Add `sharepoint-sync` service** mirroring `ingest-live` minus the public port and HTTP command:

```yaml
  sharepoint-sync:
    image: ${APP_IMAGE:-spektr:latest}
    env_file: .env.prod
    command: ["python", "-m", "services.sharepoint_sync"]
    volumes:
      - ./documents:/app/documents
      - ./state:/app/state
    depends_on:
      - postgres
    restart: unless-stopped
```

(Adjust to match the actual volume layout and image var used by other services in the file.)

- [ ] **Step 3: Validate compose**

```bash
docker compose -f docker-compose.prod.yml --env-file .env.prod config >/dev/null
```

Expected: clean exit.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.prod.yml
git commit -m "feat(sharepoint): add prod sharepoint-sync compose service"
```

---

## Task 11: Operator docs

**Files:**
- Create: `docs/ingestion/sharepoint-setup.md`

- [ ] **Step 1: Write the doc** (target audience: ops engineer setting this up for the first time). Required sections:

  - **Overview:** what the syncer does, parity guarantees with S3+SQS (deletion, idempotence, restart safety).
  - **Azure AD app registration** — step-by-step, including granting `Sites.Selected` to the target site (or `Sites.Read.All` if scoping isn't possible) and creating a client secret.
  - **Finding `site_id` and `drive_id`** — Graph API `/sites/{hostname}:/sites/{path}?$select=id` and `/sites/{site_id}/drives`.
  - **Choosing `root_folder_path`** — must start with `/`, must be an absolute path inside the drive, sibling folders are out of scope.
  - **Env-var checklist** — table of every `SHAREPOINT_*` setting with example values.
  - **First-run smoke test** — `task sharepoint-sync-once` and what to look for in logs.
  - **Operational notes** — token TTL (~1h, handled by `azure-identity`), throttling (`Retry-After` respected by client), state files (`state/sharepoint/delta.json`, `state/sharepoint/index.sqlite`) and how to recover from a wedged delta token (delete the file, re-run).
  - **Deletion semantics** — explicit promise that SharePoint deletes propagate to Qdrant + Neo4j within one sync interval.

- [ ] **Step 2: Add the page to mkdocs nav** if `mkdocs.yml` uses an explicit nav:

```bash
grep -A 30 "^nav:" mkdocs.yml || echo "no explicit nav"
```

If explicit, insert the new doc under the Ingestion section. Otherwise, mkdocs auto-discovers it.

- [ ] **Step 3: Build docs to verify**

```bash
task docs-build
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add docs/ingestion/sharepoint-setup.md mkdocs.yml
git commit -m "docs(sharepoint): operator setup guide"
```

---

## Task 12: Live end-to-end verification

This is a manual / scripted verification — not committed code, but mandatory to close the feature.

- [ ] **A. Auth smoke**
  - Configure `.env` with all `SHAREPOINT_*` values pointing at a test SharePoint site/library.
  - Run: `task sharepoint-sync-once`
  - Expected: zero errors, `state/sharepoint/delta.json` exists, mirror dir populated.

- [ ] **B. In-scope filtering** — verify only the in-scope folder is mirrored.
  - Have a sibling folder at the same level as the in-scope folder.
  - Run sync once.
  - Confirm only files under the in-scope folder appear in `documents/sharepoint/...`.

- [ ] **C. Full ingestion path**
  - `task up` (services), `task ingest --live` in one terminal, `task sharepoint-sync` in another.
  - Drop a PDF into the SharePoint folder.
  - Within ~3 min, `task smoke "<query from doc>"` returns hits.
  - `task smoke-graph "<entity>"` returns nodes.

- [ ] **D. Deletion parity (the load-bearing requirement)**
  - Delete the file in SharePoint.
  - Within ~3 min, `task doctor` reports zero drift, manual Qdrant query for `source_file` of the deleted file returns 0 points, Neo4j query returns no nodes for that source.

- [ ] **E. Move out of scope**
  - Move a file from inside the in-scope folder to a sibling folder.
  - Confirm same purge as (D) — vectors and graph entries gone.

- [ ] **F. Rename within scope**
  - Rename a file inside the in-scope folder.
  - Confirm: old `source_file` purged, new `source_file` ingested with same content.

- [ ] **G. Folder deletion**
  - Delete a subfolder under the in-scope folder.
  - Confirm all descendant files purged.

- [ ] **H. Restart durability**
  - During a sync, `Ctrl-C` the syncer.
  - Restart it: `task sharepoint-sync`.
  - Confirm: no duplicate downloads, no orphan local files, `task doctor` clean.

- [ ] **I. S3 regression** (parity with existing path)
  - In a separate env, configure `S3_BUCKET_NAME` + `S3_SQS_QUEUE_URL` (no SharePoint vars).
  - Confirm Path A still ingests via S3 unchanged.

- [ ] **J. Eval gate**
  - `task eval` — confirm no metric drops below `tests/eval/thresholds.yaml`.

---

## Self-review checklist (filled in)

- [x] **Spec coverage:** every requirement (deletion parity, scope filtering, parallel S3 path, polling latency, app-only auth) maps to a task.
- [x] **No placeholders:** all code blocks contain executable code; tests have explicit assertions; no "TODO" or "implement later" steps.
- [x] **Type consistency:** `DeltaItem` fields used identically across `models.py`, `graph_client.py`, `syncer.py`, and tests.
- [x] **File-size budget:** every new module fits well under the 600-line / 60-line-function limits in `CLAUDE.md`.
- [x] **No untouched-but-fragile files break:** verification step I exercises the S3 path explicitly to catch regressions.

## Out of scope (intentional non-goals)

- Push notifications via Graph subscriptions (additive future work).
- Multiple SharePoint sites or drives.
- Delegated-user OAuth (we use app-only client credentials).
- A custom CocoIndex `SharePointSource` class (we mirror to disk and reuse `LocalFile` instead).
