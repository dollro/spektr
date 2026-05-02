from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from services.sharepoint_sync.graph_client import GraphClient
from services.sharepoint_sync.models import DeltaItem


@pytest.fixture
def client() -> GraphClient:
    return GraphClient(drive_id="DRIVE", token_provider=lambda: "fake-token")


@respx.mock
async def test_iter_delta_normalizes_items(client: GraphClient) -> None:
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
                "@odata.deltaLink": (
                    "https://graph.microsoft.com/v1.0/drives/DRIVE/root/delta?token=NEXT"
                ),
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
    assert client.next_delta_link is not None
    assert client.next_delta_link.endswith("token=NEXT")


@respx.mock
async def test_iter_delta_follows_next_link(client: GraphClient) -> None:
    page1 = "https://graph.microsoft.com/v1.0/drives/DRIVE/root/delta"
    page2 = "https://graph.microsoft.com/v1.0/drives/DRIVE/root/delta-page2"
    respx.get(url__eq=page1).mock(
        return_value=httpx.Response(
            200,
            json={
                "value": [
                    {
                        "id": "A",
                        "name": "a.pdf",
                        "cTag": "x",
                        "file": {},
                        "parentReference": {"path": "/drive/root:/x"},
                    }
                ],
                "@odata.nextLink": page2,
            },
        )
    )
    respx.get(page2).mock(
        return_value=httpx.Response(
            200,
            json={
                "value": [
                    {
                        "id": "B",
                        "name": "b.pdf",
                        "cTag": "y",
                        "file": {},
                        "parentReference": {"path": "/drive/root:/x"},
                    }
                ],
                "@odata.deltaLink": "https://graph.microsoft.com/.../delta?token=END",
            },
        )
    )
    ids = [i.item_id async for i in client.iter_delta()]
    assert ids == ["A", "B"]
    assert client.next_delta_link is not None
    assert client.next_delta_link.endswith("token=END")


@respx.mock
async def test_download_streams_to_disk(client: GraphClient, tmp_path: Path) -> None:
    respx.get("https://download/doc.pdf").mock(
        return_value=httpx.Response(200, content=b"PDFBYTES")
    )
    dest = tmp_path / "doc.pdf"
    await client.download("https://download/doc.pdf", dest)
    assert dest.read_bytes() == b"PDFBYTES"


@respx.mock
async def test_download_does_not_send_bearer_token(
    client: GraphClient, tmp_path: Path
) -> None:
    route = respx.get("https://download/doc.pdf").mock(
        return_value=httpx.Response(200, content=b"ok")
    )
    await client.download("https://download/doc.pdf", tmp_path / "x.pdf")
    sent = route.calls[0].request
    assert "Authorization" not in sent.headers


@respx.mock
async def test_iter_delta_uses_initial_url_when_provided(client: GraphClient) -> None:
    custom = "https://graph.microsoft.com/v1.0/drives/DRIVE/root/delta?token=resume"
    route = respx.get(custom).mock(
        return_value=httpx.Response(
            200,
            json={
                "value": [],
                "@odata.deltaLink": "https://graph.microsoft.com/.../delta?token=DONE",
            },
        )
    )
    async for _ in client.iter_delta(initial_url=custom):
        pass
    assert route.called
    assert client.next_delta_link is not None
    assert client.next_delta_link.endswith("token=DONE")
