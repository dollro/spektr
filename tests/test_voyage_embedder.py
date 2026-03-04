from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.embedders.voyage import VoyageEmbedder

DIMS = 1024


@pytest.fixture
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.voyage_api_key = "test-voyage-key"
    s.voyage_api_url = "https://api.voyageai.com"
    s.voyage_text_model = "voyage-4-large"
    s.voyage_multimodal_model = "voyage-multimodal-3.5"
    s.voyage_dense_dimensions = DIMS
    s.voyage_rpm = 300
    s.voyage_max_concurrent = 10
    s.max_retries = 3
    return s


@pytest.fixture
async def embedder(mock_settings: MagicMock) -> VoyageEmbedder:
    with patch("ingestion.embedders.voyage.settings", mock_settings):
        e = VoyageEmbedder(api_key="test-voyage-key")
        e._ensure_loop_resources()
        return e


def _mock_response(
    data: list[dict],  # type: ignore[type-arg]
) -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"data": data}
    return resp


class TestEmbedText:
    async def test_correct_payload(self, embedder: VoyageEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}, {"embedding": [0.2] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text(["hello", "world"])

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "voyage-4-large"
        assert payload["input_type"] == "document"
        assert payload["input"] == ["hello", "world"]
        assert payload["output_dimensions"] == DIMS
        assert len(result) == 2
        assert len(result[0]) == DIMS

    async def test_url_is_text_endpoint(self, embedder: VoyageEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["test"])

        assert mock_post.call_args.args[0] == embedder._text_url


class TestEmbedTextQuery:
    async def test_query_task_maps_correctly(self, embedder: VoyageEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text_query("search query")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["input_type"] == "query"
        assert len(result) == DIMS


class TestEmbedImage:
    async def test_uses_multimodal_endpoint_and_model(self, embedder: VoyageEmbedder) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\nfakedata"
        expected_b64 = base64.b64encode(image_bytes).decode()
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_image(image_bytes)

        url = mock_post.call_args.args[0]
        assert url == embedder._multimodal_url
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "voyage-multimodal-3.5"
        assert payload["input"] == [
            [
                {
                    "type": "image_base64",
                    "image_base64": expected_b64,
                    "media_type": "image/png",
                }
            ]
        ]
        assert payload["input_type"] == "document"
        assert payload["output_dimensions"] == DIMS
        assert len(result) == DIMS

    async def test_custom_timeout(self, embedder: VoyageEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_image(b"fake")

        assert mock_post.call_args.kwargs["timeout"] == 120.0


class TestMultiVectorNotSupported:
    async def test_embed_multi_vector_raises(self, embedder: VoyageEmbedder) -> None:
        with pytest.raises(NotImplementedError, match="ColBERT"):
            await embedder.embed_multi_vector(b"fake")

    async def test_embed_query_multi_vector_raises(self, embedder: VoyageEmbedder) -> None:
        with pytest.raises(NotImplementedError, match="ColBERT"):
            await embedder.embed_query_multi_vector("query")
