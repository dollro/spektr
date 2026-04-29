from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.embedders.openrouter import OpenRouterEmbedder

DIMS = 3072


@pytest.fixture
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.openrouter_api_key = "test-or-key"
    s.openrouter_api_url = "https://openrouter.ai/api"
    s.openrouter_model = "google/gemini-embedding-2-preview"
    s.openrouter_dense_dimensions = DIMS
    s.openrouter_rpm = 300
    s.openrouter_max_concurrent = 10
    s.openrouter_http_referer = "https://example.test"
    s.openrouter_x_title = "Spektr-Test"
    s.max_retries = 3
    return s


@pytest.fixture
async def embedder(mock_settings: MagicMock) -> OpenRouterEmbedder:
    with patch("ingestion.embedders.openrouter.settings", mock_settings):
        e = OpenRouterEmbedder(api_key="test-or-key")
        e._ensure_loop_resources()
        return e


def _mock_response(data: list[dict]) -> httpx.Response:  # type: ignore[type-arg]
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"data": data}
    return resp


class TestEmbedText:
    async def test_correct_payload(self, embedder: OpenRouterEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}, {"embedding": [0.2] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text(["hello", "world"])

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "google/gemini-embedding-2-preview"
        assert payload["input"] == ["hello", "world"]
        assert payload["encoding_format"] == "float"
        assert payload["dimensions"] == DIMS
        assert len(result) == 2
        assert len(result[0]) == DIMS

    async def test_url_is_embeddings_endpoint(self, embedder: OpenRouterEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * DIMS}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["test"])

        assert mock_post.call_args.args[0] == embedder._url
        assert "openrouter.ai/api/v1/embeddings" in embedder._url

    async def test_dimensions_override(self, embedder: OpenRouterEmbedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * 768}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["hi"], dimensions=768)

        assert mock_post.call_args.kwargs["json"]["dimensions"] == 768


class TestRankingHeaders:
    async def test_optional_headers_attached(self, embedder: OpenRouterEmbedder) -> None:
        headers = embedder._client.headers  # type: ignore[union-attr]
        assert headers["Authorization"] == "Bearer test-or-key"
        assert headers["HTTP-Referer"] == "https://example.test"
        assert headers["X-Title"] == "Spektr-Test"


class TestUnsupported:
    async def test_embed_image_raises(self, embedder: OpenRouterEmbedder) -> None:
        with pytest.raises(NotImplementedError, match="text-only"):
            await embedder.embed_image(b"\x89PNG")

    async def test_embed_multi_vector_raises(self, embedder: OpenRouterEmbedder) -> None:
        with pytest.raises(NotImplementedError):
            await embedder.embed_multi_vector(b"\x89PNG")

    async def test_embed_query_multi_vector_raises(self, embedder: OpenRouterEmbedder) -> None:
        with pytest.raises(NotImplementedError):
            await embedder.embed_query_multi_vector("q")


class TestProtocolCompliance:
    def test_implements_embedder_protocol(self) -> None:
        from ingestion.embedder import Embedder

        assert isinstance(OpenRouterEmbedder.__new__(OpenRouterEmbedder), Embedder)
