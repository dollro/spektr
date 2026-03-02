from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ingestion.embedder import JinaV4Embedder

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def embedder() -> JinaV4Embedder:
    return JinaV4Embedder(api_key="test-key")


def _mock_response(data: list[dict]) -> httpx.Response:  # type: ignore[type-arg]
    """Create a mock httpx.Response with embedding data."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"data": data}
    return resp


class TestEmbedText:
    async def test_correct_payload(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * 2048}, {"embedding": [0.2] * 2048}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text(["hello", "world"])

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "jina-clip-v4"
        assert payload["task"] == "retrieval.passage"
        assert payload["dimensions"] == 2048
        assert payload["normalized"] is True
        assert payload["input"] == [{"text": "hello"}, {"text": "world"}]
        assert len(result) == 2
        assert len(result[0]) == 2048

    async def test_url_is_correct(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * 2048}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_text(["test"])

        assert mock_post.call_args.args[0] == JinaV4Embedder.JINA_API_URL


class TestEmbedTextQuery:
    async def test_uses_retrieval_query_task(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * 2048}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_text_query("search query")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["task"] == "retrieval.query"
        assert len(result) == 2048


class TestEmbedImage:
    async def test_correct_payload(self, embedder: JinaV4Embedder) -> None:
        image_bytes = b"\x89PNG\r\n\x1a\nfakedata"
        expected_b64 = base64.b64encode(image_bytes).decode()
        mock_resp = _mock_response([{"embedding": [0.1] * 2048}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_image(image_bytes)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["task"] == "retrieval.passage"
        expected_uri = f"data:image/png;base64,{expected_b64}"
        assert payload["input"] == [{"image": expected_uri}]
        assert len(result) == 2048

    async def test_custom_timeout(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [0.1] * 2048}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await embedder.embed_image(b"fake")

        assert mock_post.call_args.kwargs["timeout"] == 120.0


class TestEmbedMultiVector:
    async def test_correct_payload_with_colbert(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [[0.1] * 128, [0.2] * 128]}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_multi_vector(b"fake")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["embedding_type_params"] == {"output_type": "colbert"}
        assert payload["dimensions"] == 128
        assert len(result) == 2
        assert len(result[0]) == 128


class TestEmbedQueryMultiVector:
    async def test_uses_retrieval_query_and_colbert(self, embedder: JinaV4Embedder) -> None:
        mock_resp = _mock_response([{"embedding": [[0.1] * 128, [0.2] * 128]}])
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await embedder.embed_query_multi_vector("query")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["task"] == "retrieval.query"
        assert payload["embedding_type_params"] == {"output_type": "colbert"}
        assert payload["input"] == [{"text": "query"}]
        assert len(result) == 2


class TestErrorHandling:
    async def test_http_error_raises(self, embedder: JinaV4Embedder) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_resp.request = MagicMock()
        with patch.object(
            embedder._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await embedder.embed_text(["fail"])


@pytest.mark.integration
class TestEmbedderIntegration:
    @pytest.fixture
    async def live_embedder(self) -> JinaV4Embedder:
        embedder = JinaV4Embedder()
        yield embedder  # type: ignore[misc]
        await embedder.close()

    async def test_embed_text_returns_2048d(self, live_embedder: JinaV4Embedder) -> None:
        result = await live_embedder.embed_text(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 2048
        assert all(isinstance(v, float) for v in result[0])

    async def test_embed_text_query_returns_2048d(self, live_embedder: JinaV4Embedder) -> None:
        result = await live_embedder.embed_text_query("test query")
        assert len(result) == 2048

    async def test_embed_image_returns_2048d(self, live_embedder: JinaV4Embedder) -> None:
        png = (FIXTURES / "sample.png").read_bytes()
        result = await live_embedder.embed_image(png)
        assert len(result) == 2048

    async def test_embed_multi_vector_returns_128d(
        self, live_embedder: JinaV4Embedder
    ) -> None:
        png = (FIXTURES / "sample.png").read_bytes()
        result = await live_embedder.embed_multi_vector(png)
        assert len(result) > 0
        assert all(len(v) == 128 for v in result)

    async def test_embed_query_multi_vector_structure(
        self, live_embedder: JinaV4Embedder
    ) -> None:
        result = await live_embedder.embed_query_multi_vector("test")
        assert len(result) > 0
        assert all(len(v) == 128 for v in result)
