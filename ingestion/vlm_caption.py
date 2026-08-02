"""VLM captioning of visual pages, feeding the knowledge graph.

Split out of ``pipeline.py`` during the CocoIndex v1 migration: captioning is a
self-contained concern (LLM client + prompt + graph hand-off) and keeping it
inline pushed the pipeline module past the 600-line cap.

Behaviour is unchanged from v0 — captions go to the graph engine only; they are
never written to Qdrant.
"""

from __future__ import annotations

import base64

from config.logging import get_logger
from config.settings import settings
from ingestion.file_processor import TextChunk
from ingestion.graph_engine import GraphEngine

logger = get_logger(__name__)

_CAPTION_PROMPT = (
    "Describe the content of this document page in detail. "
    "Extract all entities (people, organizations, products, dates, "
    "numbers), relationships, and key facts. Be factual and concise."
)

_vlm_client_anthropic: object | None = None
_vlm_client_openai: object | None = None


def _get_vlm_client() -> object:
    """Return a lazily-initialized VLM API client (singleton)."""
    provider = settings.llm_api_type.lower()
    if provider == "anthropic":
        global _vlm_client_anthropic  # noqa: PLW0603
        if _vlm_client_anthropic is None:
            import anthropic

            _vlm_client_anthropic = anthropic.AsyncAnthropic(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
            )
        return _vlm_client_anthropic
    else:
        global _vlm_client_openai  # noqa: PLW0603
        if _vlm_client_openai is None:
            import openai

            _vlm_client_openai = openai.AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
            )
        return _vlm_client_openai


async def caption_visual_page(image_bytes: bytes) -> str:
    """Generate a text description of a visual page using a VLM."""
    provider = settings.llm_api_type.lower()
    b64_str = base64.b64encode(image_bytes).decode()
    client = _get_vlm_client()

    if provider == "anthropic":
        resp = await client.messages.create(  # type: ignore[attr-defined]
            model=settings.llm_model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_str,
                            },
                        },
                        {"type": "text", "text": _CAPTION_PROMPT},
                    ],
                }
            ],
        )
        return str(resp.content[0].text)
    resp = await client.chat.completions.create(  # type: ignore[attr-defined]
        model=settings.llm_model,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_str}"},
                    },
                    {"type": "text", "text": _CAPTION_PROMPT},
                ],
            }
        ],
    )
    return str(resp.choices[0].message.content or "")


async def caption_and_ingest_visual(
    source_file: str,
    image_bytes: bytes,
    page_number: int,
    graph_engine: GraphEngine,
) -> None:
    """Caption a visual page and ingest the resulting text into the graph."""
    try:
        caption = await caption_visual_page(image_bytes)
        if not caption or not caption.strip():
            return
        chunk = TextChunk(text=caption, chunk_index=0, page_number=page_number)
        await graph_engine.ingest([chunk], source_file)
        logger.info(
            "VLM caption ingested for %s page %d (%d chars)",
            source_file,
            page_number,
            len(caption),
        )
    except Exception:
        logger.exception(
            "VLM caption/ingestion failed for %s page %d",
            source_file,
            page_number,
        )
