"""VLM-based visual answer generation for MCP server.

Fetches page images from S3 and sends them to a vision-capable
LLM to generate a natural language answer. Enabled via
settings.vlm_generation_enabled.
"""

from __future__ import annotations

import asyncio
import base64
import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config.settings import settings

logger = logging.getLogger(__name__)


def _fetch_s3_image(s3_key: str) -> tuple[bytes, str]:
    """Download an image from S3 and return (bytes, media_type)."""
    kwargs: dict[str, str] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
    s3 = boto3.client("s3", **kwargs)
    resp = s3.get_object(Bucket=settings.s3_bucket_name, Key=s3_key)
    body = resp["Body"].read()
    content_type = resp.get("ContentType", "image/png")
    return body, content_type


async def _ask_vlm(query: str, images: list[tuple[bytes, str]]) -> str:
    """Send images + query to a vision-capable LLM."""
    provider = settings.llm_api_type.lower()

    content: list[dict] = []  # type: ignore[type-arg]
    for img_bytes, media_type in images:
        b64 = base64.b64encode(img_bytes).decode()
        if provider == "anthropic":
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                }
            )
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": (f"data:{media_type};base64,{b64}")},
                }
            )

    content.append({"type": "text", "text": query})

    if provider == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
        )
        resp = await client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text
    else:
        import openai

        client = openai.AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
        )
        resp = await client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return resp.choices[0].message.content or ""


async def generate_visual_answer(
    query: str,
    results: list[dict],  # type: ignore[type-arg]
) -> str | None:
    """Generate a VLM answer from visual search results.

    Fetches page images from S3 for the top results, sends
    them with the query to a vision-capable LLM, and returns
    the generated answer.

    Args:
        query: The user's search query.
        results: Visual search results (must have 'source_key').

    Returns:
        Generated answer string, or None on failure.
    """
    s3_keys = [r["source_key"] for r in results if r.get("source_key")]
    if not s3_keys:
        return None

    # Limit to top 3 images to control cost
    s3_keys = s3_keys[:3]

    try:
        images: list[tuple[bytes, str]] = []
        for key in s3_keys:
            img_bytes, media_type = await asyncio.to_thread(
                _fetch_s3_image,
                key,
            )
            images.append((img_bytes, media_type))

        return await _ask_vlm(query, images)
    except (BotoCoreError, ClientError, OSError, ValueError, KeyError) as exc:
        logger.exception("VLM generation failed: %s", exc)
        return None
