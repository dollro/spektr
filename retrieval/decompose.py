"""Query decomposition — split multi-part questions into sub-queries.

Each sub-query becomes an additional retrieval channel, so decomposition
composes naturally with RRF: N sub-queries produce 2N channels, all fused in
one pass. Every failure path returns [query] unchanged, so a decomposition
outage degrades to single-query retrieval rather than an error.
"""

from __future__ import annotations

import logging

from config.settings import settings

logger = logging.getLogger(__name__)

PROMPT = """Split this search query into independent sub-queries, one per line.

Rules:
- Output ONLY the sub-queries, one per line, no preamble.
- If the query asks exactly one thing, output it unchanged as a single line.
- Maximum {max_n} lines.
- Each sub-query must be self-contained and searchable on its own.

Query: {query}"""


def _parse_subqueries(raw: str, max_n: int) -> list[str]:
    """Extract sub-queries from the model's line-per-query output."""
    out: list[str] = []
    for line in raw.strip().splitlines():
        text = line.strip()
        if not text:
            continue
        # Strip a leading "1." / "1)" / "- " enumerator if present.
        for sep in (". ", ") ", "- "):
            head, found, tail = text.partition(sep)
            if found and head.lstrip("-").strip().isdigit() or (sep == "- " and not head):
                text = tail.strip()
                break
        if text:
            out.append(text)
        if len(out) >= max_n:
            break
    return out


def _model_name() -> str:
    """Decomposition model, falling back to the primary LLM when unset."""
    return settings.decompose_model or settings.llm_model


async def _call_llm(prompt: str) -> str:
    """Send the decomposition prompt to the configured provider.

    Follows the same anthropic/openai dispatch as ingestion/entity_extractor.py
    — this deployment may route Anthropic models through an OpenAI-compatible
    gateway, so the provider is a config choice, not a model-name choice.
    """
    if settings.llm_api_type.lower() == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
        )
        response = await client.messages.create(
            model=_model_name(),
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    import openai

    client_oa = openai.AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
    )
    completion = await client_oa.chat.completions.create(
        model=_model_name(),
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""


async def decompose(query: str) -> list[str]:
    """Split a query into sub-queries, or return it unchanged.

    Args:
        query: The user's original query.

    Returns:
        One or more sub-queries. Never empty; falls back to [query].
    """
    if not settings.decompose_enabled or not query.strip():
        return [query]

    prompt = PROMPT.format(query=query, max_n=settings.decompose_max_subqueries)
    try:
        raw = await _call_llm(prompt)
    except Exception:
        logger.exception("Decomposition failed, using the original query")
        return [query]

    parts = _parse_subqueries(raw, settings.decompose_max_subqueries)
    if len(parts) <= 1:
        return [query]
    return parts
