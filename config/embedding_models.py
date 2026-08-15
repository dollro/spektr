"""Embedding model registry: capabilities and the routes that serve them.

Capabilities belong to the **model**, not to the vendor endpoint that
happens to serve it. `voyage-4-large` is reachable both natively and
through OpenRouter and is the same model either way — the old
`EMBEDDING_PROVIDER` setting conflated those two axes, which is why it
could not express "gemini-2, via OpenRouter" and "voyage-4, via
OpenRouter" as the distinct choices they are.

Capabilities are expressed as the routes on which they are **implemented
in this codebase**, not as the routes on which they are theoretically
possible. Gemini Embedding 2 accepts images at the gateway, but
`OpenRouterEmbedder.embed_image` raises, so `image_routes` is empty for
it. Advertising a capability we do not implement is what caused
`IMAGE_EMBED_STRATEGY=smart` to silently drop pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EmbeddingModel = Literal["jina-v4", "voyage-4", "gemini-2"]
EmbeddingRoute = Literal["native", "openrouter"]


@dataclass(frozen=True)
class ModelSpec:
    """What a model can do, and where it can be reached."""

    # The model's native/full output size. Doubles as the ceiling for
    # Matryoshka truncation.
    default_dimensions: int
    # What we actually use when EMBEDDING_DIMENSIONS is unset. Separate from
    # default_dimensions because the best operating point is often not the
    # full size — Google recommends 768 of gemini-2's 3072, at a quarter of
    # the storage for near-identical quality.
    recommended_dimensions: int
    routes: tuple[EmbeddingRoute, ...]
    # Used when EMBEDDING_ROUTE is unset, so switching EMBEDDING_MODEL alone
    # is enough. Must be a member of `routes`.
    default_route: EmbeddingRoute
    # Which output sizes the model actually emits. Empty means Matryoshka:
    # any size from 1 up to default_dimensions is valid. Non-empty means the
    # model only supports these exact values. This is the constraint that
    # makes a single EMBEDDING_DIMENSIONS knob safe — without it, a value
    # tuned for one model silently carries over to another that rejects it.
    allowed_dimensions: tuple[int, ...] = ()
    image_routes: tuple[EmbeddingRoute, ...] = ()
    multivector_routes: tuple[EmbeddingRoute, ...] = ()
    late_chunking_routes: tuple[EmbeddingRoute, ...] = ()
    # OpenRouter model ids, empty when the model is not served there.
    openrouter_text_id: str = ""
    openrouter_image_id: str = ""


REGISTRY: dict[str, ModelSpec] = {
    # Text + image + ColBERT in one model, but Qwen-Research-licensed and
    # served only from Jina's own research-tier infrastructure — no
    # inference provider hosts it. See docs/ingestion/embeddings.md.
    "jina-v4": ModelSpec(
        default_dimensions=2048,
        # 512 is this project's documented operating point: ~4x storage
        # saving at ~95% quality retention via Matryoshka truncation.
        recommended_dimensions=512,
        routes=("native",),
        default_route="native",
        image_routes=("native",),
        multivector_routes=("native",),
        late_chunking_routes=("native",),
    ),
    # Native route uses two endpoints (text + multimodal); the OpenRouter
    # route exposes both model ids but only text is implemented here.
    "voyage-4": ModelSpec(
        default_dimensions=1024,
        # Voyage's own default, and the only route where images work.
        recommended_dimensions=1024,
        default_route="native",
        # Not Matryoshka: voyage emits these four sizes and nothing between.
        allowed_dimensions=(256, 512, 1024, 2048),
        routes=("native", "openrouter"),
        image_routes=("native",),
        openrouter_text_id="voyageai/voyage-4-large",
        openrouter_image_id="voyageai/voyage-multimodal-3.5",
    ),
    # No native route on purpose: Google direct returns 429s citing Vertex
    # quota on AI Studio keys, while OpenRouter fronts three endpoints for
    # this model and routes around a degraded one.
    #
    # Natively multimodal: the same model id embeds text and images into one
    # space, so there is no separate openrouter_image_id. Verified live —
    # `dimensions` is honoured for image input, so image vectors share the
    # text collection's size and cross-modal retrieval needs no second index.
    "gemini-2": ModelSpec(
        default_dimensions=3072,
        # Google's recommended quality/storage sweet spot.
        recommended_dimensions=768,
        default_route="openrouter",
        routes=("openrouter",),
        image_routes=("openrouter",),
        openrouter_text_id="google/gemini-embedding-2",
    ),
}


def spec(model: str) -> ModelSpec:
    """Look up a model, failing loudly on an unknown name."""
    try:
        return REGISTRY[model]
    except KeyError:
        raise ValueError(
            f"Unknown embedding model {model!r}; expected one of {sorted(REGISTRY)}"
        ) from None
