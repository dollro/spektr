"""Embedding model registry: capabilities and the routes that serve them.

Capabilities belong to the **model**, not to the vendor endpoint that
happens to serve it. `voyage-4-large` is reachable both natively and
through OpenRouter and is the same model either way — the old
`EMBEDDING_PROVIDER` setting conflated those two axes, which is why it
could not express "gemini-2, via OpenRouter" and "voyage-4, via
OpenRouter" as the distinct choices they are.

Capabilities are expressed as the routes on which they are **implemented
in this codebase**, not as the routes on which they are theoretically
possible. Advertising a capability we do not implement is what caused
`IMAGE_EMBED_STRATEGY=smart` to silently drop pages: `image_routes` must
list only routes whose client actually has a working `embed_image`.

Two entries currently exercise that rule in opposite directions.
`gemini-2` lists `openrouter` because `OpenRouterEmbedder.embed_image` is
implemented and verified against the live gateway. `qwen3-embedding-8b`
leaves `image_routes` empty because the model is text-only, even though
the same `/v1/embeddings` shape would happily accept an image field from
a server that supported one.

(This docstring previously said the opposite of the first case — that
gemini-2's `image_routes` was empty because the client raised. That was
true when written and became stale when the client gained image support,
which is precisely the drift this file exists to prevent, so it is called
out rather than quietly deleted.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EmbeddingModel = Literal["jina-v4", "voyage-4", "gemini-2", "qwen3-embedding-8b"]
# `local` is not a vendor: it is "an OpenAI-compatible /v1/embeddings served
# by a container you run" (TEI, vLLM, Infinity). It is a route rather than a
# model because the same open checkpoint can also appear behind a gateway.
EmbeddingRoute = Literal["native", "openrouter", "local"]


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
    # The HuggingFace id the TEI/vLLM/Infinity container serves on the
    # `local` route, and which the request's `model` field carries. Empty
    # when the model has no self-hosted route.
    local_model_id: str = ""
    # Open encoders (e5, bge, Qwen3) take the retrieval role as an
    # INSTRUCTION PREFIX in the text, not as an `input_type` field on the
    # request. Same endpoint, different way of saying "this is a query" —
    # which is exactly why OpenRouterEmbedder's docstring excludes that
    # model family (ingestion/embedders/openrouter.py:76-80) rather than
    # claiming to serve every model the gateway lists.
    #
    # An empty prefix is a real answer, not an unset one: Qwen3 prefixes
    # queries only, and prefixing passages as well would move the passage
    # side of the vector space for no retrieval gain.
    query_prefix: str = ""
    passage_prefix: str = ""


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
    # The first self-hostable dense entry, and the one that makes a
    # sovereign stack expressible at all: Apache-2.0 weights behind a
    # container you run, so unlike jina-v4 there is no licensing bar to
    # hosting it.
    #
    # TEXT ONLY, and `image_routes` stays empty accordingly. This is the
    # case the module docstring above is about: Qwen3-Embedding has no
    # vision tower, so there is nothing to advertise even optimistically.
    # Selecting it therefore forces IMAGE_EMBED_STRATEGY=none — the
    # Settings validator already enforces that, so the consequence lands
    # at startup as a decision rather than at ingest as dropped pages.
    "qwen3-embedding-8b": ModelSpec(
        # Verified against the Qwen/Qwen3-Embedding-8B model card and the
        # QwenLM/Qwen3-Embedding README (checked 2026-08-16): 4096-wide
        # output, MRL-trained, custom sizes valid from 32 up to 4096.
        default_dimensions=4096,
        # 1024 keeps the collection the same order of size as the hosted
        # models' operating points, which is what MRL is for. Not a
        # measured quality/size trade-off for this model specifically —
        # re-derive it from the retrieval eval before treating it as one.
        recommended_dimensions=1024,
        routes=("local",),
        default_route="local",
        # Matryoshka. The card names 32 as the practical floor, but this
        # field is an exact-value set rather than a range and so cannot
        # express it; below 32 nothing here objects.
        allowed_dimensions=(),
        image_routes=(),
        local_model_id="Qwen/Qwen3-Embedding-8B",
        # Verbatim from the model card's `get_detailed_instruct`:
        #     f"Instruct: {task_description}\nQuery:{query}"
        # The missing space after "Query:" is the template's, not a typo —
        # the prefix is part of what the model was trained on, so it is
        # copied exactly rather than tidied. The task description is the
        # card's own retrieval default; changing it changes the query-side
        # vector space, so treat it like the `input_type` mapping next
        # door and do not edit it on a live index.
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery:"
        ),
        # "No need to add instruction for retrieval documents" — model card.
        passage_prefix="",
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
