"""Reranker model registry: wire protocols and the score scales they emit.

The sibling of `config/embedding_models.py`, and for the same reason:
capabilities belong to the **model**, not to the vendor endpoint that
happens to serve it. A `RERANK_BASE_URL` string would be enough if rerank
APIs had converged the way `/v1/embeddings` did — they have not. Jina,
Cohere, Voyage and TEI disagree on the path, the request field names *and*
the response envelope, so the model string has to carry which protocol to
speak.

The field that earns this module its existence is `score_scale`.
`RERANK_SCORE_FLOOR` gates the relevance retry: `retrieval/gate.py`
compares the top-1 reranked score against the floor and widens the pool
when it falls short. `jina-reranker-v3.5` emits **unbounded logits**, so a
floor of `0.0` is a meaningful "worse than neutral" threshold. A TEI or
Cohere cross-encoder emits **sigmoid scores in [0, 1]**, where the same
`0.0` floor can never be crossed — the gate stops firing, no exception is
raised, and no log line records that a retrieval stage went dark. Scores
are not comparable across rerankers, so the floor cannot be a bare number
in `.env` with no idea which scale it was tuned on.

This is the same hazard `config/embedding_models.py:44-48` documents for
`EMBEDDING_DIMENSIONS` — a value tuned for one model silently carrying
over to another it does not fit — and it takes the same shape of fix: the
model's own spec supplies the default, and an explicit override is an
override of *that*, not of a global constant.

`cohere` and `voyage` are members of `RerankProtocol` but have no registry
entries: their adapters are a deliberate deferral, not an oversight. The
protocol shapes are known (see the design doc's wire table) and the
registry makes each a small addition, but shipping wire code that has
never been run against the vendor is not free. They are added when there
is a model entry that has actually been exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RerankProtocol = Literal["jina", "cohere", "voyage", "tei"]
RerankRoute = Literal["native", "local"]


@dataclass(frozen=True)
class RerankSpec:
    """How to talk to a reranker, and how to read what it says back."""

    # Which wire dialect the adapter must speak. Not derivable from the
    # route: `native` covers three mutually incompatible vendor APIs.
    protocol: RerankProtocol
    routes: tuple[RerankRoute, ...]
    # Used when RERANK_ROUTE is unset, so switching RERANK_MODEL alone is
    # enough. Must be a member of `routes`.
    default_route: RerankRoute
    # The model id sent on the wire. Empty for protocols that carry no
    # model field — a TEI container serves exactly one model, so the id
    # identifies which checkpoint to launch, not a request parameter.
    wire_id: str
    # The range the scores land in, and the only thing that makes
    # `default_score_floor` interpretable. "logit" is unbounded and
    # centred near zero; "unit" is a sigmoid output in [0, 1]. A floor
    # tuned for one is not merely suboptimal on the other — on the
    # logit -> unit direction it disables the relevance gate outright,
    # silently. Belongs to the model *as served*: the same checkpoint can
    # emit either scale depending on what the adapter asks for.
    score_scale: Literal["logit", "unit"]
    # What RERANK_SCORE_FLOOR adopts when unset. Note that `0.0` is a
    # legitimate floor on the logit scale, which is why the setting's
    # unset sentinel has to be None rather than 0.0.
    default_score_floor: float


REGISTRY: dict[str, RerankSpec] = {
    # Listwise rather than a cross-encoder: v3.5 ranks the whole candidate
    # list in one forward pass. Served only from Jina's own endpoint, so
    # there is no local route. This entry reproduces today's shipped
    # behaviour exactly — see the Settings defaults it is pinned against
    # in tests/test_rerank_models.py.
    "jina-reranker-v3.5": RerankSpec(
        protocol="jina",
        routes=("native",),
        default_route="native",
        wire_id="jina-reranker-v3.5",
        # Unbounded logits, stated in .env.example's RERANK_ENABLED note
        # and confirmed by the v3.5 responses recorded in
        # retrieval/rerank.py's module docstring.
        score_scale="logit",
        # A negative score means "judged less relevant than neutral", so
        # 0.0 is the natural boundary for a logit-scaled model.
        default_score_floor=0.0,
    ),
    # First self-hosted entry: a TEI container in front of the Qwen3
    # reranker checkpoint. TEI's /rerank body has no model field, so
    # `wire_id` records which checkpoint the container must serve rather
    # than anything sent per request. The collection also publishes 0.6B
    # and 8B variants; the size is an operator choice and a separate
    # registry entry if we ever want to name both.
    "qwen3-reranker": RerankSpec(
        protocol="tei",
        routes=("local",),
        default_route="local",
        wire_id="Qwen/Qwen3-Reranker-4B",
        # The checkpoint itself emits raw logit differences. It reads as
        # "unit" here because the TEI adapter sends `raw_scores: false`,
        # which applies the sigmoid server-side. The scale is a property
        # of the model *as this codebase calls it*, not of the weights —
        # flipping that flag to true invalidates the floor below.
        score_scale="unit",
        # PROVISIONAL — a plausible midpoint for a sigmoid score, not a
        # measured one. Nothing has been reranked with this model yet.
        # Calibrate against the retrieval eval set before treating it as
        # a real threshold, and do not copy it into docs as if it were.
        default_score_floor=0.5,
    ),
}


def spec(model: str) -> RerankSpec:
    """Look up a reranker, failing loudly on an unknown name."""
    try:
        return REGISTRY[model]
    except KeyError:
        raise ValueError(
            f"Unknown rerank model {model!r}; expected one of {sorted(REGISTRY)}"
        ) from None
