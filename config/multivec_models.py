"""Late-interaction (multi-vector) model registry.

The third capability slot, and the one whose independence was never
expressible before. `MULTIVEC_ENABLED` was a boolean gated on the *dense*
model: `config/settings.py` rejected it unless `EMBEDDING_MODEL=jina-v4` via
`native`, because jina-v4 happens to emit ColBERT vectors as well as dense
ones. That coupling meant changing the text embedder silently removed the
`visual_search` tool — a capability disappearing as a side effect of an
unrelated choice.

Nothing actually ties the two together. Late-interaction vectors live in
their own collection (`documents_multivec`) with their own dimensionality and
their own MaxSim comparator; they share no vector space with `documents_dense`
and are queried by a different tool. So this is a slot, not a flag: an empty
`MULTIVEC_MODEL` means the slot is unfilled and `visual_search` is
unavailable, and any registered model can fill it regardless of what the
dense slot holds.

`dimensions` is a field here rather than the `MULTIVEC_DIM` constant it
replaces (`config/constants.py`), because it is a property of the model. The
ColPali family standardised on 128, but that is a convention rather than a
guarantee, and a constant would have to be edited — invisibly to the
registry — for a model that chose otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MultivecRoute = Literal["native", "local"]


@dataclass(frozen=True)
class MultivecSpec:
    """A late-interaction model and where it can be reached."""

    routes: tuple[MultivecRoute, ...]
    # Used when MULTIVEC_ROUTE is unset, so naming the model is enough.
    # Must be a member of `routes`.
    default_route: MultivecRoute
    # Per-token vector width. Sizes `documents_multivec` at provisioning
    # time, so changing it on a populated deployment requires recreating
    # that collection — but NOT `documents_dense`, which is the whole point
    # of the slot being separate.
    dimensions: int
    # The id sent on the wire (native routes), or the checkpoint a local
    # container must serve (local routes, where there is no model field).
    wire_id: str = ""


REGISTRY: dict[str, MultivecSpec] = {
    # Reproduces today's behaviour exactly: MULTIVEC_ENABLED=true previously
    # meant "jina-v4 via native, emitting ColBERT at 128 dims", which is this
    # entry under a name. Selecting it no longer says anything about which
    # model embeds text.
    #
    # Note this is the SAME model as the dense slot's `jina-v4` and the same
    # API call with `output_type: colbert` — the duplication is deliberate,
    # because the two slots must be nameable apart even when one model can
    # serve both.
    "jina-v4-colbert": MultivecSpec(
        routes=("native",),
        default_route="native",
        dimensions=128,
        wire_id="jina-embeddings-v4",
    ),
    # The self-hosted option, and the reason this slot exists: ColQwen is the
    # current late-interaction family for visual document retrieval, served
    # through colpali-engine rather than a vendor API. Page images encode to
    # a grid of 128-d patch vectors scored by MaxSim — exactly the shape
    # `documents_multivec` is already provisioned for.
    "colqwen2.5": MultivecSpec(
        routes=("local",),
        default_route="local",
        dimensions=128,
        wire_id="vidore/colqwen2.5-v0.2",
    ),
}


def spec(model: str) -> MultivecSpec:
    """Look up a late-interaction model, failing loudly on an unknown name."""
    try:
        return REGISTRY[model]
    except KeyError:
        raise ValueError(
            f"Unknown multivector model {model!r}; expected one of {sorted(REGISTRY)}"
        ) from None
