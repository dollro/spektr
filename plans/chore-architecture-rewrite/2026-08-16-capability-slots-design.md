# Capability Slots — Design

**Branch:** `chore/architecture-rewrite`
**Date:** 2026-08-16
**Status:** Design, pending plan approval
**Implementation plan:** [`2026-08-16-capability-slots-plan.md`](./2026-08-16-capability-slots-plan.md)

**Verification basis.** Every claim below carries a `file:line` citation into the
working tree at `chore/architecture-rewrite` (forked from `develop` @ `f0c47c9`).
Model capability and licensing claims are attributed inline and were checked
against the primary model card where one exists.

---

## Goal

Make the reranker, the late-interaction model, and the dense embedder
**independently selectable**, so that no single vendor is load-bearing for the
whole retrieval path — and so that a fully self-hosted configuration becomes
expressible.

Two goals from the original project brief are currently unreachable:

- **Provider sovereignty.** `create_embedder()` dispatches to exactly three
  vendor HTTP clients (`ingestion/embedder.py:115-143`). There is no
  self-hosted route. The shipped default is `gemini-2` via OpenRouter
  (`config/settings.py:29`, `.env.example:56`) — Google's model through a US
  gateway that fans out to undisclosed upstream endpoints.
- **Reranker flexibility.** `retrieval/rerank.py:32,52` hard-wires the
  reranker to `settings.jina_api_url` and `settings.jina_api_key`, independent
  of which embedding model is active.

---

## The blocking constraint

`EMBEDDING_MODEL` selects **one** model that must supply text embedding, image
embedding, and ColBERT multi-vector output. `config/embedding_models.py`
expresses those as route tuples on a single `ModelSpec` (`image_routes`,
`multivector_routes`), and `Settings` validates them as properties of that one
choice (`config/settings.py:267-284`).

No self-hostable model supplies all three. So for as long as one key picks all
three capabilities, sovereignty is unreachable **by construction** — this is a
shape problem, not a missing-provider problem.

---

## The decomposition

Not four free slots. The vector-space constraint makes some of these genuinely
inseparable, and the design should say so rather than pretend otherwise.

| Slot | Today | Why it is (or isn't) separable |
|---|---|---|
| **dense** — text **+** image | `EMBEDDING_MODEL` × `EMBEDDING_ROUTE` | **Not separable.** Both write the named vector `dense` on `documents_dense` (`config/constants.py:113`, `ingestion/page_processor.py:188,294`). Two models means two vector spaces in one field. Stays one slot. |
| **multivec** | `MULTIVEC_ENABLED` (bool) | **Separable.** Own collection, own dimension, own comparator (`ingestion/qdrant_setup.py:60-88`). Nothing ties it to the dense model except the current registry shape. |
| **rerank** | hard-wired to Jina | **Separable.** Not a vector at all — a query-time scoring call over text. |

The dense slot keeps its current keys unchanged. The other two become
first-class model×route selections using the same registry pattern.

### What this unlocks

- `qwen3-embedding-8b@local` (dense) + `colqwen2.5@local` (multivec) +
  `qwen3-reranker@local` (rerank) — a fully self-hosted stack. All three speak
  an OpenAI-compatible or TEI-compatible HTTP API, so `LocalEmbedder` is
  largely `OpenRouterEmbedder` with the base URL moved and the task mapping
  swapped.
- `gemini-2@openrouter` (dense, keeping shared text+image space) +
  `colqwen2.5@local` (multivec) — a mixed deployment where only the visual
  half is sovereign.
- Any dense model with a self-hosted reranker, which removes the last hosted
  dependency from the query path without touching the index.

### The cost, stated up front

**Cross-modal retrieval from one shared vector space is a `gemini-2`
privilege.** A text-only self-hosted dense model means images are no longer in
`documents_dense` and are reachable only through `visual_search`/multivec, not
through the ordinary dense channel.

The existing validator already enforces this correctly — `IMAGE_EMBED_STRATEGY`
must be `none` when the active pair has no image support
(`config/settings.py:278-284`) — so the switch fails loudly at startup rather
than silently dropping pages. No new machinery needed; the consequence just has
to be a decision rather than a discovery.

Getting the shared space back while self-hosting means `jina-embeddings-v5-omni`,
which is CC BY-NC 4.0 (verified against the model card) and needs a commercial
licence from Elastic. That is a procurement decision, not an engineering one.

---

## Reranker design

This is the part with a real trap in it, and it is why the reranker gets a
registry rather than a `RERANK_BASE_URL` string.

### Wire protocols do not converge

Unlike `/v1/embeddings`, rerank APIs are not standardised. The response field
alone differs across vendors:

| Protocol | Path | Request | Response |
|---|---|---|---|
| Jina | `/v1/rerank` | `model, query, documents, top_n` | `{"results": [{"index", "relevance_score"}]}` |
| Cohere | `/v2/rerank` | `model, query, documents, top_n` | `{"results": [{"index", "relevance_score"}]}` |
| Voyage | `/v1/rerank` | `model, query, documents, top_k` | `{"data": [{"index", "relevance_score"}]}` |
| TEI / Infinity | `/rerank` | `query, texts, raw_scores` | `[{"index", "score"}]` |

So the slot needs a **protocol adapter**, mirroring `ingestion/embedders/`.
`retrieval/rerankers/` with one module per protocol, behind a minimal protocol:

```python
class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]: ...   # (original index, score)
    async def close(self) -> None: ...
```

`(index, score)` is the entire contract the callers need — both `rerank()` and
`rerank_dicts()` read only `index` and `relevance_score` from the response
today (`retrieval/rerank.py:107-111,136-141`), a property the module docstring
already relies on to survive the v2→v3.5 shape change.

### Score scale is reranker-specific — and the gate depends on it

`RERANK_SCORE_FLOOR` defaults to `0.0` (`config/settings.py:147`), calibrated
for `jina-reranker-v3.5`, whose scores are **unbounded logits** — `.env.example:323-325`
states this explicitly. A Cohere or TEI cross-encoder returns sigmoid scores in
`[0, 1]`, where a floor of `0.0` can never fire.

Swapping the reranker would therefore silently disable the relevance gate
(`retrieval/gate.py:21-33`) with no error and no log line saying so. That is
exactly the class of failure the embedding registry was built to prevent —
`EMBEDDING_DIMENSIONS=768` carrying over to a model that only emits
`256/512/1024/2048` (`config/embedding_models.py:44-48`).

**The fix is the same fix.** `RerankSpec` carries `score_scale` and
`default_score_floor`; `RERANK_SCORE_FLOOR` unset adopts the model's default,
exactly as `EMBEDDING_DIMENSIONS=0` adopts `recommended_dimensions`.

One correction from the embedding precedent: `0.0` cannot be the "unset"
sentinel here, because `0.0` is a legitimate floor for a logit-scaled model.
The field becomes `float | None`, with `None` meaning "take the model's
default".

### Graphiti's cross-encoder joins the slot

`ingestion/graphiti_client.py:91` passes `OpenAIRerankerClient` — a second,
independent, hosted reranker in the ingestion path. Once `retrieval/rerankers/`
exists, Graphiti's cross-encoder becomes a thin adapter over the configured
slot, exactly as `_JinaGraphitiEmbedder` (`ingestion/graphiti_client.py:39-67`)
already wraps the configured embedder. One reranker choice, honoured everywhere.

---

## Migration semantics

What a slot change costs is not uniform, and the docs must say which is which:

| Slot changed | Cost |
|---|---|
| **rerank** | **None.** Query-time only. No re-ingest, no collection change. |
| **multivec** | Re-ingest of `documents_multivec` only. `documents_dense` untouched. |
| **dense** | Full re-ingest + collection recreate (existing behaviour, `docs/operations/reindex.md`). |

The rerank slot being free is what makes it the right first phase: it is the
user-visible win with the lowest blast radius.

### Removed keys

`MULTIVEC_ENABLED` is replaced by `MULTIVEC_MODEL` (empty = disabled). It
follows the house precedent set by `EMBEDDING_PROVIDER`
(`config/settings.py:41,228-238`): the old key is declared on `Settings` purely
so a stale value raises a migration error rather than being swallowed by
`extra="ignore"` and silently booting with a different capability set.

`RERANK_ENABLED` is **kept**. It is a genuine off switch with documented
interaction with the gate, and collapsing it into "is the model string empty"
would lose that.

---

## Defects folded into this work

The audit findings that this restructuring dissolves as a side effect, rather
than needing separate fixes:

- **D1** — rerank 401s under the default config. The slot carries its own key
  and route; the validator requires them together.
- **D5** — `visual_search` silently disappearing when the dense model changes.
  The multivec slot is independent, so it no longer can.

The remaining audit findings (D2, D3, D4, D6, D7, D8) are unrelated to the slot
model but are cheap, are in the same files, and are sequenced into the plan's
final phase.

---

## Explicitly out of scope

**Graph fusion.** Wiring graph hits into `rrf()` — and the chunk-level
provenance it requires — is a separate plan. It has a prerequisite
(`GLiNEREngine.ingest` stores only `e.source = source_key`,
`ingestion/graph_engine.py:284-299`, so no fact can be joined to the chunk that
supports it) and it needs eval evidence to justify a design that is not
obviously a win.

**Model selection.** This plan makes models selectable; it does not pick them.
`jina-v4` → v5-omni, and `jina-v4` ColBERT → ColQwen, are retrieval changes
gated by `task eval-retrieval` (`CLAUDE.md`, Eval gate). Choosing before the
baseline is trustworthy means choosing twice.

---

## Sequencing rationale

Phase 0 comes first because D1, D2 and D5 are all the same systemic finding —
**the default configuration path is the least-exercised path in the repo**. All
three are invisible at runtime (a `degraded` key, a wrong base URL, a disabled
feature), and the unit suite mocks the embedder (`tests/conftest.py:80-96`), so
none of them can surface. Until a fresh-config boot is asserted, no later phase
has a baseline it can trust.

Phases then run cheapest-blast-radius first: rerank (no re-ingest) → local
dense route (re-ingest, but opt-in) → multivec slot (partial re-ingest) →
cleanup.
