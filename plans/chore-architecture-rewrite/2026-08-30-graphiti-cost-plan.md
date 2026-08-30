# Graphiti Cost Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop Graphiti ingestion paying full model price for calls it declared
cheap, and stop the bulk-ingest fallback reprocessing episodes it already wrote.

**Architecture:** A `SizeAwareGenericClient` subclasses `OpenAIGenericClient`
and dispatches `ModelSize.small` calls to a second client instance configured
with a cheaper model; a new `GRAPH_SMALL_MODEL` setting leaves this inert when
empty. Separately, `GraphitiWriter.ingest_bulk` queries Neo4j for episodes
already committed by a failed `add_episode_bulk` and skips them in the
sequential fallback.

**Tech Stack:** Python 3.13, graphiti-core 0.28.1, pydantic-settings, Neo4j
5.26, pytest/pytest-asyncio, ruff (95 cols), mypy strict.

**Spec:** `plans/chore-architecture-rewrite/2026-08-30-graphiti-cost-design.md`

## Global Constraints

- Python 3.13; ruff line length **95**; mypy **strict**.
- `graphiti-core` is pinned at **0.28.1**. Every line/attribute reference below
  was verified against that version. Do not upgrade as part of this work.
- New `Settings` fields MUST be added to `.env.example` or
  `tests/test_default_config.py::test_env_example_declares_every_settings_field`
  fails. Per the env-trio contract, add to `.env`, `.env.example` **and**
  `.env.prod`.
- `GRAPH_SMALL_MODEL` defaults to `""` (empty = today's behaviour exactly).
  Do not ship a non-empty default.
- Commit messages: `<type>(<scope>): <subject>`. Never mention Claude.
- Run `task lint && task typecheck && task test` before each commit.

---

### Task 1: Pin the upstream contract we are working around

Written first because it is the guard: it defines exactly which upstream
behaviour Task 2 exists to compensate for, and fails loudly when that changes.

**Files:**
- Create: `tests/test_graphiti_llm_contract.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing importable. A regression fence only.

- [ ] **Step 1: Write the failing test**

```python
"""Pins the graphiti-core behaviour that `SizeAwareGenericClient` works around.

If either test fails after a graphiti-core upgrade, read
plans/chore-architecture-rewrite/2026-08-30-graphiti-cost-design.md before
touching anything: the workaround may have become unnecessary, or may have
silently stopped working.
"""

from __future__ import annotations

import inspect

from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient


def test_generate_response_signature_is_pinned() -> None:
    """SizeAwareGenericClient overrides this exact signature.

    A change here means our override no longer overrides anything, and every
    ModelSize.small call silently reverts to the expensive model.
    """
    sig = inspect.signature(OpenAIGenericClient._generate_response)
    assert list(sig.parameters) == [
        "self",
        "messages",
        "response_model",
        "max_tokens",
        "model_size",
    ]


def test_upstream_still_ignores_model_size() -> None:
    """OpenAIGenericClient sends self.model regardless of model_size.

    When this fails, upstream has fixed the bug: delete SizeAwareGenericClient
    and set LLMConfig.small_model instead.
    """
    src = inspect.getsource(OpenAIGenericClient._generate_response)
    assert "model=self.model or DEFAULT_MODEL" in src
    assert "_get_model_for_size" not in src
```

- [ ] **Step 2: Run the test — it must PASS immediately**

Run: `uv run pytest tests/test_graphiti_llm_contract.py -v`
Expected: 2 passed. (This task pins existing behaviour, so green is correct.
If either fails now, stop — the whole premise of Task 2 is wrong.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_graphiti_llm_contract.py
git commit -m "test(graph): pin the graphiti-core model-size behaviour we work around"
```

---

### Task 2: Route `ModelSize.small` calls to a cheaper model

**Files:**
- Create: `ingestion/graphiti_llm.py`
- Create: `tests/test_graphiti_llm.py`
- Modify: `config/settings.py` (add field next to `graph_semaphore_limit`, ~line 221)
- Modify: `ingestion/graphiti_client.py:153-167`
- Modify: `.env.example`, `.env`, `.env.prod`

**Interfaces:**
- Consumes: `settings.graph_small_model` (str, `""` when unset).
- Produces: `SizeAwareGenericClient(config: LLMConfig, small_model: str = "")`
  with attribute `_delegate: OpenAIGenericClient | None`.

- [ ] **Step 1: Add the setting**

In `config/settings.py`, directly after `graph_episode_target_size`:

```python
    # Graphiti tags per-node dedupe/attribute and per-edge resolution calls
    # with ModelSize.small -- 62% of all calls on the 2026-08-30 baseline.
    # OpenAIGenericClient ignores that tag and sends LLM_MODEL for everything.
    # Naming a cheaper model here routes those calls to it. Empty keeps the
    # single-model behaviour. The model MUST support
    # `response_format: json_schema` or every dedupe call fails.
    graph_small_model: str = ""
```

- [ ] **Step 2: Declare it in all three env files**

Append to the graph section of `.env.example`, `.env` and `.env.prod`:

```bash
GRAPH_SMALL_MODEL=                      # Cheap model for Graphiti's dedupe /
                                        # edge-resolution calls (62% of them).
                                        # Empty = use LLM_MODEL for everything.
                                        # Must support json_schema output.
```

- [ ] **Step 3: Run the config test to verify the env files are consistent**

Run: `uv run pytest tests/test_default_config.py -v`
Expected: PASS. A failure names `GRAPH_SMALL_MODEL` as missing from
`.env.example` — fix the file rather than the allowlist.

- [ ] **Step 4: Write the failing tests**

```python
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message

from ingestion.graphiti_llm import SizeAwareGenericClient


def _config() -> LLMConfig:
    return LLMConfig(api_key="k", model="big/model", base_url="http://endpoint")


def test_delegate_uses_small_model_on_the_same_endpoint() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="small/model")
    assert client.model == "big/model"
    assert client._delegate is not None
    assert client._delegate.model == "small/model"


def test_no_delegate_when_small_model_is_empty() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="")
    assert client._delegate is None


@pytest.mark.asyncio
async def test_small_calls_go_to_the_delegate() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="small/model")
    assert client._delegate is not None
    client._delegate._generate_response = AsyncMock(return_value={"routed": "small"})

    with patch.object(
        OpenAIGenericClient,
        "_generate_response",
        new=AsyncMock(return_value={"routed": "big"}),
    ):
        result = await client._generate_response(
            [Message(role="user", content="hi")],
            model_size=ModelSize.small,
        )

    assert result == {"routed": "small"}
    client._delegate._generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_medium_calls_go_to_the_main_model() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="small/model")
    assert client._delegate is not None
    client._delegate._generate_response = AsyncMock(return_value={"routed": "small"})

    with patch.object(
        OpenAIGenericClient,
        "_generate_response",
        new=AsyncMock(return_value={"routed": "big"}),
    ):
        result = await client._generate_response(
            [Message(role="user", content="hi")],
            model_size=ModelSize.medium,
        )

    assert result == {"routed": "big"}
    client._delegate._generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_small_calls_fall_back_to_main_model_when_unconfigured() -> None:
    client = SizeAwareGenericClient(config=_config(), small_model="")

    with patch.object(
        OpenAIGenericClient,
        "_generate_response",
        new=AsyncMock(return_value={"routed": "big"}),
    ):
        result = await client._generate_response(
            [Message(role="user", content="hi")],
            model_size=ModelSize.small,
        )

    assert result == {"routed": "big"}
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `uv run pytest tests/test_graphiti_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion.graphiti_llm'`

- [ ] **Step 6: Write the implementation**

Create `ingestion/graphiti_llm.py`:

```python
"""Size-aware LLM client for Graphiti.

Graphiti marks its high-volume maintenance calls -- per-node dedupe and
attribute extraction, per-edge resolution and invalidation -- with
`ModelSize.small`, meaning a cheaper model is adequate. `BaseOpenAIClient`
honours that via `_get_model_for_size`; `OpenAIGenericClient`, the only client
that talks to an arbitrary OpenAI-compatible endpoint and therefore the only
one usable with OpenRouter, does not. It records `model.size` as a tracing
attribute and then sends `self.model` regardless
(`openai_generic_client.py:124,157`).

On the 2026-08-30 baseline that meant 1,824 of 2,920 calls -- 62% -- ran on the
full model at full price.

The fix delegates to a second client rather than mutating `self.model` around a
`super()` call: `SEMAPHORE_LIMIT` is 10, so the instance is shared across
concurrent coroutines and swapping an attribute mid-flight is a data race.
"""

from __future__ import annotations

import typing

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel


class SizeAwareGenericClient(OpenAIGenericClient):
    """`OpenAIGenericClient` that honours `ModelSize.small`.

    With *small_model* empty there is no delegate and behaviour is identical to
    the base class.
    """

    def __init__(self, config: LLMConfig, small_model: str = "") -> None:
        super().__init__(config=config)
        self._delegate: OpenAIGenericClient | None = None
        if small_model:
            self._delegate = OpenAIGenericClient(
                config=LLMConfig(
                    api_key=config.api_key,
                    model=small_model,
                    base_url=config.base_url,
                )
            )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        # The delegate's own `generate_response` is deliberately bypassed: the
        # outer client's already applied tracing and retry before reaching here.
        if model_size == ModelSize.small and self._delegate is not None:
            return await self._delegate._generate_response(
                messages,
                response_model,
                max_tokens=max_tokens,
                model_size=model_size,
            )
        return await super()._generate_response(
            messages,
            response_model,
            max_tokens=max_tokens,
            model_size=model_size,
        )
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_graphiti_llm.py -v`
Expected: 5 passed.

- [ ] **Step 8: Wire it into the Graphiti client**

In `ingestion/graphiti_client.py`, replace the `OpenAIGenericClient` import
(line 37) with:

```python
from ingestion.graphiti_llm import SizeAwareGenericClient  # noqa: E402
```

and replace `llm_client=OpenAIGenericClient(config=llm_config),` (line 164) with:

```python
            llm_client=SizeAwareGenericClient(
                config=llm_config,
                small_model=settings.graph_small_model,
            ),
```

- [ ] **Step 9: Verify the whole suite is green**

Run: `task lint && task typecheck && task test`
Expected: lint clean, mypy clean, all tests pass.

- [ ] **Step 10: Commit**

```bash
git add ingestion/graphiti_llm.py ingestion/graphiti_client.py config/settings.py \
        tests/test_graphiti_llm.py .env.example .env.prod
git commit -m "feat(graph): route Graphiti's small-model calls to GRAPH_SMALL_MODEL"
```

---

### Task 3: Resume the sequential fallback instead of restarting it

**Files:**
- Modify: `ingestion/graph_writer.py:119-194`
- Modify: `tests/test_graph_writer_bulk.py`

**Interfaces:**
- Consumes: `graphiti_core.Graphiti` (for `client.driver.execute_query`).
- Produces: `GraphitiWriter._existing_episode_names(client, source_key) -> set[str]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph_writer_bulk.py`:

```python
class TestGraphitiWriterBulkResume:
    @pytest.mark.asyncio
    async def test_fallback_skips_episodes_already_committed(self) -> None:
        """A failed bulk call leaves episodes behind; the fallback must skip them."""
        from ingestion.graph_writer import GraphitiWriter

        chunks = [
            TextChunk(text="a" * 2000, chunk_index=0, page_number=1),
            TextChunk(text="b" * 2000, chunk_index=1, page_number=2),
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(side_effect=RuntimeError("402"))
        mock_client.driver.execute_query = AsyncMock(
            return_value=([{"name": "test.pdf:p1:c0"}], None, None)
        )

        with (
            patch("ingestion.graph_writer.get_graphiti", return_value=mock_client),
            patch("ingestion.graph_writer.settings") as mock_settings,
        ):
            mock_settings.graph_episode_target_size = 1500
            await GraphitiWriter().ingest_bulk(chunks=chunks, source_key="test.pdf")

        names = [c.kwargs["name"] for c in mock_client.add_episode.call_args_list]
        assert names == ["test.pdf:p2:c1"]

    @pytest.mark.asyncio
    async def test_fallback_processes_all_when_nothing_committed(self) -> None:
        from ingestion.graph_writer import GraphitiWriter

        chunks = [
            TextChunk(text="a" * 2000, chunk_index=0, page_number=1),
            TextChunk(text="b" * 2000, chunk_index=1, page_number=2),
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(side_effect=RuntimeError("402"))
        mock_client.driver.execute_query = AsyncMock(return_value=([], None, None))

        with (
            patch("ingestion.graph_writer.get_graphiti", return_value=mock_client),
            patch("ingestion.graph_writer.settings") as mock_settings,
        ):
            mock_settings.graph_episode_target_size = 1500
            await GraphitiWriter().ingest_bulk(chunks=chunks, source_key="test.pdf")

        assert mock_client.add_episode.await_count == 2

    @pytest.mark.asyncio
    async def test_resume_query_failure_degrades_to_full_reprocess(self) -> None:
        """The optimisation must never turn a recoverable failure into a hard one."""
        from ingestion.graph_writer import GraphitiWriter

        chunks = [
            TextChunk(text="a" * 2000, chunk_index=0, page_number=1),
            TextChunk(text="b" * 2000, chunk_index=1, page_number=2),
        ]

        mock_client = AsyncMock()
        mock_client.add_episode_bulk = AsyncMock(side_effect=RuntimeError("402"))
        mock_client.driver.execute_query = AsyncMock(side_effect=RuntimeError("neo4j down"))

        with (
            patch("ingestion.graph_writer.get_graphiti", return_value=mock_client),
            patch("ingestion.graph_writer.settings") as mock_settings,
        ):
            mock_settings.graph_episode_target_size = 1500
            await GraphitiWriter().ingest_bulk(chunks=chunks, source_key="test.pdf")

        assert mock_client.add_episode.await_count == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_graph_writer_bulk.py -v -k Resume`
Expected: `test_fallback_skips_episodes_already_committed` FAILS with
`names == ['test.pdf:p1:c0', 'test.pdf:p2:c1']` — the current code reprocesses
everything. The other two pass already; they are the regression fence.

- [ ] **Step 3: Add the lookup helper**

In `ingestion/graph_writer.py`, add to `GraphitiWriter` (after `ingest_bulk`):

```python
    @staticmethod
    async def _existing_episode_names(client: Graphiti, source_key: str) -> set[str]:
        """Episode names already in the graph for *source_key*.

        Returns an empty set on any failure. The resume path is an
        optimisation: it must never promote a recoverable bulk failure into a
        hard one, so the worst case is the previous behaviour.
        """
        try:
            records, _, _ = await client.driver.execute_query(
                "MATCH (e:Episodic) WHERE e.source_description = $src "
                "RETURN e.name AS name",
                src=source_key,
            )
        except Exception:
            logger.warning(
                "Could not read committed episodes for %s; reprocessing all",
                source_key,
                exc_info=True,
            )
            return set()
        return {record["name"] for record in records}
```

Add the import at the top of the file, beside the other `graphiti_core` imports:

```python
from graphiti_core import Graphiti
```

- [ ] **Step 4: Filter the fallback loop**

Replace the `# Sequential fallback` block (`graph_writer.py:181-194`) with:

```python
        # Sequential fallback, resuming rather than restarting.
        # add_episode_bulk commits incrementally, so a mid-run failure leaves
        # part of the document in the graph and reports nothing about which
        # part. Reprocessing the full list pays the whole extraction pipeline
        # again for episodes already written -- measured 2026-08-30: 64 of 71
        # episodes duplicated, roughly half the run's LLM spend.
        #
        # The filter runs here rather than before the bulk call on purpose.
        # Episode names are content-independent, so filtering up front would
        # also skip episodes left by an earlier run of an since-edited
        # document. Filtering after a failure keeps the meaning exact: skip
        # what this call already wrote.
        committed = await self._existing_episode_names(client, source_key)
        pending = [episode for episode in episodes if episode.name not in committed]
        if committed:
            logger.info(
                "Resuming %s: %d of %d episodes already committed",
                source_key,
                len(episodes) - len(pending),
                len(episodes),
            )

        for episode in pending:
            try:
                await client.add_episode(
                    name=episode.name,
                    episode_body=episode.content,
                    source_description=episode.source_description,
                    reference_time=episode.reference_time,
                )
            except Exception:
                logger.exception(
                    "Sequential fallback failed for episode %s",
                    episode.name,
                )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_graph_writer_bulk.py -v`
Expected: all pass, including the three new ones.

- [ ] **Step 6: Verify the whole suite is green**

Run: `task lint && task typecheck && task test`
Expected: lint clean, mypy clean, all tests pass.

- [ ] **Step 7: Commit**

```bash
git add ingestion/graph_writer.py tests/test_graph_writer_bulk.py
git commit -m "fix(graph): resume the bulk-ingest fallback instead of restarting it"
```

---

### Task 4: Verify against the recorded baseline

Neither fix is trustworthy until a real ingest confirms it. `GRAPH_SMALL_MODEL`
in particular carries a quality risk (weaker dedupe) and a hard failure risk
(a model that will not emit `json_schema`).

**Files:**
- Modify: `.env` (local only — the chosen model value)
- Modify: `docs/ingestion/knowledge-graph.md`

- [ ] **Step 1: Choose a small model and verify it emits structured output**

Pick a cheap model on the configured `LLM_BASE_URL` and confirm it honours
`response_format: json_schema` before ingesting anything:

```bash
uv run python -c "
import asyncio, os
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message
from ingestion.graphiti_llm import SizeAwareGenericClient
from config.settings import settings

async def main():
    c = SizeAwareGenericClient(
        config=LLMConfig(api_key=settings.llm_api_key, model=settings.llm_model,
                         base_url=settings.llm_base_url or None),
        small_model=settings.graph_small_model,
    )
    from pydantic import BaseModel
    class Out(BaseModel):
        answer: str
    print(await c._generate_response(
        [Message(role='user', content='Reply with {\"answer\": \"ok\"}')],
        response_model=Out, model_size=ModelSize.small))
asyncio.run(main())
"
```

Expected: a parsed dict. A `400` or unparseable response means the model is
unusable here — pick another and repeat. Do **not** proceed to Step 2 until
this returns cleanly.

- [ ] **Step 2: Wipe and re-ingest the same corpus**

```bash
curl -s -X DELETE http://localhost:6333/collections/documents_dense
docker exec spektr-neo4j-1 cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (n) DETACH DELETE n"
rm -rf state/cocoindex.db state/ingestion_failures.db
task ingest 2>&1 | tee /tmp/ingest-after.log
```

- [ ] **Step 3: Compare against the baseline**

```bash
grep -c "POST https://openrouter.ai/api/v1/chat/completions" /tmp/ingest-after.log
docker exec spektr-neo4j-1 cypher-shell -u neo4j -p "$NEO4J_PASSWORD" --format plain \
  "MATCH (n:Entity) WITH n.name AS nm, count(*) AS c WHERE c>1
   RETURN count(*) AS dup_entity_names, sum(c-1) AS surplus"
docker exec spektr-neo4j-1 cypher-shell -u neo4j -p "$NEO4J_PASSWORD" --format plain \
  "MATCH (e:Episodic) WITH e.name AS n, count(*) AS c WHERE c>1 RETURN count(*) AS dup_episodes"
task doctor
```

Acceptance, against `2026-08-30-graphiti-cost-design.md`:

| Check | Baseline | Required |
|-|-|-|
| Duplicate Episodic names | 64 | **0** |
| Duplicate entity names | 0 | **0** (no regression from a weaker model) |
| Duplicate facts | 4 of 1,006 | ≤ 10 |
| Entities / edges | 818 / 1,006 | within ±15% |
| `task doctor` | all in sync | all in sync |
| OpenRouter spend | $6.97 | **record it** |

A duplicate-entity count above 0 means the small model's dedupe is materially
worse — revert `GRAPH_SMALL_MODEL` to empty and record the finding.

- [ ] **Step 4: Run the retrieval eval**

Run: `task eval-retrieval`
Expected: recall@10, nDCG@10 and MRR at or above
`tests/eval/thresholds.yaml`. A drop means the graph got worse; do not ship the
small model on a drop.

- [ ] **Step 5: Document the outcome**

Add a "Cost" section to `docs/ingestion/knowledge-graph.md` recording: the
228× baseline amplification, what `GRAPH_SMALL_MODEL` does and which model was
verified, the measured before/after spend, and that `add_episode_bulk` failures
now resume. Note that Graphiti bills roughly one LLM call per entity plus one
per edge, so cost tracks extracted structure rather than page count.

- [ ] **Step 6: Commit**

```bash
git add docs/ingestion/knowledge-graph.md .env.example
git commit -m "docs(ingestion): record Graphiti's cost model and the small-model slot"
```

---

## Self-Review

**Spec coverage.** Design defect 1 → Tasks 1, 2, 4. Design defect 2 → Task 3.
Design "out of scope" items are absent from the plan, as intended. The design's
"must not default to non-empty" constraint is in Global Constraints and Task 2
Step 1; the "must support json_schema" risk is gated in Task 4 Step 1; the
"private method drift" risk is fenced by Task 1; the "not free money" quality
risk is gated by Task 4 Step 3's duplicate-entity check and Step 4's eval.

**Placeholders.** None — every code step carries the literal code, every run
step the literal command and expected output.

**Type consistency.** `SizeAwareGenericClient(config, small_model)` and
`_delegate` are used identically in Task 2's tests, implementation and Task 4's
smoke script. `_existing_episode_names(client, source_key) -> set[str]` matches
its single call site. `settings.graph_small_model` / `GRAPH_SMALL_MODEL` are
consistent across settings, env files, wiring and verification.
