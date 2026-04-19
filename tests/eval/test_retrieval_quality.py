"""End-to-end retrieval quality eval via RAGAS.

For each golden-set item: hits hybrid_search (vector + graph) to get
contexts, generates an answer via Anthropic Claude, then scores the
whole batch with RAGAS metrics.  Asserts each metric against the
threshold in tests/eval/thresholds.yaml.

Tests are marked `eval` so they only run under `task eval`.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from config.settings import settings

pytestmark = pytest.mark.eval


ANSWER_PROMPT = """\
You are a helpful assistant. Answer the question using ONLY the context
below. If the context does not contain enough information, reply with
"I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""


def _filter_golden(
    golden: list[dict[str, Any]], ingested: set[str]
) -> list[dict[str, Any]]:
    """Drop items whose required_source isn't currently ingested."""
    return [item for item in golden if item.get("required_source") in ingested]


async def _retrieve(query: str) -> list[str]:
    """Call hybrid_search and extract context strings."""
    from server.tools.hybrid_search import hybrid_search

    result = await hybrid_search(query, limit=5)
    contexts: list[str] = []
    for r in result.get("vector_results", []):
        text = r.get("text") or r.get("text_content") or ""
        if text:
            contexts.append(text)
    for r in result.get("graph_results", []):
        fact = r.get("fact")
        if fact:
            contexts.append(fact)
    return contexts


async def _generate_answer(question: str, contexts: list[str]) -> str:
    """Minimal LLM call using the same dispatcher the rest of the app uses."""
    from ingestion.entity_extractor import get_llm_client

    client = get_llm_client()
    prompt = ANSWER_PROMPT.format(
        context="\n\n---\n\n".join(contexts) if contexts else "(no context found)",
        question=question,
    )
    return await client.chat(messages=[{"role": "user", "content": prompt}])


async def _build_dataset(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build RAGAS rows: {user_input, response, retrieved_contexts}."""
    rows: list[dict[str, Any]] = []
    for item in items:
        contexts = await _retrieve(item["question"])
        answer = await _generate_answer(item["question"], contexts)
        rows.append(
            {
                "user_input": item["question"],
                "response": answer,
                "retrieved_contexts": contexts or ["(no context retrieved)"],
                "reference": item.get("expected_answer_hint", ""),
                "_item_id": item["id"],
            }
        )
    return rows


def _build_ragas_llm() -> Any:  # type: ignore[valid-type]
    """Build a RAGAS-compatible LLM wrapper matching the app's LLM config."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    # RAGAS works well with OpenAI-compatible endpoints (Anthropic direct,
    # OpenRouter, Ollama, vLLM, etc.) via ChatOpenAI + base_url. For pure
    # Anthropic we could branch to ChatAnthropic, but the OpenAI-compatible
    # path covers both.
    if settings.llm_base_url:
        base = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=60,
        )
    elif settings.llm_api_type == "anthropic":
        from langchain_anthropic import ChatAnthropic

        base = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout=60,
            stop=None,
        )
    else:
        base = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout=60,
        )
    return LangchainLLMWrapper(base)


def _score_with_ragas(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Run RAGAS over the prepared rows and return mean scores per metric."""
    from datasets import Dataset
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    ds_rows = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in rows
    ]
    dataset = Dataset.from_list(ds_rows)

    llm = _build_ragas_llm()
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            LLMContextPrecisionWithoutReference(),
            ResponseRelevancy(),
        ],
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
    )

    # RAGAS returns a Result with per-row scores; average per metric.
    scores = result.scores
    metric_means: dict[str, float] = {}
    if scores:
        keys = scores[0].keys()
        for k in keys:
            vals = [
                s[k]
                for s in scores
                if k in s and isinstance(s[k], int | float)
            ]
            if vals:
                metric_means[k] = sum(vals) / len(vals)
    return metric_means


def _normalize_metric_name(name: str) -> str:
    """Map RAGAS metric column names to threshold keys."""
    mapping = {
        "faithfulness": "faithfulness",
        "llm_context_precision_without_reference": "context_precision",
        "answer_relevancy": "answer_relevancy",
    }
    return mapping.get(name, name)


def _write_report(
    reports_dir: Path,
    rows: list[dict[str, Any]],
    scores: dict[str, float],
    thresholds: dict[str, float],
) -> Path:
    """Dump a timestamped JSON artifact for trending + debugging."""
    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"{ts}.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "row_count": len(rows),
                "scores": scores,
                "thresholds": thresholds,
                "rows": [
                    {
                        "id": r.get("_item_id"),
                        "question": r["user_input"],
                        "answer": r["response"][:400],
                        "context_count": len(r["retrieved_contexts"]),
                    }
                    for r in rows
                ],
            },
            indent=2,
        )
    )
    return path


REFUSAL_PHRASES = ("don't have enough information", "no information", "cannot answer")


def test_retrieval_quality_meets_thresholds(
    golden_set: list[dict[str, Any]],
    thresholds: dict[str, float],
    ingested_sources: set[str],
    eval_reports_dir: Path,
) -> None:
    """Full eval loop: filter → retrieve → answer → score → assert."""
    items = _filter_golden(golden_set, ingested_sources)
    if not items:
        pytest.skip(
            f"No golden items match ingested sources {sorted(ingested_sources)}. "
            "Run `task ingest` on documents referenced by golden_set.yaml."
        )

    rows = asyncio.run(_build_dataset(items))

    # Negative / out-of-domain rows are checked separately via a refusal
    # phrase — RAGAS penalizes "I don't know" answers on answer_relevancy.
    def _is_negative(row: dict[str, Any]) -> bool:
        item = next(i for i in items if i["id"] == row["_item_id"])
        return bool(item.get("skip_in_ragas"))

    ragas_rows = [r for r in rows if not _is_negative(r)]
    negative_rows = [r for r in rows if _is_negative(r)]

    # 1. Metric-based assertion on non-negative rows
    raw_scores = _score_with_ragas(ragas_rows) if ragas_rows else {}

    # Normalize RAGAS column names and report
    scores = {_normalize_metric_name(k): v for k, v in raw_scores.items()}
    report = _write_report(eval_reports_dir, rows, scores, thresholds)
    print(f"\nEval report: {report}")
    print(f"Scores: {scores}")

    failures: list[str] = []
    for metric, floor in thresholds.items():
        actual = scores.get(metric)
        if actual is None:
            failures.append(f"{metric}: MISSING (got none; available: {list(scores)})")
        elif actual < floor:
            failures.append(f"{metric}: {actual:.3f} < {floor:.2f} (threshold)")

    # 2. Refusal assertion on negative rows
    for row in negative_rows:
        answer_lc = row["response"].lower()
        if not any(phrase in answer_lc for phrase in REFUSAL_PHRASES):
            failures.append(
                f"negative Q {row['_item_id']}: expected refusal phrase, "
                f"got: {row['response'][:160]!r}"
            )

    assert not failures, "\n".join(failures)
