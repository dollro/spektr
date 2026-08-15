"""Aggregate relevance-gate telemetry from Spektr's JSON logs.

Answers "how often does the hybrid_search retry actually fire, and does it
help?" — the question that decides whether first-stage recall is worth
investing in (late interaction, better embeddings) or whether the reranker
is already seeing the right candidates.

Reads the records emitted by ``retrieval/gate.py::log_gate_decision``: one
per gated pipeline run, fired or not, so the rate has a denominator.

Requires LOG_FORMAT=json (the default). Records are written to stdout by
the MCP server process, so in production read them from the container logs.

Usage:
    docker compose logs mcp | python -m scripts.retry_stats
    python -m scripts.retry_stats --from logs/mcp.log
    python -m scripts.retry_stats                    # read stdin
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field


@dataclass
class GateStats:
    """Counters accumulated over a stream of gate records."""

    total: int = 0
    inert: int = 0  # gate could not fire: no rerank score to test
    fired: int = 0
    empty_first_pass: int = 0
    helped: int = 0
    scores: list[float] = field(default_factory=list)

    @property
    def gated(self) -> int:
        """Runs where the gate could actually fire."""
        return self.total - self.inert


def _iter_records(lines: Iterable[str]) -> Iterator[dict]:  # type: ignore[type-arg]
    """Yield gate-decision records, skipping non-JSON and unrelated lines.

    Container log lines are often prefixed (``mcp-1  | {...}``), so we scan
    for the first brace rather than requiring the line to be pure JSON.
    """
    for line in lines:
        start = line.find("{")
        if start == -1:
            continue
        try:
            record = json.loads(line[start:])
        except (ValueError, TypeError):
            continue
        if isinstance(record, dict) and "gate_fired" in record:
            yield record


def collect(lines: Iterable[str]) -> GateStats:
    """Fold a log stream into counters."""
    stats = GateStats()
    for record in _iter_records(lines):
        stats.total += 1
        if not record.get("gate_reranked"):
            stats.inert += 1
            continue
        score = record.get("top_score")
        if isinstance(score, (int, float)):
            stats.scores.append(float(score))
        if not record.get("gate_fired"):
            continue
        stats.fired += 1
        if score is None:
            stats.empty_first_pass += 1
        if record.get("retry_helped"):
            stats.helped += 1
    return stats


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. `values` must be non-empty."""
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def _pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:5.1f}%" if whole else "    — "


def _render(stats: GateStats) -> str:
    """Format the report. Percentages always name their denominator."""
    if stats.total == 0:
        return (
            "No relevance-gate records found.\n\n"
            "Expected JSON log lines containing 'gate_fired'. Check that\n"
            "LOG_FORMAT=json, the MCP server has served hybrid_search calls\n"
            "since this instrumentation was deployed, and that you are\n"
            "reading that process's stdout."
        )

    out = [f"Relevance gate — {stats.total} evaluations", ""]
    out.append(
        f"  Gate could fire (reranked)  {stats.gated:6d}  {_pct(stats.gated, stats.total)}"
    )
    out.append(
        f"  Gate inert (no rerank)      {stats.inert:6d}  {_pct(stats.inert, stats.total)}"
        "   <- excluded below"
    )
    out.append("")

    if stats.gated == 0:
        out.append("  No run had a rerank score, so the gate never had anything to test.")
        out.append("  Check RERANK_ENABLED and whether the reranker is degrading.")
        return "\n".join(out)

    out.append(
        f"  Retry fired                 {stats.fired:6d}  {_pct(stats.fired, stats.gated)}"
        f"   of {stats.gated} gated runs"
    )
    if stats.fired:
        out.append(
            f"    first pass empty          {stats.empty_first_pass:6d}  "
            f"{_pct(stats.empty_first_pass, stats.fired)}   of fired"
        )
        out.append(
            f"    retry improved top-1      {stats.helped:6d}  "
            f"{_pct(stats.helped, stats.fired)}   of fired"
        )
    out.append("")

    if stats.scores:
        out.append("  Top-1 score before retry (reranker logits, unbounded — not 0..1)")
        for label, pct in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            out.append(f"    {label}  {_percentile(stats.scores, pct):+.3f}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="source",
        help="Log file to read. Defaults to stdin.",
    )
    args = parser.parse_args()

    if args.source:
        with open(args.source, encoding="utf-8", errors="replace") as handle:
            stats = collect(handle)
    else:
        if sys.stdin.isatty():
            parser.error("no input: pipe logs in or pass --from <file>")
        stats = collect(sys.stdin)

    print(_render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
