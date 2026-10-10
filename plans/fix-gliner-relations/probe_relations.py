"""Read-only probe: why do GLiNER relations not reach the graph?

Runs the same extraction + post-filters as GLiNEREngine.ingest over the PDFs
in documents/, writes nothing anywhere, and classifies every raw relation by
the filter that would drop it. Base schema only (no LLM schema induction).

    PYTHONPATH=. uv run python plans/fix-gliner-relations/probe_relations.py \
        [docs_dir] [threshold=0.5] [window_chars=0]

window_chars > 0 splits each page into windows of about that size (H4);
page texts are cached next to this script so reruns skip Docling.
"""

import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

from gliner2 import GLiNER2

from config.constants import ENTITY_TYPES, RELATION_CONSTRAINTS, RELATIONSHIP_TYPES
from ingestion.file_processor import file_to_pages
from ingestion.graph_engine import GLiNEREngine

MIN_CHARS = 200  # GLiNEREngine._merge_chunks default


def classify(rel_key: str, head: str, tail: str, name_types: dict[str, set[str]]) -> str:
    noise = GLiNEREngine._is_noise.__get__(GLiNEREngine.__new__(GLiNEREngine))
    if not head or not tail:
        return "empty"
    if head == tail:
        return "self_ref"
    if noise(head) or noise(tail):
        return "noise"
    h, t = name_types.get(head, set()), name_types.get(tail, set())
    if rel_key not in RELATION_CONSTRAINTS:
        return "kept_unconstrained" if h and t else "kept_but_node_missing"
    if not h and not t:
        return "drop_both_not_entities"
    if not h:
        return "drop_head_not_entity"
    if not t:
        return "drop_tail_not_entity"
    allowed_s, allowed_t = RELATION_CONSTRAINTS[rel_key]
    if not (h & allowed_s and t & allowed_t):
        return "drop_type_mismatch"
    return "kept"


def load_texts(docs: Path) -> dict[str, list[str]]:
    cache = Path(__file__).with_name("page_texts.json")
    if cache.exists():
        loaded: dict[str, list[str]] = json.loads(cache.read_text())
        return loaded
    texts = {
        pdf.name: [p.text for p in file_to_pages(pdf.name, pdf.read_bytes()).pages]
        for pdf in sorted(docs.glob("*.pdf"))
    }
    cache.write_text(json.dumps(texts))
    return texts


def windows(text: str, size: int) -> list[str]:
    if size <= 0:
        return [text]
    return textwrap.wrap(" ".join(text.split()), size, break_long_words=False)


def main() -> None:
    docs = Path(sys.argv[1] if len(sys.argv) > 1 else "documents")
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    schema = model.create_schema().entities(ENTITY_TYPES).relations(RELATIONSHIP_TYPES)

    outcomes: Counter[str] = Counter()
    by_rel: Counter[tuple[str, str]] = Counter()
    samples: dict[str, list[str]] = {}
    kept_edges: set[tuple[str, str, str]] = set()
    n_texts = n_entities = 0
    for doc, pages in load_texts(docs).items():
        texts = [w for page in pages if len(page.strip()) >= MIN_CHARS
                 for w in windows(page.strip(), window)]
        for text in texts:
            n_texts += 1
            result = model.extract(text, schema, threshold=threshold)
            name_types: dict[str, set[str]] = {}
            for etype, names in result.get("entities", {}).items():
                for name in names:
                    name_types.setdefault(name.strip().title(), set()).add(etype)
            n_entities += len(name_types)
            for rel_type, pairs in result.get("relation_extraction", {}).items():
                rel_key = rel_type.lower().replace(" ", "_")
                for head, tail in pairs:
                    h, t = head.strip().title(), tail.strip().title()
                    outcome = classify(rel_key, h, t, name_types)
                    outcomes[outcome] += 1
                    by_rel[(rel_key, outcome)] += 1
                    if outcome.startswith("kept"):
                        kept_edges.add((h, rel_key, t))
                    ex = samples.setdefault(outcome, [])
                    if len(ex) < 8:
                        ex.append(
                            f"{h} {sorted(name_types.get(h, []))} -[{rel_key}]-> "
                            f"{t} {sorted(name_types.get(t, []))}"
                        )
        print(f"done {doc}: {len(texts)} texts", file=sys.stderr)

    print(json.dumps({
        "threshold": threshold,
        "window_chars": window,
        "texts": n_texts,
        "distinct_entities_per_text_sum": n_entities,
        "raw_relations": sum(outcomes.values()),
        "distinct_kept_edges": len(kept_edges),
        "kept_edges": sorted(" | ".join(e) for e in kept_edges),
        "outcomes": dict(outcomes.most_common()),
        "by_relation": {f"{r}|{o}": c for (r, o), c in sorted(by_rel.items())},
        "samples": samples,
    }, indent=2))


if __name__ == "__main__":
    main()
