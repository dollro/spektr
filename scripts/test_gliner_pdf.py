"""Quick smoke test: GLiNER2 entity extraction on a real PDF.

Extracts text from PDF pages, runs GLiNER2 extraction, prints results.
No Neo4j or Qdrant needed.

Usage:
    PYTHONPATH=. uv run python scripts/test_gliner_pdf.py documents/test.pdf
"""

from __future__ import annotations

import sys
import time

from gliner2 import GLiNER2

from config.constants import ENTITY_TYPES, RELATIONSHIP_TYPES
from ingestion.file_processor import docling_chunk, file_to_pages, semantic_chunk
from ingestion.graph_engine import GLiNEREngine


def main(pdf_path: str) -> None:
    with open(pdf_path, "rb") as f:
        content = f.read()

    filename = pdf_path.rsplit("/", 1)[-1]
    print(f"Processing: {filename} ({len(content) / 1024:.0f} KB)\n")

    # 1. Extract pages + Docling document
    result = file_to_pages(filename, content)
    pages = result.pages
    print(f"Pages extracted: {len(pages)}")

    # 2. Get chunks (prefer Docling, fallback to semantic)
    if result.docling_document:
        chunks = docling_chunk(result.docling_document)
        print(f"Docling chunks: {len(chunks)}")
    else:
        chunks = []
        for page in pages:
            chunks.extend(semantic_chunk(page.text, page.page_number))
        print(f"Semantic chunks (fallback): {len(chunks)}")

    # 3. Merge chunks (same logic as GLiNEREngine)
    merged = GLiNEREngine._merge_chunks(chunks)
    avg = sum(len(t) for t in merged) // max(len(merged), 1)
    print(f"Merged page texts: {len(merged)} (avg {avg} chars)")

    # 4. Load GLiNER2
    print("\nLoading GLiNER2 model...")
    t0 = time.monotonic()
    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    print(f"Model loaded in {time.monotonic() - t0:.1f}s")

    schema = (
        extractor.create_schema()
        .entities(ENTITY_TYPES)
        .relations(RELATIONSHIP_TYPES)
    )

    # 5. Extract
    max_texts = min(15, len(merged))
    print(f"\nExtracting from {max_texts} merged texts...\n")

    all_entities: dict[str, set[str]] = {}
    all_relations: list[tuple[str, str, str]] = []

    t0 = time.monotonic()
    for i, text in enumerate(merged[:max_texts]):
        extraction = extractor.extract(text[:1500], schema)

        entities = extraction.get("entities", {})
        relations = extraction.get("relation_extraction", {})

        for etype, names in entities.items():
            all_entities.setdefault(etype, set()).update(
                n.strip().title() for n in names if n.strip()
            )

        for rtype, pairs in relations.items():
            for head, tail in pairs:
                all_relations.append(
                    (head.strip().title(), rtype, tail.strip().title())
                )

        ent_count = sum(len(v) for v in entities.values())
        rel_count = sum(len(v) for v in relations.values())
        print(
            f"  Text {i + 1} ({len(text)} chars):"
            f" {ent_count} entities, {rel_count} relations"
        )

    elapsed = time.monotonic() - t0
    print(f"\nExtraction done in {elapsed:.1f}s ({elapsed / max_texts:.2f}s/text)")

    # 6. Summary
    print("\n--- ENTITIES ---")
    for etype, names in sorted(all_entities.items()):
        if names:
            print(f"  {etype}: {', '.join(sorted(names))}")

    print(f"\n--- RELATIONS ({len(all_relations)}) ---")
    for head, rel, tail in all_relations:
        print(f"  {head} --[{rel}]--> {tail}")

    total_ents = sum(len(v) for v in all_entities.values())
    print(f"\nTotals: {total_ents} unique entities, {len(all_relations)} relations")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "documents/test.pdf"
    main(path)
