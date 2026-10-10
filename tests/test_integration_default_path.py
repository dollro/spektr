"""Phase 0.2 — the default retrieval path must run without degrading.

`multi_search` never raises: every stage in `retrieval/pipeline.py` reports a
failure by appending its name to `degraded` and returning what it has. That
contract is what keeps the tool robust, and it is also what makes silent
breakage cheap — a missing rerank credential, an unreachable sparse model or a
dead graph engine all surface as nothing worse than a slightly poorer answer.

This is the one test that treats `degraded` as a failure, so it asserts a
correctly configured deployment end to end instead of stage by stage. It is
the runtime counterpart to `tests/test_default_config.py`, which asserts the
same thing statically against `.env.example`.

Isolation: the collections are the redirected `test_*` copies and Neo4j is the
ephemeral container, both set up in `tests/conftest.py`. Nothing here may be
pointed at the dev instances — per `CLAUDE.md` the Neo4j fixtures wipe whatever
they are aimed at, and that would destroy the developer's knowledge graph.

Credentials: this exercises the real embedder, the real miniCOIL model and the
real reranker, so it needs the working credentials `conftest.py` loads from
`.env`. That is the point — mocking any of them would remove the only thing
this test checks.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient, models

from config.constants import DENSE_COLLECTION, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from server.tools.multi_search import multi_search

pytestmark = pytest.mark.integration

QUERY = "How does Spektr rerank fused retrieval candidates?"

# (text, source_file). Two documents, one of which plainly answers QUERY, so
# both the lexical and the semantic channel have something to rank.
_DOCS: list[tuple[str, str]] = [
    (
        "Spektr retrieves with a dense vector channel and a sparse miniCOIL "
        "channel, fuses the two lists with reciprocal rank fusion, and then "
        "reranks the merged candidates against the original query.",
        "retrieval.md",
    ),
    (
        "The ingestion pipeline splits each page into chunks, writes them to "
        "the Qdrant dense collection, and extracts entities into the Neo4j "
        "knowledge graph.",
        "ingestion.md",
    ),
]


async def _seed(client: QdrantClient) -> None:
    """Write production-shaped points: both named vectors, real embeddings.

    Deliberately not constant synthetic vectors. The dense vector has to be
    the width the configured embedder actually emits (or Qdrant rejects the
    point), and the sparse vector has to come from the same miniCOIL model the
    query side encodes with (or the lexical channel matches nothing). A test
    that seeds vectors the pipeline cannot match would still see an empty
    `degraded` list, and would assert nothing at all.
    """
    from ingestion.embedder import create_embedder
    from ingestion.sparse_embedder import encode_documents

    texts = [text for text, _ in _DOCS]

    embedder = create_embedder()
    try:
        dense = await embedder.embed_text(texts, task="passage")
    finally:
        await embedder.close()
    sparse = encode_documents(texts)

    client.upsert(
        collection_name=DENSE_COLLECTION,
        points=[
            models.PointStruct(
                id=index + 1,
                vector={
                    DENSE_VECTOR_NAME: dense[index],
                    SPARSE_VECTOR_NAME: sparse[index],
                },
                payload={
                    "source_file": source_file,
                    "content_type": "text_chunk",
                    "page_number": 1,
                    "chunk_index": 0,
                    "text_content": text,
                    "metadata": {"source_key": source_file},
                },
            )
            for index, (text, source_file) in enumerate(_DOCS)
        ],
        wait=True,
    )


async def test_default_path_does_not_degrade(qdrant_client: QdrantClient) -> None:
    """A healthy deployment answers a plain query with no degradation."""
    await _seed(qdrant_client)

    response = await multi_search(QUERY, limit=5)

    assert "degraded" not in response, (
        f"The default retrieval path degraded: {response['degraded']}. Every "
        "stage failure is swallowed into this list, so a name here is a broken "
        "channel, a broken reranker or a broken graph engine — not a tuning "
        "problem. Check the credentials and services for the named stage."
    )
    # Only set when dense AND sparse both failed; degraded would have caught
    # that already, but the key carries the louder message.
    assert "error" not in response, response["error"]

    assert response["results"], (
        "No results for a query that directly matches a seeded document. "
        "Retrieval reported no degradation, so the channels ran — meaning the "
        "seeded vectors and the query vectors are in different spaces."
    )

    # server/tools/graph_search.py catches its own exceptions and returns an
    # error dict rather than raising, so multi_search's run_graph never sees a
    # failure and "graph" cannot reach `degraded`. Assert on the payload
    # instead, or a dead graph engine passes this test unnoticed.
    graph_errors = [fact for fact in response["graph_facts"] if "error" in fact]
    assert not graph_errors, f"graph_search reported an error: {graph_errors}"
