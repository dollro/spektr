"""Reranker protocol adapters, one per wire contract.

Mirrors `ingestion/embedders/`: this package holds implementations, while the
`Reranker` protocol and the `create_reranker()` factory live one level up in
`retrieval/rerank.py` — the same split as `ingestion/embedder.py` versus
`ingestion/embedders/`.

Adapters are keyed by *protocol*, not vendor, because rerank APIs did not
converge the way `/v1/embeddings` did. `config/rerank_models.py` maps a model
onto the protocol that serves it.
"""

from __future__ import annotations
