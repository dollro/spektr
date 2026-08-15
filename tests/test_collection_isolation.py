"""Guard: the test suite must never touch the real Qdrant collections.

The integration fixtures drop and recreate collections wholesale. If they ever
resolve to the production names, running the suite silently destroys a
developer's ingested corpus *and* any Path B live-session points sharing
``documents_dense``. These assertions are the canary for that.
"""

from __future__ import annotations

from config.constants import DENSE_COLLECTION, MULTIVEC_COLLECTION

PROD_DENSE = "documents_dense"
PROD_MULTIVEC = "documents_multivec"


def test_dense_collection_is_not_the_production_one() -> None:
    assert DENSE_COLLECTION != PROD_DENSE, (
        "Test session resolved DENSE_COLLECTION to the production collection. "
        "tests/conftest.py must override QDRANT_DENSE_COLLECTION before "
        "config.constants is imported."
    )


def test_multivec_collection_is_not_the_production_one() -> None:
    assert MULTIVEC_COLLECTION != PROD_MULTIVEC, (
        "Test session resolved MULTIVEC_COLLECTION to the production collection. "
        "tests/conftest.py must override QDRANT_MULTIVEC_COLLECTION before "
        "config.constants is imported."
    )


def test_production_defaults_are_unchanged(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Without the overrides, the names must still be the production ones."""
    import importlib

    import config.constants as constants

    monkeypatch.delenv("QDRANT_DENSE_COLLECTION", raising=False)
    monkeypatch.delenv("QDRANT_MULTIVEC_COLLECTION", raising=False)
    reloaded = importlib.reload(constants)
    try:
        assert reloaded.DENSE_COLLECTION == PROD_DENSE
        assert reloaded.MULTIVEC_COLLECTION == PROD_MULTIVEC
    finally:
        importlib.reload(constants)
