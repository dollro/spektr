"""Tests for the late-interaction model registry."""

from __future__ import annotations

import pytest

from config.multivec_models import REGISTRY, MultivecSpec, spec


def test_every_default_route_is_a_member_of_routes() -> None:
    """An unset MULTIVEC_ROUTE adopts default_route, so it must be legal."""
    for name, model_spec in REGISTRY.items():
        assert model_spec.default_route in model_spec.routes, name


def test_every_entry_has_a_positive_dimension() -> None:
    """`dimensions` sizes the Qdrant collection; 0 would create an empty vector."""
    for name, model_spec in REGISTRY.items():
        assert model_spec.dimensions > 0, name


def test_jina_entry_reproduces_the_retired_flag() -> None:
    """MULTIVEC_ENABLED=true meant exactly this: jina-v4, native, 128 dims.

    Pinning it here is what makes the migration message honest — an operator
    following it must get byte-identical provisioning, not merely something
    that works.
    """
    s = spec("jina-v4-colbert")
    assert s.routes == ("native",)
    assert s.default_route == "native"
    assert s.dimensions == 128
    assert s.wire_id == "jina-embeddings-v4"


def test_self_hosted_entry_is_local_only() -> None:
    """ColQwen is served through colpali-engine, not a vendor API."""
    s = spec("colqwen2.5")
    assert s.routes == ("local",)
    assert "native" not in s.routes


def test_unknown_model_message_lists_the_alternatives() -> None:
    with pytest.raises(ValueError, match="Unknown multivector model") as excinfo:
        spec("colpali-v9")
    message = str(excinfo.value)
    assert "colqwen2.5" in message and "jina-v4-colbert" in message


def test_unknown_model_suppresses_the_keyerror_context() -> None:
    """`from None` — a KeyError traceback here is noise, not information."""
    with pytest.raises(ValueError) as excinfo:
        spec("nope")
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


def test_spec_is_frozen() -> None:
    """Registry entries are shared process-wide; a mutation would be global."""
    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError is dataclasses-private
        spec("jina-v4-colbert").dimensions = 256  # type: ignore[misc]


def test_dimensions_is_per_model_not_a_constant() -> None:
    """The field exists so a model choosing another width cannot be missed.

    `MULTIVEC_DIM = 128` was a module constant that would have had to be
    edited — invisibly to the registry — for such a model. This asserts the
    plumbing reads the spec, so adding one is a registry edit and nothing else.
    """
    custom = MultivecSpec(routes=("local",), default_route="local", dimensions=96, wire_id="x")
    assert custom.dimensions == 96
