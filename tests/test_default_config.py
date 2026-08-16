"""The shipped `.env.example` is a product artifact, so it gets tests.

Every fresh deployment starts with `cp .env.example .env`. Until this module
existed, nothing in the suite ever constructed `Settings` from that file, so
a key `Settings` needs but the example never mentions — or a validator the
example violates — shipped silently and surfaced as a broken first boot.

**Env isolation is load-bearing here.** pydantic-settings ranks `os.environ`
ABOVE the dotenv file, and `tests/conftest.py` calls `dotenv.load_dotenv()`
at import time, which puts the developer's own `.env` into `os.environ`.
Without neutralising that, the local machine would quietly supply everything
the example omits and these tests would pass for the wrong reason. Hence
`_settings_from_env_example`, which deletes every `os.environ` name that maps
to a `Settings` field before loading, and the provenance assertions in
`test_env_example_loads` that prove the file — not the machine — was the
source.
"""

from __future__ import annotations

import os
import pathlib
import re

import dotenv
import pytest

from config.settings import Settings

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_EXAMPLE = REPO_ROOT / ".env.example"

# A key *declaration*, not a mention. Anchored at the start of the line so a
# key named inside prose does not count: the RERANK_ENABLED block says
# "Rescore fused candidates with RERANK_MODEL." in a trailing comment, and a
# substring match would let the single most conspicuous gap pass. A leading
# `#` is allowed because genuinely optional keys ship commented out (HF_HOME,
# COCOINDEX_LMDB_MAP_SIZE, the QDRANT_*_COLLECTION overrides).
_KEY_DECLARATION = re.compile(r"^[ \t]*#?[ \t]*([A-Z][A-Z0-9_]*)=")

# Fields declared on Settings *solely* so a stale value fails loudly — see
# `_validate_embedding_selection`, which raises a migration error the moment
# EMBEDDING_PROVIDER is non-empty. Shipping them in .env.example would hand
# every fresh deployment the very error they exist to raise, so their absence
# is the correct state. This set is permanent; it is NOT the Phase 4 backlog.
RETIRED_KEYS = frozenset({"EMBEDDING_PROVIDER", "MULTIVEC_ENABLED"})

# Settings fields `.env.example` does not declare today. Determined
# empirically against the current tree, not from the plan's prose.
#
# Phase 4 of plans/chore-architecture-rewrite/2026-08-16-capability-slots-plan.md
# ("Env trio sync") adds these to .env, .env.example and .env.prod and EMPTIES
# this set. Until then, every entry is a key a fresh deployment cannot
# discover from the shipped example — the whole retrieval pipeline's
# configuration surface is invisible.
KNOWN_MISSING = frozenset(
    {
        "DECOMPOSE_ENABLED",
        "DECOMPOSE_MAX_SUBQUERIES",
        "DECOMPOSE_MODEL",
        "LIVE_INGEST_PORT",
        "MCP_SERVER_URL",
        "RETRY_ENABLED",
        "RETRY_LIMIT_MULTIPLIER",
        "RRF_K",
        "SCHEMA_CACHE_TTL",
        "SCHEMA_INDUCTION_ENABLED",
        "SCHEMA_INDUCTION_MODEL",
        "SPARSE_ENABLED",
        "SPARSE_MODEL",
    }
)


def _declared_keys() -> set[str]:
    """Keys `.env.example` declares, including deliberately commented-out ones."""
    return {
        match.group(1)
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if (match := _KEY_DECLARATION.match(line))
    }


def _settings_from_env_example(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Load `.env.example` with the ambient environment neutralised.

    Deletes every `os.environ` name that resolves to a `Settings` field, in
    any case (pydantic-settings is case-insensitive by default, so it would
    otherwise pick up a lowercase name too), leaving the dotenv source as the
    only one that can supply a value. Init kwargs are the sole
    higher-priority source and none are passed.
    """
    for name in list(os.environ):
        if name.lower() in Settings.model_fields:
            monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]


def test_env_example_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped example config constructs Settings without raising.

    This is the whole point of the module: `cp .env.example .env` must
    produce a bootable deployment. Every cross-field validator on `Settings`
    is a way for the example to become unbootable without anyone noticing —
    the D2 graphiti/LLM rule caught the example red-handed the moment this
    test first ran — so this asserts the shipped values, not the comments
    describing them.
    """
    s = _settings_from_env_example(monkeypatch)

    # Provenance: prove the file supplied these, not the developer's .env
    # leaking in through os.environ. Without this, an incomplete cleanup
    # would turn the construction above into a test of the local machine —
    # and a machine that boots is exactly the case that hides an example
    # that does not.
    values = dotenv.dotenv_values(ENV_EXAMPLE)
    assert s.embedding_model == values["EMBEDDING_MODEL"]
    assert s.image_embed_strategy == values["IMAGE_EMBED_STRATEGY"]
    assert s.neo4j_password == values["NEO4J_PASSWORD"]
    assert s.document_source == values["DOCUMENT_SOURCE"]
    assert s.log_level == values["LOG_LEVEL"]


def test_env_example_declares_every_settings_field() -> None:
    """Every Settings field is discoverable from the shipped example."""
    declared = _declared_keys()
    expected = {name.upper() for name in Settings.model_fields} - RETIRED_KEYS

    leaked = RETIRED_KEYS & declared
    assert not leaked, (
        f"{sorted(leaked)} is declared in .env.example but exists on Settings "
        "only to raise a migration error. A fresh deployment copying this file "
        "would fail to boot. Remove the key from .env.example."
    )

    stale = KNOWN_MISSING & declared
    assert not stale, (
        f"{sorted(stale)} is now declared in .env.example — drop it from "
        "KNOWN_MISSING so the allowlist keeps shrinking toward empty."
    )

    missing = expected - declared - KNOWN_MISSING
    assert not missing, (
        f"Settings fields with no key in .env.example: {sorted(missing)}. "
        "A deployment cannot configure what the example never mentions — add "
        "the keys (with comments) to .env, .env.example and .env.prod."
    )


def test_rerank_credentials_present_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The D1 regression test: reranking is either off, or it is credentialled.

    The shipped example used to turn `RERANK_ENABLED=true` on while selecting
    `EMBEDDING_MODEL=gemini-2`, which routes through OpenRouter and never
    needs a Jina key — so `JINA_API_KEY=` was left empty and the reranker
    authenticated against api.jina.ai with an empty bearer token. Every query
    on a fresh deployment degraded to fusion order, silently, because the
    pipeline reports a rerank failure rather than raising it.

    Asserted through `rerank_endpoint`, the property the adapters actually
    read, so this stays true however the slot resolves its credentials —
    reaching for `jina_api_key` here would re-pin the very coupling D1
    removed.

    Stated as a disjunction rather than "a key is set", because both
    resolutions are legitimate: an example that ships reranking off is as
    correct as one that ships a credential, and off is what the file now
    chooses (an example file cannot carry a real key). What is never correct
    is enabled plus empty, which is the state this pins.
    """
    s = _settings_from_env_example(monkeypatch)

    base_url, api_key = s.rerank_endpoint
    assert (not s.rerank_enabled) or (base_url and api_key), (
        f"RERANK_ENABLED=true but rerank_endpoint resolves to "
        f"{(base_url, bool(api_key))!r} from the shipped example config. "
        "Reranking would fail on every query and the pipeline would degrade "
        "to fusion order — ship a credential, or RERANK_ENABLED=false."
    )
