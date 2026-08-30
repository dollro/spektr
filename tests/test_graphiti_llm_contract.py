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
