"""Shared HTTP concerns for reranker adapters.

The embedder clients each carry their own copy of `_is_retryable`
(`ingestion/embedders/jina.py:26`, `openrouter.py:52`, `voyage.py:25`) because
they were written at different times and diverged — voyage's still omits the
transient-network branch the other two have. This package is new, so the
predicate lives in one place from the start.
"""

from __future__ import annotations

import httpx


def is_retryable(exc: BaseException) -> bool:
    """Return True for HTTP 429/5xx and transient network errors.

    The reranker this replaces retried on bare `httpx.HTTPStatusError`
    (`retrieval/rerank.py:42` before this change), which burned all
    `MAX_RETRIES` attempts on a 401 or a 400 — errors that cannot succeed on
    a retry. A missing API key therefore cost three round trips and three
    exponential backoff sleeps before degrading.

    `httpx.TimeoutException` is a SIBLING of `NetworkError`, not a subclass of
    `ConnectError`, so it must be listed explicitly — omitting it is what made
    a plain connect timeout non-retryable in the embedder clients.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ReadError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ),
    )


def raise_for_rerank_status(resp: httpx.Response, provider: str) -> None:
    """Raise a labelled `HTTPStatusError` for any non-200 response.

    Separated from the retry predicate so the two cannot drift: this decides
    *whether* something failed, `is_retryable` decides whether to try again.
    """
    if resp.status_code == 200:
        return
    raise httpx.HTTPStatusError(
        f"{provider} rerank error ({resp.status_code}): {resp.text}",
        request=resp.request,
        response=resp,
    )
