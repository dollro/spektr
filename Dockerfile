# syntax=docker/dockerfile:1.7

FROM python:3.13-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app

# --extra gliner: GRAPH_ENGINE is a runtime setting, so the image must carry
# both engines or `GRAPH_ENGINE=gliner` fails at first graph-engine use — at
# ingest time, not build time, because the gliner2 import is lazy. The extra
# adds only gliner/gliner2/sentencepiece; torch is already a base dep via
# docling. Omitting it here is the same trap as a bare `uv sync` locally.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra gliner

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra gliner


FROM python:3.13-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -u 1000 -m spektr

WORKDIR /app

COPY --from=builder --chown=spektr:spektr /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /app/state /app/documents /app/backups \
    && chown spektr:spektr /app/state /app/documents /app/backups

USER spektr

EXPOSE 8000 8001 8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "server.mcp_server"]
