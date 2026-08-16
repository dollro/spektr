from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.constants import DENSE_COLLECTION, MULTIVEC_COLLECTION
from config.embedding_models import EmbeddingModel, EmbeddingRoute, spec
from config.multivec_models import MultivecRoute
from config.multivec_models import spec as multivec_spec
from config.rerank_models import RerankRoute
from config.rerank_models import spec as rerank_spec


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Jina v4 (only needed when embedding_model = "jina-v4")
    jina_api_key: str = ""
    jina_api_url: str = "https://api.jina.ai"
    jina_model: str = "jina-embeddings-v4"
    jina_rpm: int = 500  # requests per minute (free=500, tier1=500, tier2=5000)
    jina_tpm: int = 100_000  # tokens per minute (free=100k, tier1=10M, tier2=100M)
    jina_batch_size: int = 10  # max texts per embed_text API call

    # Embedding selection: WHICH model, and WHICH route serves it.
    # These are orthogonal — see config/embedding_models.py.
    embedding_model: EmbeddingModel = "gemini-2"
    # Empty -> the model's default_route. Changing EMBEDDING_MODEL alone is
    # therefore enough; pin this only to override the registry's choice.
    embedding_route: EmbeddingRoute | Literal[""] = ""
    # 0 -> the model's recommended_dimensions, which is the point of leaving
    # it at 0: the right size follows the model instead of being retuned by
    # hand on every switch.
    embedding_dimensions: int = 0
    # Removed in favour of the pair above. Declared so a stale value fails
    # loudly: `extra="ignore"` would otherwise drop it and silently boot
    # with the defaults, which on a prod box means embedding with the
    # wrong model against an index built by the right one.
    embedding_provider: str = ""

    # Voyage (only needed when embedding_model = "voyage-4")
    voyage_api_key: str = ""
    voyage_api_url: str = "https://api.voyageai.com"
    voyage_text_model: str = "voyage-4-large"
    voyage_multimodal_model: str = "voyage-multimodal-3.5"
    voyage_rpm: int = 300
    voyage_max_concurrent: int = 10

    # OpenRouter (only needed when embedding_route = "openrouter").
    # The model id is resolved from the registry, not configured here.
    openrouter_api_key: str = ""
    openrouter_api_url: str = "https://openrouter.ai/api"
    # Gemini rejects >100 inputs per embeddings call (BatchEmbedContentsRequest).
    openrouter_batch_size: int = 100
    openrouter_rpm: int = 300
    openrouter_max_concurrent: int = 10
    openrouter_http_referer: str = ""  # optional, for openrouter.ai rankings
    openrouter_x_title: str = ""  # optional, for openrouter.ai rankings

    # Self-hosted embedding endpoint (only when embedding_route = "local").
    # An OpenAI-compatible /v1/embeddings server: TEI, vLLM or Infinity.
    # This is the route that makes a sovereign deployment expressible — the
    # other three are all vendor APIs.
    local_embed_api_url: str = ""
    local_embed_api_key: str = ""  # usually unset; TEI is typically unauthenticated
    # No RPM bucket on this route: a local server has no request quota, it has
    # finite GPU memory. Concurrency is the real constraint, so it is the only
    # one bounded here.
    local_max_concurrent: int = 8
    local_batch_size: int = 32

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    # Defaults come from config.constants so the two never drift; both resolve
    # the same QDRANT_*_COLLECTION env vars.
    qdrant_dense_collection: str = DENSE_COLLECTION
    qdrant_multivec_collection: str = MULTIVEC_COLLECTION

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Root for every piece of mutable runtime state. ONE knob, because the
    # three paths below are otherwise independently overridable and it only
    # takes forgetting one to write into a tree you meant to leave alone.
    #
    # It defaults inside the repo (gitignored) because local dev runs the app
    # on the host via uv — `task up` starts only Qdrant and Neo4j. That is
    # convenient and wrong the moment the working tree is shared: a sandbox,
    # a CI checkout, or a second worktree all inherit the host's ledger and
    # can delete it. Point STATE_DIR outside the tree in those environments.
    #
    # In the containerised stack this is already covered — the `ingest_state`
    # NAMED volume is mounted at /app/state on every pipeline service. Keep it
    # named rather than a bind mount: LMDB is mmap-based, and Docker Desktop's
    # file-sharing layer on macOS/Windows does not give correct mmap
    # semantics, so a bind-mounted ledger can corrupt rather than merely be
    # slow.
    state_dir: str = "state"

    # CocoIndex internal state (v1 stores the target-state ledger, memoization
    # cache and component tree in a local LMDB directory — no PostgreSQL).
    # Empty -> {STATE_DIR}/cocoindex.db, resolved in place by a validator so
    # every consumer still reads a concrete path.
    cocoindex_db_path: str = ""
    # Empty -> {STATE_DIR}/ingestion_failures.db. Was a module constant in
    # `ingestion/_failure_tracker.py`, which made it the one state path
    # STATE_DIR could not move.
    failure_db_path: str = ""

    # Document Source — exactly one of: local, s3, sharepoint
    document_source: Literal["local", "s3", "sharepoint"] = "local"
    local_documents_path: str = "documents"

    # AWS (only used when document_source=s3)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    aws_endpoint_url: str = ""
    s3_bucket_name: str = ""
    s3_prefix: str = ""
    # Optional. CocoIndex v1 dropped the built-in S3 push trigger, so SQS is
    # used purely as a trigger for a catch-up run. Without a queue URL, live
    # mode falls back to interval-only sweeps.
    s3_sqs_queue_url: str = ""
    s3_sqs_debounce_seconds: float = 5.0  # coalesce an event burst into one run
    s3_full_scan_interval_hours: float = 24.0  # safety net for missed events

    # SharePoint (only used when all sharepoint_* required fields are set)
    sharepoint_tenant_id: str = ""
    sharepoint_client_id: str = ""
    sharepoint_client_secret: str = ""
    sharepoint_site_id: str = ""
    sharepoint_drive_id: str = ""
    sharepoint_root_folder_path: str = ""  # e.g. "/Engineering/Specs"
    sharepoint_local_subdir: str = "sharepoint"
    sharepoint_sync_interval_seconds: int = 180
    sharepoint_state_dir: str = ""  # empty -> {STATE_DIR}/sharepoint

    # LLM
    llm_api_type: str = "anthropic"
    llm_model: str = "claude-haiku-4.5"  # ingest cost scales with this
    llm_api_key: str | None = None
    llm_base_url: str = ""

    # MCP
    # "http" is streamable-http (modern default, recommended). "sse" is the
    # legacy long-lived-stream transport — keep for old clients. "stdio" is
    # for subprocess MCP clients.
    mcp_transport: Literal["http", "sse", "streamable-http", "stdio"] = "http"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8080
    mcp_path: str = "/mcp"  # URL path for http/sse transports
    mcp_api_key: str = ""
    # Connect URL the agent uses to reach the MCP server. Defaults to localhost
    # (built from mcp_port + mcp_path); set to the service hostname in
    # containerized deploys (e.g. http://mcp:8080/mcp under docker-compose).
    mcp_server_url: str = ""

    # Resilience
    pipeline_timeout: int = 3600  # per-file timeout in seconds (default 1h)
    pipeline_max_retries: int = 3  # failures per file before poison-pill swallow
    # CocoIndex v1 defaults max_inflight_components to 1024, which would fan out
    # far past the embedding providers' rate limits. Bound files-in-flight here.
    pipeline_max_concurrent_files: int = 4
    graphiti_concurrency: int = 3  # max concurrent Graphiti episode ingestions
    jina_max_concurrent: int = 5
    extraction_timeout: int = 30
    tool_timeout: int = 30
    max_retries: int = 3
    rerank_enabled: bool = True

    # Retrieval pipeline
    sparse_enabled: bool = True
    sparse_model: str = "Qdrant/minicoil-v1"
    rrf_k: int = 60
    rerank_model: str = "jina-reranker-v3.5"
    # Empty -> the model's default_route. Changing RERANK_MODEL alone is
    # therefore enough; pin this only to override the registry's choice.
    rerank_route: RerankRoute | Literal[""] = ""
    # Both empty -> resolved per route (see the rerank_endpoint property).
    # Kept separate from JINA_API_KEY so the reranker stops being hostage to
    # the embedding model's vendor: RERANK_ENABLED defaults true while
    # EMBEDDING_MODEL defaults to gemini-2, so a by-the-book deployment used
    # to 401 on every query and degrade silently.
    rerank_api_key: str = ""
    rerank_api_url: str = ""
    rerank_candidates: int = 50  # fused candidates sent to the reranker
    # None -> the model's default_score_floor. 0.0 CANNOT be the unset
    # sentinel: it is a legitimate floor on the logit scale (and is exactly
    # jina-reranker-v3.5's default), so a falsy check would make an explicit
    # 0.0 indistinguishable from "not set" — the bug the embedding registry
    # avoids only because 0 dimensions is meaningless.
    rerank_score_floor: float | None = None
    retry_enabled: bool = True
    retry_limit_multiplier: int = 3  # candidate-pool widening on gated retry
    decompose_enabled: bool = True
    decompose_model: str = ""  # empty -> fall back to llm_model
    decompose_max_subqueries: int = 4

    vlm_generation_enabled: bool = False
    # Late-interaction slot. Empty = unfilled = `visual_search` unavailable.
    # Independent of EMBEDDING_MODEL: this collection shares no vector space
    # with documents_dense, so the two choices were never actually coupled.
    multivec_model: str = ""
    multivec_route: MultivecRoute | Literal[""] = ""
    # Replaced by MULTIVEC_MODEL. Declared so a stale value fails loudly
    # rather than being dropped by `extra="ignore"` — the same reasoning as
    # embedding_provider above. A silently ignored `true` here would leave
    # visual_search dark on a deployment that thought it was enabled.
    multivec_enabled: bool = False
    graph_enabled: bool = True
    graph_semaphore_limit: int = 10  # SEMAPHORE_LIMIT for Graphiti LLM concurrency
    graph_episode_target_size: int = 1500  # Target chars per Graphiti episode
    graph_engine: Literal["graphiti", "gliner"] = "graphiti"

    # Live ingestion
    live_ingest_port: int = 8001
    ingest_api_key: str = ""

    # Schema induction
    schema_induction_enabled: bool = True
    schema_induction_model: str = ""  # empty -> fall back to llm_model
    schema_cache_ttl: int = 3600  # seconds
    image_embed_strategy: Literal["smart", "all", "none"] = "smart"
    # Longest-side cap applied before an image is embedded. `.env.example`
    # documented this key for a long time while no field existed to receive
    # it, so `extra="ignore"` swallowed it and nothing ever resized: pages
    # render at 150 DPI (ingestion/file_processor.py), so an A4 page reached
    # the embedding API at roughly 1240x1754. That is billed area on every
    # provider and, on Jina's tile-based meter, ~28k estimated tokens per
    # page against a default TPM of 100k. 0 disables the cap.
    image_embed_max_px: int = 400

    # Observability
    log_level: str = "INFO"
    log_format: str = "json"
    logfire_token: str = ""
    observability_local_only: bool = True
    service_name: str = "spektr"

    @property
    def sharepoint_enabled(self) -> bool:
        """True only when all required SharePoint fields are populated."""
        return all(
            (
                self.sharepoint_tenant_id,
                self.sharepoint_client_id,
                self.sharepoint_client_secret,
                self.sharepoint_site_id,
                self.sharepoint_drive_id,
                self.sharepoint_root_folder_path,
            )
        )

    @property
    def dense_dimensions(self) -> int:
        """Dense vector dimensions: explicit override, else the model's
        recommended size (not its full size — see ModelSpec)."""
        return self.embedding_dimensions or spec(self.embedding_model).recommended_dimensions

    @property
    def embedding_model_id(self) -> str:
        """Wire-level model id for the active model+route pair.

        The per-vendor dict at the end covers only the two models with a
        native client. Anything else must be handled by a branch above it —
        it raises `KeyError` rather than returning something plausible,
        because a wrong model id produces vectors from the wrong space
        rather than an error.
        """
        s = spec(self.embedding_model)
        if self.embedding_route == "openrouter":
            return s.openrouter_text_id
        if self.embedding_route == "local":
            # The checkpoint the TEI/vLLM container serves. Sent on the wire
            # by servers that accept a `model` field, and recorded in the
            # Qdrant payload's `embedder_model` either way — which is what
            # scripts/doctor.py reads to spot a mixed index.
            return s.local_model_id
        return {"jina-v4": self.jina_model, "voyage-4": self.voyage_text_model}[
            self.embedding_model
        ]

    @property
    def embedding_image_model_id(self) -> str:
        """Wire-level model id used for image input.

        Falls back to the text id: a natively multimodal model (gemini-2)
        embeds both through one id, while a split family (voyage-4) needs a
        dedicated one.
        """
        s = spec(self.embedding_model)
        if self.embedding_route == "openrouter":
            return s.openrouter_image_id or s.openrouter_text_id
        if self.embedding_route == "local":
            # No self-hosted model in the registry embeds images today, and
            # `supports_image_embedding` is False for all of them, so this is
            # unreachable via the validated paths. Return the text id rather
            # than a vendor field that has nothing to do with this route.
            return s.local_model_id
        return self.voyage_multimodal_model

    @property
    def supports_image_embedding(self) -> bool:
        """True when embed_image is implemented for this model+route."""
        return self.embedding_route in spec(self.embedding_model).image_routes

    @field_validator("rerank_score_floor", mode="before")
    @classmethod
    def _blank_floor_means_unset(cls, value: object) -> object:
        """Treat `RERANK_SCORE_FLOOR=` as unset rather than as a parse error.

        Every other "leave empty for the default" key in `.env` is a str or an
        int where `""`/`0` is a usable sentinel. This one is `float | None`,
        and pydantic rejects `""` as a float outright — so without this, the
        documented way to adopt the model's default fails at startup.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def multivec_slot_filled(self) -> bool:
        """True when a late-interaction model is configured.

        Replaces the `multivec_enabled` boolean. Read this rather than the
        model string so callers do not have to know that "" means off.
        """
        return bool(self.multivec_model)

    @property
    def multivec_dimensions(self) -> int:
        """Per-token vector width for `documents_multivec`.

        Sourced from the active model rather than the old `MULTIVEC_DIM`
        constant, which could not vary with the model it was sizing.
        """
        return multivec_spec(self.multivec_model).dimensions

    @property
    def resolved_score_floor(self) -> float:
        """Relevance-gate floor: explicit override, else the model's default.

        Read this rather than `rerank_score_floor` — the raw field is None
        when unset, and a floor is only interpretable alongside the
        `score_scale` of the model that produced the score.
        """
        if self.rerank_score_floor is not None:
            return self.rerank_score_floor
        return rerank_spec(self.rerank_model).default_score_floor

    @property
    def rerank_endpoint(self) -> tuple[str, str]:
        """(base_url, api_key) for the active rerank model+route pair.

        Falls back to the Jina credentials on the native route so existing
        `.env` files that only set `JINA_API_KEY` keep working — that was the
        sole source before this slot existed. The local route deliberately
        does NOT fall back: a self-hosted endpoint has no sensible default
        host, and silently pointing it at api.jina.ai would be worse than
        failing.
        """
        if self.rerank_route == "local":
            return self.rerank_api_url, self.rerank_api_key
        return (
            self.rerank_api_url or self.jina_api_url,
            self.rerank_api_key or self.jina_api_key,
        )

    @model_validator(mode="after")
    def _validate_embedding_selection(self) -> Self:
        if self.embedding_provider:
            msg = (
                "EMBEDDING_PROVIDER has been replaced by EMBEDDING_MODEL + "
                "EMBEDDING_ROUTE, which separate the model from the endpoint "
                "that serves it. Migrate: "
                "jina -> EMBEDDING_MODEL=jina-v4, EMBEDDING_ROUTE=native; "
                "voyage -> EMBEDDING_MODEL=voyage-4, EMBEDDING_ROUTE=native; "
                "openrouter -> EMBEDDING_MODEL=gemini-2, EMBEDDING_ROUTE=openrouter. "
                "Then remove EMBEDDING_PROVIDER."
            )
            raise ValueError(msg)
        s = spec(self.embedding_model)
        # Resolve the route before it is checked, so an unset value adopts
        # the model's default rather than failing as an illegal pair.
        if not self.embedding_route:
            self.embedding_route = s.default_route
        if self.embedding_dimensions:
            allowed = s.allowed_dimensions
            if allowed and self.embedding_dimensions not in allowed:
                msg = (
                    f"EMBEDDING_DIMENSIONS={self.embedding_dimensions} is not "
                    f"supported by {self.embedding_model}; it emits "
                    f"{', '.join(str(d) for d in allowed)}."
                )
                raise ValueError(msg)
            if not allowed and not 0 < self.embedding_dimensions <= s.default_dimensions:
                msg = (
                    f"EMBEDDING_DIMENSIONS={self.embedding_dimensions} is out of "
                    f"range for {self.embedding_model}: Matryoshka truncation "
                    f"allows 1..{s.default_dimensions}."
                )
                raise ValueError(msg)
        if self.embedding_route not in s.routes:
            msg = (
                f"EMBEDDING_MODEL={self.embedding_model} is not served via "
                f"EMBEDDING_ROUTE={self.embedding_route}. "
                f"Available routes: {', '.join(s.routes)}."
            )
            raise ValueError(msg)
        # The MULTIVEC_ENABLED cross-slot check that used to live here is
        # gone: it rejected multi-vector unless the DENSE model happened to
        # emit ColBERT, which made changing the text embedder silently remove
        # the visual_search tool. The slot validates its own route instead —
        # see _validate_multivec_selection.
        # Catches the silent-drop failure mode: `smart` asks for image
        # embeddings that the active client raises NotImplementedError for,
        # which the page-level try/except then swallows.
        if self.image_embed_strategy != "none" and not self.supports_image_embedding:
            msg = (
                f"IMAGE_EMBED_STRATEGY={self.image_embed_strategy} requires image "
                f"embedding, which is not implemented for {self.embedding_model} "
                f"via {self.embedding_route}. Set IMAGE_EMBED_STRATEGY=none."
            )
            raise ValueError(msg)
        if self.document_source == "s3" and not self.s3_bucket_name:
            # S3_SQS_QUEUE_URL is optional since the CocoIndex v1 migration:
            # SQS is now only a trigger for a catch-up run, and live mode
            # degrades to interval-only sweeps without it.
            msg = "DOCUMENT_SOURCE=s3 requires S3_BUCKET_NAME to be set."
            raise ValueError(msg)
        if self.document_source == "sharepoint" and not self.sharepoint_enabled:
            msg = (
                "DOCUMENT_SOURCE=sharepoint requires all SHAREPOINT_* fields to "
                "be set: TENANT_ID, CLIENT_ID, CLIENT_SECRET, SITE_ID, DRIVE_ID, "
                "ROOT_FOLDER_PATH."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _resolve_state_paths(self) -> Self:
        """Derive the three state paths from STATE_DIR unless pinned.

        Resolved in place — the same shape as `embedding_route` above — so
        consumers (`ingestion/app.py`, `scripts/backup.py`, the failure
        tracker) keep reading a concrete path and never have to know that
        STATE_DIR exists.

        An explicitly set path still wins, which is what keeps existing
        `.env` files with `COCOINDEX_DB_PATH=state/cocoindex.db` working
        unchanged.
        """
        root = Path(self.state_dir)
        if not self.cocoindex_db_path:
            self.cocoindex_db_path = str(root / "cocoindex.db")
        if not self.failure_db_path:
            self.failure_db_path = str(root / "ingestion_failures.db")
        if not self.sharepoint_state_dir:
            self.sharepoint_state_dir = str(root / "sharepoint")
        return self

    @model_validator(mode="after")
    def _validate_multivec_selection(self) -> Self:
        """Resolve the late-interaction route, and reject the retired flag."""
        if self.multivec_enabled:
            msg = (
                "MULTIVEC_ENABLED has been replaced by MULTIVEC_MODEL, which "
                "selects the late-interaction model independently of "
                "EMBEDDING_MODEL — the two were never actually coupled. "
                "Migrate: MULTIVEC_ENABLED=true -> "
                "MULTIVEC_MODEL=jina-v4-colbert (identical behaviour); "
                "MULTIVEC_ENABLED=false -> leave MULTIVEC_MODEL empty. "
                "Then remove MULTIVEC_ENABLED."
            )
            raise ValueError(msg)
        if not self.multivec_model:
            return self
        s = multivec_spec(self.multivec_model)
        if not self.multivec_route:
            self.multivec_route = s.default_route
        if self.multivec_route not in s.routes:
            msg = (
                f"MULTIVEC_MODEL={self.multivec_model} is not served via "
                f"MULTIVEC_ROUTE={self.multivec_route}. "
                f"Available routes: {', '.join(s.routes)}."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_rerank_selection(self) -> Self:
        """Resolve the rerank route and reject a slot that cannot be reached.

        The failure this exists for is silent, not loud. `retrieval/rerank.py`
        used to read `settings.jina_api_key` unconditionally, while
        `RERANK_ENABLED` defaults true and `EMBEDDING_MODEL` defaults to
        gemini-2 — so a deployment that followed `.env.example` sent every
        rerank call with an empty bearer token, caught the 401, appended
        "rerank" to `degraded`, and returned RRF-ordered results forever.
        Nothing raised, and the only symptom was a key in the response.
        """
        s = rerank_spec(self.rerank_model)
        # Resolve before checking, so an unset value adopts the model's
        # default rather than failing as an illegal pair. Done even when
        # reranking is off, so `settings.rerank_route` is never the empty
        # string at runtime — readers should not have to know whether the
        # feature flag happened to be set.
        if not self.rerank_route:
            self.rerank_route = s.default_route

        if not self.rerank_enabled:
            # Nothing is reached, so credentials need not resolve. The model
            # and route are still validated above rather than left to rot.
            return self

        if self.rerank_route not in s.routes:
            msg = (
                f"RERANK_MODEL={self.rerank_model} is not served via "
                f"RERANK_ROUTE={self.rerank_route}. "
                f"Available routes: {', '.join(s.routes)}."
            )
            raise ValueError(msg)

        # An explicit floor can still be on the wrong scale. The registry
        # default follows the model, but an operator who pinned 0.0 in the
        # jina era and later switched to a unit-scaled reranker keeps that
        # value — and a [0,1] score can never fall below 0.0, so the gate
        # goes dark exactly as if the default had carried over. Catching only
        # the unset case would leave the more likely mistake uncaught.
        if s.score_scale == "unit" and self.rerank_score_floor is not None:
            floor = self.rerank_score_floor
            if not 0.0 < floor < 1.0:
                msg = (
                    f"RERANK_SCORE_FLOOR={floor} is unreachable for "
                    f"{self.rerank_model}, which emits scores in [0, 1]: the "
                    "relevance gate would "
                    + ("never fire" if floor <= 0.0 else "always fire")
                    + ". Leave RERANK_SCORE_FLOOR empty to adopt the model's "
                    f"default of {s.default_score_floor}, or set a value "
                    "strictly between 0 and 1."
                )
                raise ValueError(msg)

        base_url, api_key = self.rerank_endpoint
        if not base_url:
            msg = (
                f"RERANK_ENABLED=true with RERANK_MODEL={self.rerank_model} via "
                f"{self.rerank_route} needs RERANK_API_URL — a self-hosted "
                "reranker has no default host. Set it, or RERANK_ENABLED=false."
            )
            raise ValueError(msg)
        # A local TEI container is normally unauthenticated, so an empty key
        # is legitimate there and only the hosted routes require one.
        if self.rerank_route != "local" and not api_key:
            msg = (
                f"RERANK_ENABLED=true with RERANK_MODEL={self.rerank_model} needs "
                "an API key: set RERANK_API_KEY (or JINA_API_KEY, which it falls "
                "back to). Reranking is independent of EMBEDDING_MODEL, so this "
                "is required even when embedding through another provider. "
                "Set RERANK_ENABLED=false to retrieve without reranking."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_graph_llm_selection(self) -> Self:
        """Reject a graph engine that cannot reach the configured LLM.

        `ingestion/graphiti_client.py` builds `OpenAIGenericClient` and
        `OpenAIRerankerClient` unconditionally — Graphiti speaks only the
        OpenAI wire format — and an empty base_url resolves to
        https://api.openai.com/v1 inside the OpenAI SDK. So `anthropic` plus
        an empty LLM_BASE_URL does not fall back to Anthropic; it ships an
        Anthropic key and a Claude model id to OpenAI, and every episode
        fails mid-ingest instead of at startup.
        """
        if not (self.graph_enabled and self.graph_engine == "graphiti"):
            return self
        if self.llm_api_type.lower() == "openai" or self.llm_base_url:
            return self
        msg = (
            f"GRAPH_ENGINE=graphiti with LLM_API_TYPE={self.llm_api_type} and an "
            "empty LLM_BASE_URL cannot work: Graphiti only speaks the OpenAI wire "
            f"format, so it would send LLM_MODEL={self.llm_model} to "
            f"https://api.openai.com/v1 carrying a key issued for {self.llm_api_type}. Either "
            "set LLM_BASE_URL to an OpenAI-compatible gateway serving that model "
            "(e.g. https://openrouter.ai/api/v1), or set GRAPH_ENGINE=gliner to "
            "extract entities locally on CPU instead."
        )
        raise ValueError(msg)


settings = Settings()  # type: ignore[call-arg]  # required fields supplied via .env
