from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.constants import DENSE_COLLECTION, MULTIVEC_COLLECTION
from config.embedding_models import EmbeddingModel, EmbeddingRoute, spec


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

    # CocoIndex internal state (v1 stores the target-state ledger, memoization
    # cache and component tree in a local LMDB directory — no PostgreSQL).
    # Lives under state/ so the existing `ingest_state` volume covers it.
    cocoindex_db_path: str = "state/cocoindex.db"

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
    sharepoint_state_dir: str = "state/sharepoint"

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
    rerank_candidates: int = 50  # fused candidates sent to the reranker
    rerank_score_floor: float = 0.0  # v3.5 scores are unbounded; <0 means judged irrelevant
    retry_enabled: bool = True
    retry_limit_multiplier: int = 3  # candidate-pool widening on gated retry
    decompose_enabled: bool = True
    decompose_model: str = ""  # empty -> fall back to llm_model
    decompose_max_subqueries: int = 4

    vlm_generation_enabled: bool = False
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
        """Wire-level model id for the active model+route pair."""
        s = spec(self.embedding_model)
        if self.embedding_route == "openrouter":
            return s.openrouter_text_id
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
        return self.voyage_multimodal_model

    @property
    def supports_image_embedding(self) -> bool:
        """True when embed_image is implemented for this model+route."""
        return self.embedding_route in spec(self.embedding_model).image_routes

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
        if self.multivec_enabled and self.embedding_route not in s.multivector_routes:
            msg = (
                f"ColBERT multi-vector is not available for "
                f"{self.embedding_model} via {self.embedding_route}. "
                "Set MULTIVEC_ENABLED=false, or use EMBEDDING_MODEL=jina-v4 "
                "with EMBEDDING_ROUTE=native."
            )
            raise ValueError(msg)
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


settings = Settings()  # type: ignore[call-arg]  # required fields supplied via .env
