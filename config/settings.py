from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Jina v4 (only needed when embedding_provider = "jina")
    jina_api_key: str = ""
    jina_api_url: str = "https://api.jina.ai"
    jina_model: str = "jina-embeddings-v4"
    jina_dense_dimensions: int = 2048
    jina_rpm: int = 500  # requests per minute (free=500, tier1=500, tier2=5000)
    jina_tpm: int = 100_000  # tokens per minute (free=100k, tier1=10M, tier2=100M)
    jina_batch_size: int = 10  # max texts per embed_text API call

    # Embedding provider selection
    embedding_provider: str = "jina"  # "jina" | "voyage" | "openrouter"

    # Voyage (only needed when embedding_provider = "voyage")
    voyage_api_key: str = ""
    voyage_api_url: str = "https://api.voyageai.com"
    voyage_text_model: str = "voyage-4-large"
    voyage_multimodal_model: str = "voyage-multimodal-3.5"
    voyage_dense_dimensions: int = 1024
    voyage_rpm: int = 300
    voyage_max_concurrent: int = 10

    # OpenRouter (only needed when embedding_provider = "openrouter")
    # Default targets Gemini Embedding 2 (text-only, 3072d, MRL truncation).
    # Any OpenRouter-served embedding model can be selected here.
    openrouter_api_key: str = ""
    openrouter_api_url: str = "https://openrouter.ai/api"
    openrouter_model: str = "google/gemini-embedding-2-preview"
    openrouter_dense_dimensions: int = 3072
    openrouter_rpm: int = 300
    openrouter_max_concurrent: int = 10
    openrouter_http_referer: str = ""  # optional, for openrouter.ai rankings
    openrouter_x_title: str = ""  # optional, for openrouter.ai rankings

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_dense_collection: str = "documents_dense"
    qdrant_multivec_collection: str = "documents_multivec"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # PostgreSQL
    database_url: str = "postgresql://cocoindex:cocoindex@localhost:5432/cocoindex"

    # Document Source — exactly one of: local, s3, sharepoint
    document_source: Literal["local", "s3", "sharepoint"] = "local"
    local_documents_path: str = "documents"

    # AWS (only used when document_source=s3)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    aws_endpoint_url: str = ""
    s3_bucket_name: str = ""
    s3_sqs_queue_url: str = ""

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
    llm_model: str = "claude-sonnet-4-20250514"
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
    graphiti_concurrency: int = 3  # max concurrent Graphiti episode ingestions
    jina_max_concurrent: int = 5
    extraction_timeout: int = 30
    tool_timeout: int = 30
    max_retries: int = 3
    rerank_enabled: bool = True
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
    schema_induction_model: str = "claude-haiku-4-5-20251001"
    schema_cache_ttl: int = 3600  # seconds
    image_embed_strategy: str = "smart"  # "smart" | "all"

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
        """Return dense vector dimensions for the active embedding provider."""
        if self.embedding_provider == "voyage":
            return self.voyage_dense_dimensions
        if self.embedding_provider == "openrouter":
            return self.openrouter_dense_dimensions
        return self.jina_dense_dimensions

    @model_validator(mode="after")
    def _validate_provider_features(self) -> Self:
        if self.embedding_provider == "voyage" and self.multivec_enabled:
            msg = (
                "Voyage does not support ColBERT multi-vector embeddings. "
                "Set multivec_enabled=False or use embedding_provider=jina."
            )
            raise ValueError(msg)
        if self.embedding_provider == "openrouter" and self.multivec_enabled:
            msg = (
                "OpenRouter does not support ColBERT multi-vector embeddings. "
                "Set multivec_enabled=False or use embedding_provider=jina."
            )
            raise ValueError(msg)
        if self.document_source == "s3" and not (
            self.s3_bucket_name and self.s3_sqs_queue_url
        ):
            msg = (
                "DOCUMENT_SOURCE=s3 requires both S3_BUCKET_NAME and "
                "S3_SQS_QUEUE_URL to be set."
            )
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
