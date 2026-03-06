from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Jina v4
    jina_api_key: str
    jina_api_url: str = "https://api.jina.ai"
    jina_model: str = "jina-embeddings-v4"
    jina_dense_dimensions: int = 2048
    jina_rpm: int = 500  # requests per minute (free=500, tier1=500, tier2=5000)
    jina_tpm: int = 100_000  # tokens per minute (free=100k, tier1=10M, tier2=100M)
    jina_batch_size: int = 10  # max texts per embed_text API call

    # Embedding provider selection
    embedding_provider: str = "jina"  # "jina" | "voyage"

    # Voyage (only needed when embedding_provider = "voyage")
    voyage_api_key: str = ""
    voyage_api_url: str = "https://api.voyageai.com"
    voyage_text_model: str = "voyage-4-large"
    voyage_multimodal_model: str = "voyage-multimodal-3.5"
    voyage_dense_dimensions: int = 1024
    voyage_rpm: int = 300
    voyage_max_concurrent: int = 10

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

    # Document Source
    document_source: str = "local"  # "local" or "s3"
    local_documents_path: str = "documents"

    # AWS (only used when document_source=s3)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    aws_endpoint_url: str = ""
    s3_bucket_name: str = ""
    s3_sqs_queue_url: str = ""

    # LLM
    llm_api_type: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key: str | None = None
    llm_base_url: str = ""

    # MCP
    mcp_transport: str = "sse"
    mcp_port: int = 8080
    mcp_api_key: str = ""

    # Resilience
    pipeline_timeout: int = 3600  # per-file timeout in seconds (default 1h)
    graphiti_concurrency: int = 3  # max concurrent Graphiti episode ingestions
    jina_max_concurrent: int = 5
    extraction_timeout: int = 30
    tool_timeout: int = 30
    max_retries: int = 3
    rerank_enabled: bool = False
    vlm_generation_enabled: bool = False
    multivec_enabled: bool = False
    graph_enabled: bool = True
    graph_semaphore_limit: int = 10  # SEMAPHORE_LIMIT for Graphiti LLM concurrency
    graph_episode_target_size: int = 1500  # Target chars per Graphiti episode
    graph_engine: str = "graphiti"  # "graphiti" | "gliner"
    image_embed_strategy: str = "smart"  # "smart" | "all"

    # Observability
    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def dense_dimensions(self) -> int:
        """Return dense vector dimensions for the active embedding provider."""
        if self.embedding_provider == "voyage":
            return self.voyage_dense_dimensions
        return self.jina_dense_dimensions

    @model_validator(mode="after")
    def _validate_provider_features(self) -> Self:
        if self.embedding_provider == "voyage" and self.multivec_enabled:
            msg = (
                "Voyage does not support ColBERT multi-vector embeddings. "
                "Set multivec_enabled=False or use embedding_provider=jina."
            )
            raise ValueError(msg)
        return self


settings = Settings()
