from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Jina v4
    jina_api_key: str
    jina_model: str = "jina-clip-v4"
    jina_dense_dimensions: int = 2048

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
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key: str | None = None

    # MCP
    mcp_transport: str = "sse"
    mcp_port: int = 8080
    mcp_api_key: str = ""

    # Resilience
    jina_max_concurrent: int = 5
    extraction_timeout: int = 30
    tool_timeout: int = 30
    max_retries: int = 3
    rerank_enabled: bool = False
    vlm_generation_enabled: bool = False

    # Observability
    log_level: str = "INFO"
    log_format: str = "json"


settings = Settings()
