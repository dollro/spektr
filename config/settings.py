from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Jina v4
    jina_api_key: str = ""
    jina_model: str = "jina-clip-v4"
    jina_dense_dimensions: int = 2048

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_dense_collection: str = "documents_dense"
    qdrant_multivec_collection: str = "documents_multivec"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # PostgreSQL
    database_url: str = (
        "postgresql://cocoindex:cocoindex@localhost:5432/cocoindex"
    )

    # AWS
    s3_bucket_name: str = ""
    s3_sqs_queue_url: str = ""
    aws_region: str = "us-east-1"

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key: str = ""

    # MCP
    mcp_transport: str = "sse"
    mcp_port: int = 8000


settings = Settings()
