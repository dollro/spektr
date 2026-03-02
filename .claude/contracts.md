# Integration Contracts — Spektr

These contracts define the exact interfaces between modules. Teammates MUST implement these signatures exactly. Deviation requires lead approval.

---

## Contract 1: Config Module → All Modules

**Producer:** foundation agent
**Consumers:** all agents

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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
    database_url: str

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
```

```python
# config/constants.py
DENSE_COLLECTION = "documents_dense"
MULTIVEC_COLLECTION = "documents_multivec"
DENSE_DIM = 2048
MULTIVEC_DIM = 128
ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "PRODUCT", "TECHNOLOGY",
    "LOCATION", "CONCEPT", "EVENT",
]
RELATIONSHIP_TYPES = [
    "WORKS_AT", "PARTNERS_WITH", "PRODUCES", "USES_TECHNOLOGY",
    "LOCATED_IN", "ACQUIRED", "COMPETES_WITH", "REFERENCES",
]
```

---

## Contract 2: JinaV4Embedder → CocoIndex Ops, MCP Tools

**Producer:** vector-layer agent (task 1.6)
**Consumers:** vector-layer (task 2.4), MCP tools (Phase 3)

```python
# ingestion/embedder.py
class JinaV4Embedder:
    """Shared httpx.AsyncClient, initialized once."""

    def __init__(self, api_key: str | None = None) -> None: ...
    async def close(self) -> None: ...

    async def embed_text(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
        dimensions: int = 2048,
    ) -> list[list[float]]:
        """Batch text → list of dense vectors (2048d each)."""

    async def embed_text_query(
        self, query: str, dimensions: int = 2048
    ) -> list[float]:
        """Single query text → dense vector (2048d). Uses retrieval.query LoRA."""

    async def embed_image(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[float]:
        """Single image → dense vector (2048d)."""

    async def embed_multi_vector(
        self, image_bytes: bytes, media_type: str = "image/png"
    ) -> list[list[float]]:
        """Single image → list of ColBERT token vectors (128d each)."""

    async def embed_query_multi_vector(
        self, query: str
    ) -> list[list[float]]:
        """Single query text → list of ColBERT token vectors (128d each)."""
```

**API endpoint:** `https://api.jina.ai/v1/embeddings`
**Headers:** `Authorization: Bearer {api_key}`, `Content-Type: application/json`

---

## Contract 3: File Processor → Pipeline

**Producer:** vector-layer agent (task 2.1)
**Consumer:** pipeline agent (task 2.5)

```python
# ingestion/file_processor.py
from dataclasses import dataclass

@dataclass
class Page:
    image_bytes: bytes       # PNG bytes for image/PDF pages, empty for text
    text: str                # text content for text pages, empty for image
    page_number: int
    content_type: str        # "pdf" | "image" | "text"

@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page_number: int

def file_to_pages(filename: str, content: bytes) -> list[Page]:
    """MIME-classify file, convert to list of Pages.
    PDF → multiple Pages with PNG bytes (300 DPI).
    Image → single Page with original bytes.
    Text → single Page with text content.
    Unknown → empty list + log warning.
    """

def semantic_chunk(text: str, max_chunk_size: int = 512) -> list[TextChunk]:
    """Split text into chunks preserving paragraph boundaries."""
```

---

## Contract 4: Entity Extractor → Graph Writer

**Producer:** graph-layer agent (task 2.2)
**Consumer:** graph-layer agent (task 2.3), pipeline agent (task 2.5)

```python
# ingestion/entity_extractor.py
from pydantic import BaseModel
from typing import Any, Literal
from config.constants import ENTITY_TYPES, RELATIONSHIP_TYPES

EntityType = Literal["PERSON", "ORGANIZATION", "PRODUCT", "TECHNOLOGY",
                     "LOCATION", "CONCEPT", "EVENT"]
RelationType = Literal["WORKS_AT", "PARTNERS_WITH", "PRODUCES",
                       "USES_TECHNOLOGY", "LOCATED_IN", "ACQUIRED",
                       "COMPETES_WITH", "REFERENCES"]

class Entity(BaseModel):
    name: str
    type: EntityType
    description: str

class Relationship(BaseModel):
    source: str          # entity name
    target: str          # entity name
    relation: RelationType
    properties: dict[str, Any] = {}

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]

async def extract_entities(text: str, llm_client: "LLMClient") -> ExtractionResult:
    """Extract entities + relationships from text via LLM.
    Retry once on parse failure. Return empty result after max retries.
    Normalize entity names: strip whitespace, title case.
    """
```

---

## Contract 5: Graph Writer → Pipeline

**Producer:** graph-layer agent (task 2.3)
**Consumer:** pipeline agent (task 2.5)

```python
# ingestion/graph_writer.py
class GraphWriter:
    def __init__(self, uri: str, user: str, password: str) -> None: ...
    async def close(self) -> None: ...

    async def upsert_document(self, s3_key: str, **properties) -> None:
        """MERGE Document node on s3_key."""

    async def upsert_chunk(
        self, chunk_id: str, text_preview: str,
        page_number: int, s3_key: str
    ) -> None:
        """MERGE Chunk node, create HAS_CHUNK rel to Document."""

    async def upsert_entity(self, entity: Entity) -> None:
        """MERGE Entity on (name, type), set description + timestamps."""

    async def upsert_relationship(
        self, source: str, target: str,
        relation: str, properties: dict
    ) -> None:
        """APOC merge relationship with dynamic type."""

    async def link_chunk_to_entity(
        self, chunk_id: str, entity_name: str, confidence: float
    ) -> None:
        """MERGE MENTIONS relationship between Chunk and Entity."""

    async def write_extraction_result(
        self, s3_key: str, chunk_id: str,
        extraction_result: ExtractionResult
    ) -> None:
        """Orchestrate all upserts for one chunk's entities."""
```

---

## Contract 6: Qdrant Setup → Pipeline, MCP Tools

**Producer:** vector-layer agent (task 1.4)
**Consumer:** pipeline agent, MCP tools

```python
# ingestion/qdrant_setup.py
from qdrant_client import QdrantClient

async def create_dense_collection(client: QdrantClient) -> None:
    """Create documents_dense: size=2048, distance=COSINE.
    Payload indexes: source_file (KEYWORD), content_type (KEYWORD).
    Idempotent — skip if exists.
    """

async def create_multivec_collection(client: QdrantClient) -> None:
    """Create documents_multivec: named vector 'colbert', size=128,
    distance=COSINE, multivector_config=MaxSim.
    Payload index: source_file (KEYWORD).
    Idempotent — skip if exists.
    """

async def ensure_collections(client: QdrantClient) -> None:
    """Call both creation functions."""
```

---

## Contract 7: Neo4j Setup → Graph Writer, MCP Tools

**Producer:** graph-layer agent (task 1.5)
**Consumer:** graph-layer (task 2.3), MCP tools

```python
# ingestion/neo4j_setup.py
from neo4j import AsyncDriver

async def create_neo4j_schema(driver: AsyncDriver) -> None:
    """Create constraints (IF NOT EXISTS):
    - Document.s3_key uniqueness
    - Entity(name, type) composite uniqueness
    - Chunk.id uniqueness
    Verify APOC: RETURN apoc.version()
    """
```

---

## Contract 8: LLM Client → Entity Extractor

**Producer:** graph-layer agent (task 2.2)
**Consumer:** internal to entity_extractor

```python
# Simple LLM client abstraction
from typing import Protocol

class LLMClient(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> str:
        """Send messages to LLM, return response text."""

# Implementations: AnthropicClient, OpenAIClient
# Selected by settings.llm_provider
```
