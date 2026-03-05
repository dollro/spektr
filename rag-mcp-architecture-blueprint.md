# RAG-as-MCP-Server — Architecture Blueprint

**Stack:** Jina v4 (API) · Neo4j · Qdrant · CocoIndex · PostgreSQL · FastMCP · Pydantic AI · AWS S3/SQS

---

## 1. System Overview

### Decided Architecture

A hybrid GraphRAG + multimodal vector search system exposed as an MCP server that agents can use as a tool. Jina v4 serves as the **single embedding model** for both text and visual content. Neo4j provides entity-relationship graph search. Qdrant stores all vector embeddings with multi-vector (ColBERT-style) support.

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS S3 Bucket                            │
│   (PDFs, scanned docs, images, markdown, slides, etc.)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ S3 Event Notification
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AWS SQS Queue                              │
│   (file create / update / delete events)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Push-based, real-time
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CocoIndex Pipeline                            │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │ Classify  │──▶│ PDF → Images │──▶│  Jina v4 API         │    │
│  │ by MIME   │   │ (300 DPI)    │   │  (embed text+images) │    │
│  └──────────┘   └──────────────┘   └──────────┬───────────┘    │
│                                                │                │
│  ┌──────────────────────┐    ┌─────────────────┴──────────┐    │
│  │ Entity Extraction    │    │ Vectors → Qdrant            │    │
│  │ (LLM-based)          │    │ (single-vector + multi-vec) │    │
│  └──────────┬───────────┘    └────────────────────────────┘    │
│             │                                                   │
│             ▼                                                   │
│  ┌──────────────────────┐                                      │
│  │ Entities → Neo4j     │                                      │
│  │ (nodes + relations)  │                                      │
│  └──────────────────────┘                                      │
│                                                                 │
│  State tracked in PostgreSQL (incremental processing)           │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastMCP Server (SSE)                          │
│                                                                 │
│  Tools exposed:                                                 │
│  ┌────────────────┐ ┌────────────────┐ ┌─────────────────┐     │
│  │ vector_search  │ │ graph_search   │ │ hybrid_search   │     │
│  │ (Qdrant dense) │ │ (Neo4j Cypher) │ │ (vector + graph)│     │
│  └────────────────┘ └────────────────┘ └─────────────────┘     │
│  ┌────────────────┐ ┌────────────────┐                         │
│  │ visual_search  │ │ ingest_url     │                         │
│  │ (Qdrant multi) │ │ (on-demand)    │                         │
│  └────────────────┘ └────────────────┘                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ MCP Protocol (SSE / stdio)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agent (Pydantic AI)                           │
│  Decides which tool(s) to call based on query intent            │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Role | Why This Choice |
|-----------|------|-----------------|
| **Jina v4 API** | All embeddings (text + visual), single-vector and multi-vector modes | Unified embedding space, no modality gap, task-specific LoRA adapters, 32K context |
| **Qdrant** | Vector storage and retrieval | Native multi-vector/ColBERT support essential for visual document retrieval |
| **Neo4j** | Knowledge graph for entity relationships | Cypher queries for multi-hop reasoning, relationship-aware retrieval |
| **PostgreSQL** | CocoIndex state tracking only | Required by CocoIndex for incremental processing metadata |
| **CocoIndex** | Ingestion pipeline orchestration | Incremental processing, S3/SQS native support, declarative pipeline definition |
| **FastMCP** | MCP server exposing RAG tools | Standard MCP protocol, SSE transport for streaming |
| **Pydantic AI** | Agent framework | Clean tool integration, streaming support, provider-agnostic LLM usage |

---

## 2. Project Structure

```
rag-mcp-server/
├── docker-compose.yml          # Neo4j, Qdrant, PostgreSQL
├── .env.example                # All configuration
├── pyproject.toml              # Python dependencies
│
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py             # CocoIndex flow definition (S3 → embed → store)
│   ├── file_processor.py       # MIME classification, PDF→images, text chunking
│   ├── embedder.py             # Jina v4 API wrapper (single + multi-vector)
│   ├── entity_extractor.py     # LLM-based entity + relationship extraction
│   └── graph_writer.py         # Neo4j entity/relationship upsert
│
├── server/
│   ├── __init__.py
│   ├── mcp_server.py           # FastMCP server definition + tool registration
│   ├── tools/
│   │   ├── vector_search.py    # Qdrant dense vector search
│   │   ├── visual_search.py    # Qdrant multi-vector (ColBERT) search
│   │   ├── graph_search.py     # Neo4j Cypher search
│   │   └── hybrid_search.py    # Combined vector + graph
│   ├── providers.py            # LLM provider abstraction
│   └── models.py               # Pydantic data models
│
├── agent/
│   ├── __init__.py
│   ├── agent.py                # Pydantic AI agent with MCP tool bindings
│   └── api.py                  # FastAPI endpoint (optional, for direct access)
│
├── config/
│   ├── settings.py             # Pydantic Settings for all config
│   └── constants.py            # Collection names, model names, etc.
│
└── tests/
    ├── test_embedder.py
    ├── test_ingestion.py
    └── test_tools.py
```

---

## 3. Infrastructure — Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334

  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"   # Browser UI
      - "7687:7687"   # Bolt protocol
    volumes:
      - neo4j_data:/data
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-changeme}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: "apoc.*"

  postgres:
    image: postgres:17
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: cocoindex
      POSTGRES_USER: cocoindex
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-changeme}

volumes:
  qdrant_data:
  neo4j_data:
  postgres_data:
```

---

## 4. Qdrant Collections Schema

Two collections: one for dense (single-vector) retrieval, one for multi-vector (ColBERT-style) visual retrieval. Both populated by Jina v4 but using different output modes.

### 4.1 Dense Collection — `documents_dense`

For text-heavy content and fast semantic search.

```python
# ingestion/embedder.py — Qdrant collection setup

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    PayloadSchemaType, TextIndexParams
)

DENSE_COLLECTION = "documents_dense"
DENSE_DIM = 2048  # Jina v4 default, truncatable to 128/256/512/1024

def create_dense_collection(client: QdrantClient):
    client.create_collection(
        collection_name=DENSE_COLLECTION,
        vectors_config=VectorParams(
            size=DENSE_DIM,
            distance=Distance.COSINE,
        ),
    )
    # Payload indexes for filtered search
    client.create_payload_index(
        collection_name=DENSE_COLLECTION,
        field_name="source_file",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=DENSE_COLLECTION,
        field_name="content_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )
```

**Point payload schema:**

```python
# Each point in documents_dense carries this payload:
{
    "id": "uuid-v4",
    "source_file": "reports/q3-financials.pdf",
    "content_type": "text_chunk",        # text_chunk | image | pdf_page
    "page_number": 3,                    # null for non-paginated content
    "chunk_index": 0,                    # position within page/document
    "text_content": "Revenue grew 23%...", # original text (for text chunks)
    "metadata": {
        "mime_type": "application/pdf",
        "ingested_at": "2026-02-18T10:30:00Z",
        "source_key": "reports/q3-financials.pdf",
        "char_count": 512,
    }
}
```

### 4.2 Multi-Vector Collection — `documents_multivec`

For visually rich content — scanned PDFs, slides, charts, images. Uses Qdrant's multi-vector support for ColBERT-style late interaction.

```python
MULTIVEC_COLLECTION = "documents_multivec"
MULTIVEC_DIM = 128  # Jina v4 multi-vector token dimension

def create_multivec_collection(client: QdrantClient):
    client.create_collection(
        collection_name=MULTIVEC_COLLECTION,
        vectors_config={
            "colbert": VectorParams(
                size=MULTIVEC_DIM,
                distance=Distance.COSINE,
                multivector_config={"comparator": "max_sim"},
                # max_sim = ColBERT MaxSim scoring
            ),
        },
    )
    client.create_payload_index(
        collection_name=MULTIVEC_COLLECTION,
        field_name="source_file",
        field_schema=PayloadSchemaType.KEYWORD,
    )
```

**Point payload schema:**

```python
# Each point in documents_multivec carries this payload:
{
    "id": "uuid-v4",
    "source_file": "scans/invoice-2024-003.pdf",
    "content_type": "pdf_page",          # pdf_page | image | slide
    "page_number": 1,
    "metadata": {
        "mime_type": "application/pdf",
        "ingested_at": "2026-02-18T10:30:00Z",
        "source_key": "scans/invoice-2024-003.pdf",
        "image_width": 2550,
        "image_height": 3300,
    }
    # Note: no text_content — this is visual-only retrieval
    # The actual page image can be re-fetched from S3 for VLM generation
}
```

### 4.3 When to Use Which Collection

| Query Type | Collection | Jina v4 Mode | Example |
|------------|-----------|--------------|---------|
| Text semantic search | `documents_dense` | `task=retrieval`, single-vector | "What were Q3 revenue figures?" |
| Visual document search | `documents_multivec` | multi-vector (ColBERT) | "Find the org chart diagram" |
| Hybrid (text + visual) | Both, then merge | Both modes | "Show me the table comparing suppliers" |

---

## 5. Neo4j Knowledge Graph Schema

### 5.1 Node Types

```cypher
// Document node — one per source file
CREATE CONSTRAINT doc_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.source_key IS UNIQUE;

// Entity nodes — extracted from content
CREATE CONSTRAINT entity_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE;

// Chunk node — links to its parent document
CREATE CONSTRAINT chunk_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
```

### 5.2 Node Properties

```cypher
// :Document
{
  source_key: "reports/q3-financials.pdf",     // unique identifier
  filename: "q3-financials.pdf",
  mime_type: "application/pdf",
  ingested_at: datetime(),
  page_count: 12,
  source_bucket: "my-rag-bucket"
}

// :Entity
{
  name: "Acme Corp",                       // canonical name
  type: "ORGANIZATION",                    // PERSON | ORGANIZATION | PRODUCT |
                                           // TECHNOLOGY | LOCATION | CONCEPT | EVENT
  description: "Manufacturing company...", // optional summary
  first_seen: datetime(),
  last_seen: datetime()
}

// :Chunk
{
  id: "uuid-v4",                           // matches Qdrant point ID
  text_preview: "Revenue grew 23%...",     // first 200 chars
  page_number: 3,
  chunk_index: 0,
  char_count: 512
}
```

### 5.3 Relationship Types

```cypher
// Document contains Chunks
(:Document)-[:HAS_CHUNK {page_number: 3}]->(:Chunk)

// Chunk mentions Entity
(:Chunk)-[:MENTIONS {confidence: 0.95, context: "Acme Corp reported..."}]->(:Entity)

// Entity-to-Entity relationships (extracted by LLM)
(:Entity {type:"ORGANIZATION"})-[:PARTNERS_WITH {since: "2024"}]->(:Entity {type:"ORGANIZATION"})
(:Entity {type:"PERSON"})-[:WORKS_AT {role: "CEO"}]->(:Entity {type:"ORGANIZATION"})
(:Entity {type:"ORGANIZATION"})-[:PRODUCES]->(:Entity {type:"PRODUCT"})
(:Entity {type:"ORGANIZATION"})-[:USES_TECHNOLOGY]->(:Entity {type:"TECHNOLOGY"})
(:Entity {type:"ORGANIZATION"})-[:LOCATED_IN]->(:Entity {type:"LOCATION"})
(:Entity {type:"ORGANIZATION"})-[:ACQUIRED]->(:Entity {type:"ORGANIZATION"})
(:Entity {type:"ORGANIZATION"})-[:COMPETES_WITH]->(:Entity {type:"ORGANIZATION"})

// Document-level relationships
(:Document)-[:REFERENCES]->(:Document)     // cross-references between docs
(:Document)-[:MENTIONS_ENTITY]->(:Entity)  // direct doc→entity link
```

### 5.4 Graph Query Examples

```cypher
// Find all entities connected to "Acme Corp" within 2 hops
MATCH path = (e:Entity {name: "Acme Corp"})-[*1..2]-(connected)
RETURN path

// Find documents mentioning entities related to a topic
MATCH (e:Entity)-[:MENTIONS]-(c:Chunk)-[:HAS_CHUNK]-(d:Document)
WHERE e.name CONTAINS "AI" OR e.type = "TECHNOLOGY"
RETURN d.filename, collect(DISTINCT e.name) AS entities
ORDER BY size(entities) DESC
LIMIT 10

// Multi-hop: which people work at companies that use a given technology?
MATCH (p:Entity {type:"PERSON"})-[:WORKS_AT]->(org:Entity)-[:USES_TECHNOLOGY]->(tech:Entity)
WHERE tech.name = "Kubernetes"
RETURN p.name, org.name
```

---

## 6. Jina v4 Embedding Integration

### 6.1 API Wrapper

```python
# ingestion/embedder.py

import httpx
from typing import Union
from pathlib import Path
import base64

JINA_API_URL = "https://api.jina.ai/v1/embeddings"

class JinaV4Embedder:
    """
    Single wrapper for all embedding modes.
    
    Jina v4 supports:
    - text → single-vector (2048d, truncatable)
    - text → multi-vector (128d per token, ColBERT-style)
    - image → single-vector (same space as text)
    - image → multi-vector (128d per patch)
    
    Task-specific LoRA adapters (selected via 'task' param):
    - "retrieval.query"  → optimized for queries
    - "retrieval.passage" → optimized for documents/passages
    - "text-matching"    → symmetric similarity
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    
    async def embed_text(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
        dimensions: int = 2048,
    ) -> list[list[float]]:
        """Single-vector text embedding for dense retrieval."""
        payload = {
            "model": "jina-embeddings-v4",
            "task": task,
            "dimensions": dimensions,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": t} for t in texts],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(JINA_API_URL, json=payload, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data["data"]]
    
    async def embed_text_query(self, query: str, dimensions: int = 2048) -> list[float]:
        """Embed a query — uses retrieval.query LoRA adapter."""
        results = await self.embed_text([query], task="retrieval.query", dimensions=dimensions)
        return results[0]
    
    async def embed_image(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
    ) -> list[float]:
        """Single-vector image embedding (same space as text)."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": "jina-embeddings-v4",
            "task": "retrieval.passage",
            "normalized": True,
            "embedding_type": "float",
            "input": [{"image": f"data:{media_type};base64,{b64}"}],
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(JINA_API_URL, json=payload, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
        return data["data"][0]["embedding"]
    
    async def embed_multi_vector(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
    ) -> list[list[float]]:
        """
        Multi-vector (ColBERT-style) embedding for visual documents.
        Returns list of 128-dim vectors, one per image patch/token.
        Used for documents_multivec collection in Qdrant.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": "jina-embeddings-v4",
            "task": "retrieval.passage",
            "normalized": True,
            "embedding_type": "float",
            "input": [{"image": f"data:{media_type};base64,{b64}"}],
            # Multi-vector mode — Jina v4 returns token-level embeddings
            "embedding_type_params": {"output_type": "colbert"},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(JINA_API_URL, json=payload, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
        return data["data"][0]["embedding"]  # list of 128-dim vectors
    
    async def embed_query_multi_vector(self, query: str) -> list[list[float]]:
        """Multi-vector query embedding for ColBERT-style search."""
        payload = {
            "model": "jina-embeddings-v4",
            "task": "retrieval.query",
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": query}],
            "embedding_type_params": {"output_type": "colbert"},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(JINA_API_URL, json=payload, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
        return data["data"][0]["embedding"]
```

### 6.2 Important Notes on Jina v4 API Usage

- **Task parameter matters**: always use `retrieval.query` for queries and `retrieval.passage` for documents — this selects the appropriate LoRA adapter and produces asymmetric embeddings optimized for retrieval
- **Dimensions are truncatable**: 2048 is default but you can request 128, 256, 512, 1024 via the `dimensions` param with minimal quality loss (Matryoshka training)
- **Rate limits**: batch your embedding calls — the API handles internal batching efficiently, send multiple items per request
- **Images**: supports up to 20 megapixel images; for PDFs rendered at 300 DPI, a standard A4 page is about 8.7 megapixels, well within limits

---

## 7. CocoIndex Ingestion Pipeline

### 7.1 Pipeline Definition

```python
# ingestion/pipeline.py

import os
import mimetypes
from io import BytesIO
from dataclasses import dataclass

import cocoindex
from pdf2image import convert_from_bytes

from ingestion.embedder import JinaV4Embedder
from ingestion.entity_extractor import extract_entities
from ingestion.graph_writer import write_to_neo4j

# --- Data classes for pipeline ---

@dataclass
class Page:
    page_number: int | None
    image: bytes           # PNG bytes of the page
    text_content: str      # extracted text (empty for scanned-only docs)

@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page_number: int | None


# --- File classification + conversion ---

@cocoindex.op.function()
def file_to_pages(filename: str, content: bytes) -> list[Page]:
    """
    Classify by MIME type, convert to uniform page representation.
    PDFs → rendered page images + text extraction.
    Images → single page with image bytes.
    Text files → single page with text content.
    """
    mime_type, _ = mimetypes.guess_type(filename)
    
    if mime_type == "application/pdf":
        images = convert_from_bytes(content, dpi=300)
        pages = []
        for i, image in enumerate(images):
            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                png_bytes = buffer.getvalue()
            # TODO: optionally extract text via pdfplumber for hybrid indexing
            pages.append(Page(
                page_number=i + 1,
                image=png_bytes,
                text_content="",  # filled by text extraction step
            ))
        return pages
    
    elif mime_type and mime_type.startswith("image/"):
        return [Page(page_number=None, image=content, text_content="")]
    
    elif mime_type and mime_type.startswith("text/"):
        text = content.decode("utf-8", errors="replace")
        return [Page(page_number=None, image=b"", text_content=text)]
    
    else:
        return []


@cocoindex.op.function()
def semantic_chunk(text: str, max_chunk_size: int = 512) -> list[TextChunk]:
    """
    Split text into semantic chunks.
    
    For production, consider:
    - LangChain's RecursiveCharacterTextSplitter
    - Semantic chunking based on embedding similarity
    - Context-aware chunking (preserve paragraphs, headers)
    """
    # Simplified chunking — replace with semantic chunker
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= max_chunk_size:
            chunks.append(TextChunk(
                text=" ".join(current_chunk),
                chunk_index=len(chunks),
                page_number=None,
            ))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(TextChunk(
            text=" ".join(current_chunk),
            chunk_index=len(chunks),
            page_number=None,
        ))
    
    return chunks


# --- Main CocoIndex flow ---

@cocoindex.flow_def(name="RAGIngestion")
def rag_ingestion_flow(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope,
):
    """
    Main ingestion pipeline:
    S3 → classify → pages → embed (Jina v4) → store (Qdrant + Neo4j)
    """
    bucket_name = os.environ["S3_BUCKET_NAME"]
    sqs_queue_url = os.environ.get("S3_SQS_QUEUE_URL")
    
    # Source: S3 bucket with SQS for real-time updates
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.AmazonS3(
            bucket_name=bucket_name,
            included_patterns=["*.pdf", "*.png", "*.jpg", "*.jpeg",
                              "*.md", "*.txt", "*.pptx"],
            binary=True,
            sqs_queue_url=sqs_queue_url,
        )
    )
    
    # Collectors for Qdrant output
    dense_output = data_scope.add_collector()
    multivec_output = data_scope.add_collector()
    
    with data_scope["documents"].row() as doc:
        # Convert all files to pages
        doc["pages"] = flow_builder.transform(
            file_to_pages,
            filename=doc["filename"],
            content=doc["content"],
        )
        
        with doc["pages"].row() as page:
            # --- Dense embeddings (text content) ---
            # For pages with text, chunk and embed
            # For pages with images, embed the image as single-vector
            
            # --- Multi-vector embeddings (visual content) ---
            # For pages with images, create ColBERT-style embeddings
            # This is where Jina v4 multi-vector mode shines
            
            # Collect for Qdrant export
            dense_output.collect(
                id=cocoindex.GeneratedField.UUID,
                filename=doc["filename"],
                page=page["page_number"],
                # embedding populated by Jina v4 transform
            )
    
    # Export to Qdrant
    qdrant_connection = cocoindex.targets.QdrantConnection(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )
    
    dense_output.export(
        "documents_dense",
        cocoindex.targets.Qdrant(
            connection=qdrant_connection,
            collection_name="documents_dense",
        ),
        primary_key_fields=["id"],
    )
```

### 7.2 Running the Pipeline

```bash
# One-time full index
cocoindex update

# Live update mode — continuously watches S3 via SQS
cocoindex server -L
```

### 7.3 Important: CocoIndex + Custom Embedding Functions

CocoIndex natively supports `SentenceTransformerEmbed` and `ColPaliEmbedImage`, but for Jina v4 via API you'll need a **custom CocoIndex operation** that wraps the Jina API calls. This bridges CocoIndex's declarative pipeline with the Jina v4 HTTP API:

```python
# ingestion/jina_cocoindex_ops.py

@cocoindex.op.function()
async def jina_embed_text(text: str) -> list[float]:
    """Custom CocoIndex op wrapping Jina v4 for text."""
    embedder = JinaV4Embedder(api_key=os.environ["JINA_API_KEY"])
    results = await embedder.embed_text([text], task="retrieval.passage")
    return results[0]

@cocoindex.op.function()
async def jina_embed_image(image_bytes: bytes) -> list[float]:
    """Custom CocoIndex op wrapping Jina v4 for images (single-vector)."""
    embedder = JinaV4Embedder(api_key=os.environ["JINA_API_KEY"])
    return await embedder.embed_image(image_bytes)

@cocoindex.op.function()
async def jina_embed_image_multivec(image_bytes: bytes) -> list[list[float]]:
    """Custom CocoIndex op wrapping Jina v4 for images (multi-vector)."""
    embedder = JinaV4Embedder(api_key=os.environ["JINA_API_KEY"])
    return await embedder.embed_multi_vector(image_bytes)
```

---

## 8. Entity Extraction for Neo4j

### 8.1 LLM-Based Entity Extraction

```python
# ingestion/entity_extractor.py

import json
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    type: str           # PERSON, ORGANIZATION, PRODUCT, TECHNOLOGY, LOCATION, CONCEPT
    description: str = ""

class Relationship(BaseModel):
    source: str         # entity name
    target: str         # entity name
    relation: str       # WORKS_AT, PARTNERS_WITH, PRODUCES, USES_TECHNOLOGY, etc.
    properties: dict = {}

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

Return JSON with this exact structure:
{
  "entities": [
    {"name": "...", "type": "PERSON|ORGANIZATION|PRODUCT|TECHNOLOGY|LOCATION|CONCEPT|EVENT", "description": "..."}
  ],
  "relationships": [
    {"source": "entity_name", "target": "entity_name", "relation": "WORKS_AT|PARTNERS_WITH|PRODUCES|USES_TECHNOLOGY|LOCATED_IN|ACQUIRED|COMPETES_WITH", "properties": {}}
  ]
}

Rules:
- Normalize entity names (e.g., "Google LLC" → "Google")
- Only extract clearly stated relationships, don't infer
- Keep descriptions brief (1 sentence max)
- Use CONCEPT type for abstract topics, methodologies, standards

Text:
{text}
"""


async def extract_entities(text: str, llm_client) -> ExtractionResult:
    """
    Extract entities and relationships from text using an LLM.
    
    The llm_client can be any provider (OpenAI, Anthropic, Ollama).
    For production, consider batching chunks and deduplicating entities.
    """
    response = await llm_client.chat(
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
        response_format={"type": "json_object"},
    )
    
    data = json.loads(response.content)
    return ExtractionResult(**data)
```

### 8.2 Writing to Neo4j

```python
# ingestion/graph_writer.py

from neo4j import AsyncGraphDatabase

class GraphWriter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    async def upsert_document(self, doc: dict):
        """Create or update a Document node."""
        async with self.driver.session() as session:
            await session.run("""
                MERGE (d:Document {source_key: $source_key})
                SET d.filename = $filename,
                    d.mime_type = $mime_type,
                    d.ingested_at = datetime(),
                    d.page_count = $page_count,
                    d.source_bucket = $source_bucket
            """, **doc)
    
    async def upsert_chunk(self, chunk_id: str, text_preview: str,
                           page_number: int, source_key: str):
        """Create Chunk node and link to Document."""
        async with self.driver.session() as session:
            await session.run("""
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text_preview = $text_preview,
                    c.page_number = $page_number
                WITH c
                MATCH (d:Document {source_key: $source_key})
                MERGE (d)-[:HAS_CHUNK {page_number: $page_number}]->(c)
            """, chunk_id=chunk_id, text_preview=text_preview[:200],
                 page_number=page_number, source_key=source_key)
    
    async def upsert_entity(self, entity: dict):
        """Create or update an Entity node."""
        async with self.driver.session() as session:
            await session.run("""
                MERGE (e:Entity {name: $name, type: $type})
                SET e.description = $description,
                    e.last_seen = datetime()
                ON CREATE SET e.first_seen = datetime()
            """, **entity)
    
    async def upsert_relationship(self, source: str, target: str,
                                   relation: str, properties: dict = {}):
        """Create relationship between entities."""
        async with self.driver.session() as session:
            # Dynamic relationship type via APOC
            await session.run("""
                MATCH (s:Entity {name: $source})
                MATCH (t:Entity {name: $target})
                CALL apoc.merge.relationship(s, $relation, $props, {}, t, {})
                YIELD rel
                RETURN rel
            """, source=source, target=target,
                 relation=relation, props=properties)
    
    async def link_chunk_to_entity(self, chunk_id: str, entity_name: str,
                                    confidence: float = 1.0):
        """Link a Chunk to an Entity it mentions."""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (e:Entity {name: $entity_name})
                MERGE (c)-[r:MENTIONS]->(e)
                SET r.confidence = $confidence
            """, chunk_id=chunk_id, entity_name=entity_name,
                 confidence=confidence)
    
    async def close(self):
        await self.driver.close()
```

---

## 9. MCP Server & Tools

### 9.1 Server Definition

```python
# server/mcp_server.py

from mcp.server.fastmcp import FastMCP
from server.tools.vector_search import vector_search
from server.tools.visual_search import visual_search
from server.tools.graph_search import graph_search
from server.tools.hybrid_search import hybrid_search

mcp = FastMCP(
    name="rag-knowledge-base",
    description="Multimodal RAG with knowledge graph — search documents, "
                "images, and entity relationships",
)

# Register all tools
mcp.tool()(vector_search)
mcp.tool()(visual_search)
mcp.tool()(graph_search)
mcp.tool()(hybrid_search)

if __name__ == "__main__":
    mcp.run(transport="sse")  # or "stdio" for local agent integration
```

### 9.2 Tool Implementations

```python
# server/tools/vector_search.py

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from ingestion.embedder import JinaV4Embedder
from config.settings import settings

async def vector_search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    source_file: str | None = None,
) -> list[dict]:
    """
    Semantic search over document chunks using dense vectors.
    
    Best for: text-based queries, finding relevant passages,
    searching by meaning rather than keywords.
    
    Args:
        query: Natural language search query
        limit: Max results to return (default 10)
        content_type: Filter by type — "text_chunk", "pdf_page", "image"
        source_file: Filter by specific source file path
    
    Returns:
        List of matching chunks with text, metadata, and relevance scores
    """
    embedder = JinaV4Embedder(api_key=settings.jina_api_key)
    client = QdrantClient(url=settings.qdrant_url)
    
    # Embed query with retrieval.query LoRA adapter
    query_vector = await embedder.embed_text_query(query)
    
    # Build optional filters
    conditions = []
    if content_type:
        conditions.append(FieldCondition(
            key="content_type", match=MatchValue(value=content_type)
        ))
    if source_file:
        conditions.append(FieldCondition(
            key="source_file", match=MatchValue(value=source_file)
        ))
    
    search_filter = Filter(must=conditions) if conditions else None
    
    results = client.query_points(
        collection_name="documents_dense",
        query=query_vector,
        limit=limit,
        query_filter=search_filter,
        with_payload=True,
    )
    
    return [
        {
            "score": point.score,
            "text": point.payload.get("text_content", ""),
            "source_file": point.payload.get("source_file"),
            "page_number": point.payload.get("page_number"),
            "content_type": point.payload.get("content_type"),
            "metadata": point.payload.get("metadata", {}),
        }
        for point in results.points
    ]
```

```python
# server/tools/visual_search.py

async def visual_search(
    query: str,
    limit: int = 5,
) -> list[dict]:
    """
    Visual document search using multi-vector (ColBERT) embeddings.
    
    Best for: finding specific charts, diagrams, tables, scanned pages,
    or any visually rich content. Understands layout, not just text.
    
    Args:
        query: Natural language description of what you're looking for
        limit: Max results to return (default 5)
    
    Returns:
        List of matching document pages with source file, page number,
        and relevance scores. Page images can be retrieved from S3.
    """
    embedder = JinaV4Embedder(api_key=settings.jina_api_key)
    client = QdrantClient(url=settings.qdrant_url)
    
    # Multi-vector query embedding
    query_vectors = await embedder.embed_query_multi_vector(query)
    
    results = client.query_points(
        collection_name="documents_multivec",
        query=query_vectors,
        using="colbert",
        limit=limit,
        with_payload=True,
    )
    
    return [
        {
            "score": point.score,
            "source_file": point.payload.get("source_file"),
            "page_number": point.payload.get("page_number"),
            "content_type": point.payload.get("content_type"),
            "source_key": point.payload.get("metadata", {}).get("source_key"),
            "metadata": point.payload.get("metadata", {}),
        }
        for point in results.points
    ]
```

```python
# server/tools/graph_search.py

from neo4j import AsyncGraphDatabase

async def graph_search(
    query: str,
    search_type: str = "entity",
    limit: int = 10,
) -> list[dict]:
    """
    Knowledge graph search for entity relationships and connections.
    
    Best for: finding how things are connected, multi-hop questions,
    "who works at", "what companies use X", relationship queries.
    
    Args:
        query: Entity name or relationship question
        search_type: "entity" (find entity + connections) or
                     "path" (find paths between entities)
        limit: Max results to return
    
    Returns:
        List of entities, relationships, and connected documents
    """
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    
    async with driver.session() as session:
        if search_type == "entity":
            result = await session.run("""
                // Full-text search on entity names and descriptions
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                   OR toLower(e.description) CONTAINS toLower($query)
                
                // Get connected entities (1 hop)
                OPTIONAL MATCH (e)-[r]-(connected:Entity)
                
                // Get source documents
                OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
                OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
                
                RETURN e.name AS entity,
                       e.type AS type,
                       e.description AS description,
                       collect(DISTINCT {
                           name: connected.name,
                           type: connected.type,
                           relation: type(r)
                       }) AS connections,
                       collect(DISTINCT d.filename) AS source_documents
                LIMIT $limit
            """, query=query, limit=limit)
            
            records = [record.data() async for record in result]
        
        elif search_type == "path":
            # Find shortest paths between two entities mentioned in query
            # Agent should parse the query to extract entity names
            result = await session.run("""
                MATCH (start:Entity), (end:Entity)
                WHERE toLower(start.name) CONTAINS toLower($query)
                MATCH path = shortestPath((start)-[*..4]-(end))
                WHERE start <> end
                RETURN [n IN nodes(path) | n.name] AS path_nodes,
                       [r IN relationships(path) | type(r)] AS path_relations,
                       length(path) AS hops
                ORDER BY hops
                LIMIT $limit
            """, query=query, limit=limit)
            
            records = [record.data() async for record in result]
    
    await driver.close()
    return records
```

```python
# server/tools/hybrid_search.py

async def hybrid_search(
    query: str,
    limit: int = 10,
) -> dict:
    """
    Combined vector + graph search with result fusion.
    
    Best for: complex questions that benefit from both semantic
    similarity AND entity relationships. Runs both searches in
    parallel, then merges and deduplicates results.
    
    Args:
        query: Natural language question
        limit: Max results per source
    
    Returns:
        Combined results from vector search and graph search,
        with source attribution for each result.
    """
    import asyncio
    
    # Run both searches in parallel
    vector_results, graph_results = await asyncio.gather(
        vector_search(query, limit=limit),
        graph_search(query, search_type="entity", limit=limit),
    )
    
    return {
        "vector_results": vector_results,
        "graph_results": graph_results,
        "query": query,
        "strategy": "parallel_fusion",
    }
```

---

## 10. Configuration

### 10.1 Settings

```python
# config/settings.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Jina v4
    jina_api_key: str
    jina_model: str = "jina-embeddings-v4"
    jina_dense_dimensions: int = 2048
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_dense_collection: str = "documents_dense"
    qdrant_multivec_collection: str = "documents_multivec"
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"
    
    # PostgreSQL (CocoIndex state only)
    database_url: str = "postgresql://cocoindex:changeme@localhost:5432/cocoindex"
    
    # AWS S3
    s3_bucket_name: str
    s3_sqs_queue_url: str | None = None
    aws_region: str = "eu-central-1"
    
    # LLM for entity extraction
    llm_provider: str = "anthropic"       # anthropic | openai | ollama
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key: str | None = None
    
    # MCP Server
    mcp_transport: str = "sse"            # sse | stdio
    mcp_port: int = 8080
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 10.2 Environment File

```bash
# .env.example

# === Jina v4 (required) ===
JINA_API_KEY=jina_xxxxxxxxxxxxxxxxxxxx

# === Qdrant ===
QDRANT_URL=http://localhost:6333

# === Neo4j ===
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password

# === PostgreSQL (CocoIndex) ===
DATABASE_URL=postgresql://cocoindex:your-password@localhost:5432/cocoindex
POSTGRES_PASSWORD=your-password

# === AWS S3 ===
S3_BUCKET_NAME=your-rag-bucket
S3_SQS_QUEUE_URL=https://sqs.eu-central-1.amazonaws.com/123456789/YourQueueName
AWS_REGION=eu-central-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# === LLM for entity extraction ===
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=sk-ant-...

# === MCP Server ===
MCP_TRANSPORT=sse
MCP_PORT=8080
```

---

## 11. Key References

Each reference listed with what specifically to take from it and what to be cautious about.

### Core Architecture References

| Reference | What to Use | Caveats |
|-----------|------------|---------|
| **[Cole Medin — Agentic RAG Knowledge Graph](https://github.com/coleam00/ottomator-agents/tree/main/agentic-rag-knowledge-graph)** | Overall agent architecture pattern, Pydantic AI agent structure, Neo4j + pgvector dual-search approach, FastAPI streaming pattern, project layout (`agent/`, `ingestion/`, `sql/`) | Uses pgvector not Qdrant, uses OpenAI/nomic embeddings not Jina v4, text-only (no multimodal). Adapt the structure but replace the embedding and vector DB layers. |
| **[Cole Medin — MCP Server Template](https://github.com/coleam00/mcp-mem0)** | MCP server scaffolding pattern, FastMCP tool registration, SSE transport setup | This is a mem0-based template, not RAG. Take the MCP server skeleton only, ignore the mem0 memory parts. |
| **[Cole Medin — Pydantic AI MCP Agent](https://github.com/coleam00/ottomator-agents/tree/main/pydantic-ai-mcp-agent)** | How to build a Pydantic AI agent that connects to MCP servers as tools, provider abstraction pattern | Good reference for the agent side. The MCP client connection code is directly reusable. |

### Ingestion & Data Pipeline References

| Reference | What to Use | Caveats |
|-----------|------------|---------|
| **[CocoIndex — Multi-Format Indexing](https://cocoindex.io/examples/multi_format_index)** | `file_to_pages` pattern (MIME classification → page images), PDF-to-image conversion at 300 DPI, Qdrant export, ColPali visual embedding flow | Uses ColPali natively — for Jina v4, you need a custom CocoIndex op (see Section 7.3). The pipeline shape is right, just swap the embedding function. |
| **[CocoIndex — S3 + SQS Pipeline](https://cocoindex.io/examples/s3_sqs_pipeline)** | S3 source configuration, SQS queue integration, live update mode (`cocoindex server -L`), incremental processing pattern | The example uses `SentenceTransformerEmbed` — replace with your Jina v4 custom op. IAM permissions guide is accurate and directly usable. |
| **[CocoIndex — Live Updates Docs](https://cocoindex.io/docs/tutorials/live_updates)** | `FlowLiveUpdater` class for programmatic control, refresh interval configuration, change detection mechanics | — |

### Embedding Model References

| Reference | What to Use | Caveats |
|-----------|------------|---------|
| **[Jina v4 — Model Card (HuggingFace)](https://huggingface.co/jinaai/jina-embeddings-v4)** | API usage examples, task/LoRA adapter selection, dimension truncation, multi-vector output | API examples on the model card show the self-hosted `sentence-transformers` usage — for API usage, follow the [Jina Embedding API docs](https://jina.ai/embeddings/) instead. |
| **[Jina v4 — Technical Report](https://arxiv.org/abs/2506.18902)** | Architecture understanding (single-stream, not dual-encoder), benchmark numbers, modality gap analysis, LoRA adapter details | Academic paper — not for implementation, but useful to understand why unified embedding works. |
| **[ColPali Engine](https://github.com/illuin-tech/colpali)** | Understanding ColBERT-style multi-vector retrieval, MaxSim scoring, how multi-vector differs from single-vector | We use Jina v4's multi-vector mode, not ColPali directly. But the retrieval mechanics (MaxSim, per-patch embeddings) are the same concept. Useful for understanding how Qdrant's `max_sim` comparator works. |

### Multimodal RAG Pattern References

| Reference | What to Use | Caveats |
|-----------|------------|---------|
| **[dollro — Multimodal RAG Demo](https://github.com/dollro/multimodal-rag-demo)** | End-to-end multimodal RAG pattern with Qwen3-VL, visual retrieval + VLM generation pipeline | Uses Qwen3-VL embeddings, not Jina v4. The retrieval→VLM generation pattern is reusable conceptually. |
| **[HuggingFace — Multimodal RAG Cookbook](https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_vlms)** | ColPali + Qdrant retrieval flow, how to feed retrieved images to a VLM for answer generation | Uses `byaldi` library and ColPali directly. Take the VLM generation pattern but adapt retrieval to use Jina v4 via API. |
| **[Pipeshub RAG Service](https://github.com/pipeshub-ai/pipeshub-ai)** | Full-stack RAG service design, connector patterns, document processing pipeline | Much larger/heavier than what we need. Good for inspiration on document processing edge cases but don't adopt the full framework. |

### Knowledge Graph References

| Reference | What to Use | Caveats |
|-----------|------------|---------|
| **[Cole Medin — Archon](https://github.com/coleam00/Archon)** | Crawl → chunk → embed → RAG pattern for code/docs, how to structure a RAG system for coding assistants | Focused on code documentation RAG, uses Supabase not Qdrant. Chunking strategy ideas are good, but the vector DB and embedding layers differ. |
| **[Neo4j — GraphRAG + Agentic Architecture Blog](https://neo4j.com/blog/developer/graphrag-and-agentic-architecture-with-neoconverse/)** | Conceptual understanding of GraphRAG vs traditional RAG, when graph queries outperform vector search, tool configuration patterns for agents | Neo4j marketing content — the code examples use their specific NeoConverse framework which we don't use. Take the architectural concepts, not the code. |
| **[DeepLearning.AI — Agentic Knowledge Graph Construction](https://learn.deeplearning.ai/courses/agentic-knowledge-graph-construction/)** | Multi-agent pattern for automated graph schema design, entity extraction workflows, how to connect structured (CSV) and unstructured (text) data in the same graph | Uses Google ADK framework, not Pydantic AI. The graph construction methodology (propose schema → validate → construct) is excellent but the specific framework code won't port directly. |

---

## 12. Implementation Order

Suggested build sequence, each step testable independently:

### Phase 1 — Foundation (Days 1–2)

1. `docker-compose up` — Neo4j, Qdrant, PostgreSQL running
2. Create Qdrant collections (dense + multivec) with correct schemas
3. Create Neo4j constraints and indexes
4. Test Jina v4 API wrapper — embed sample text + image, verify vectors

### Phase 2 — Ingestion Pipeline (Days 3–5)

5. Build `file_to_pages` processor (MIME classification, PDF→images)
6. Build custom CocoIndex ops for Jina v4 embedding
7. Wire up CocoIndex flow: local files → embed → Qdrant
8. Add entity extraction → Neo4j writing
9. Switch source to S3 + SQS, test with sample bucket

### Phase 3 — MCP Server (Days 6–7)

10. Scaffold FastMCP server with `vector_search` tool
11. Add `visual_search` (multi-vector), `graph_search`, `hybrid_search`
12. Test each tool independently via MCP client

### Phase 4 — Agent Integration (Day 8)

13. Build Pydantic AI agent that connects to MCP server
14. Agent system prompt: when to use which tool
15. End-to-end test: upload file to S3 → wait for ingestion → query via agent

### Phase 5 — Production Hardening

16. Error handling, retries, rate limiting on Jina API
17. Monitoring: ingestion lag, query latency, embedding costs
18. Optional: add re-ranking step (Jina Reranker or cross-encoder)
19. Optional: VLM generation for visual search results (Qwen3-VL / GPT-4o)
