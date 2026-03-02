# Implementation Plan: RAG-as-MCP-Server (Spektr)

## Overview

Build a hybrid GraphRAG + multimodal vector search system exposed as an MCP server. Documents land in S3, trigger SQS events, flow through a CocoIndex pipeline (classify, PDF-to-images, Jina v4 embeddings, entity extraction), persist in Qdrant (dense + ColBERT multi-vector collections) and Neo4j (knowledge graph), and are queried via FastMCP tools consumed by a Pydantic AI agent.

**Stack:** Jina v4 (API) | Neo4j | Qdrant | CocoIndex | PostgreSQL | FastMCP | Pydantic AI | AWS S3/SQS

**Blueprint reference:** `/home/rodo/Coding/spektr/rag-mcp-architecture-blueprint.md`

**Key external references:**
- Cole Medin's agentic-rag-knowledge-graph repo -- overall architecture pattern, Neo4j integration
- Cole Medin's MCP server template -- FastMCP scaffolding, SSE transport
- CocoIndex multi-format indexing + S3/SQS examples -- pipeline shape, source config
- Jina v4 model card + API docs -- embedding API contract, task/LoRA adapter selection

---

## Task Registry

| ID | Task | Phase | Depends On | Parallel With | Parallel Safe | Risk Flags | Agent Focus | Effort |
|-|-|-|-|-|-|-|-|-|
| 1.1 | Project scaffold + pyproject.toml | 1 | -- | 1.2 | Yes, different files | -- | backend | S |
| 1.2 | Docker Compose + infrastructure health | 1 | -- | 1.1 | Yes, different files | -- | devops | S |
| 1.3 | Configuration module (settings + constants) | 1 | 1.1 | -- | No | -- | backend | S |
| 1.4 | Qdrant collection provisioning | 1 | 1.2, 1.3 | 1.5 | Yes, different DBs | R1 | backend | S |
| 1.5 | Neo4j schema provisioning | 1 | 1.2, 1.3 | 1.4 | Yes, different DBs | R1 | backend | S |
| 1.6 | Jina v4 embedder wrapper + tests | 1 | 1.3 | 1.4, 1.5 | Yes, no shared files | R2 | backend | M |
| 2.1 | File processor (MIME classify, PDF-to-images, chunking) | 2 | 1.1 | 2.2 | Yes, no shared files | -- | backend | M |
| 2.2 | Entity extractor + Pydantic models | 2 | 1.3 | 2.1 | Yes, no shared files | R3 | backend | M |
| 2.3 | Graph writer (Neo4j upserts) | 2 | 1.5, 2.2 | -- | No | -- | backend | M |
| 2.4 | Custom CocoIndex Jina v4 ops | 2 | 1.6 | 2.1, 2.2 | Yes, no shared files | R4 | backend | M |
| 2.5 | CocoIndex pipeline (local files source) | 2 | 2.1, 2.3, 2.4 | -- | No | R4, R5 | backend | L |
| 2.6 | S3+SQS source integration | 2 | 2.5 | -- | No | R6 | backend | M |
| 2.7 | Ingestion integration tests | 2 | 2.5 | 2.6 | Review required, tests touch pipeline | -- | backend | M |
| 3.1 | MCP server scaffold + vector_search tool | 3 | 1.4, 1.6 | -- | No | -- | backend | M |
| 3.2 | visual_search tool (ColBERT multi-vector) | 3 | 3.1 | 3.3 | Yes, separate tool files | -- | backend | M |
| 3.3 | graph_search tool (Neo4j Cypher) | 3 | 1.5, 3.1 | 3.2 | Yes, separate tool files | -- | backend | M |
| 3.4 | hybrid_search tool (vector + graph fusion) | 3 | 3.2, 3.3 | -- | No | -- | backend | S |
| 3.5 | Server data models + providers abstraction | 3 | 3.1 | 3.2, 3.3 | Yes, separate files | -- | backend | S |
| 3.6 | MCP tool integration tests | 3 | 3.4 | -- | No | -- | backend | M |
| 4.1 | Pydantic AI agent with MCP tool bindings | 4 | 3.4 | 4.2 | Yes, separate files | -- | fullstack | M |
| 4.2 | FastAPI endpoint (optional HTTP access) | 4 | 4.1 | -- | No | -- | fullstack | S |
| 4.3 | End-to-end integration test | 4 | 4.1, 2.6 | -- | No | R6 | fullstack | L |
| 5.1 | Error handling, retries, rate limiting | 5 | 3.4, 2.6 | 5.2 | Yes, different files | -- | backend | M |
| 5.2 | Monitoring + observability hooks | 5 | 3.4, 2.6 | 5.1 | Yes, different files | -- | devops | M |
| 5.3 | Optional: re-ranking step | 5 | 3.4 | 5.4 | Yes, independent feature | -- | backend | M |
| 5.4 | Optional: VLM generation for visual results | 5 | 3.2 | 5.3 | Yes, independent feature | -- | backend | M |

---

## Risk Summary

| ID | Risk | Affected Tasks | Mitigation |
|-|-|-|-|
| R1 | Infrastructure containers fail to start or have version incompatibilities (Neo4j APOC plugin, Qdrant multi-vector config) | 1.4, 1.5, all downstream | Pin exact image versions in docker-compose.yml. Validate APOC plugin loads in a smoke test. Test Qdrant multi-vector config with a dummy point before pipeline runs. |
| R2 | Jina v4 API contract mismatch -- multi-vector (ColBERT) output format may differ from blueprint assumptions | 1.6, 2.4, 3.2 | Write an isolated integration test against the live Jina API with both single-vector and multi-vector modes before building pipeline. Capture raw response shapes. |
| R3 | Entity extraction LLM produces inconsistent JSON -- varying entity names, relationship types not in schema, malformed output | 2.2, 2.3, 2.5 | Use Pydantic model validation on LLM output with retry on parse failure. Normalize entity names (lowercase, strip suffixes). Log extraction failures rather than crashing pipeline. |
| R4 | CocoIndex custom operation API -- wrapping async Jina API calls as CocoIndex ops may conflict with CocoIndex's execution model | 2.4, 2.5 | Review CocoIndex source and examples for async op support before implementation. Fallback: wrap async calls with `asyncio.run()` in sync ops if CocoIndex ops must be synchronous. |
| R5 | CocoIndex + Qdrant export for multi-vector collections -- native Qdrant target may not support multi-vector named vectors | 2.5 | Test Qdrant export with multi-vector payload. Fallback: write multi-vector points directly via Qdrant client in a custom CocoIndex op, bypassing the declarative export for the multivec collection. |
| R6 | AWS S3/SQS permissions and event configuration are external to this codebase -- IAM roles, bucket notifications, queue policies | 2.6, 4.3 | Document required IAM permissions and S3 event notification setup in a separate AWS_SETUP.md. Provide a LocalStack-based docker-compose override for local development/testing. |

---

## Traceability Matrix

| Capability | Description | Tasks | Data Entities | API/Tools | Status |
|-|-|-|-|-|-|
| CAP-1 | Documents ingested from S3 via SQS into pipeline | 2.1, 2.4, 2.5, 2.6 | Document, Page, TextChunk | S3 source, SQS queue | -- |
| CAP-2 | PDF/image/text classified by MIME, PDFs rendered to images at 300 DPI | 2.1, 2.5 | Page | file_to_pages op | -- |
| CAP-3 | Text chunks embedded as dense vectors (Jina v4, 2048d) and stored in Qdrant | 1.6, 2.4, 2.5, 1.4 | documents_dense collection | Jina v4 API (retrieval.passage) | -- |
| CAP-4 | Page images embedded as multi-vectors (ColBERT, 128d) and stored in Qdrant | 1.6, 2.4, 2.5, 1.4 | documents_multivec collection | Jina v4 API (ColBERT mode) | -- |
| CAP-5 | Entities and relationships extracted from text and stored in Neo4j | 2.2, 2.3, 2.5, 1.5 | Entity, Chunk, Document nodes + relationships | LLM entity extraction | -- |
| CAP-6 | Dense vector search via MCP tool | 3.1, 3.5 | documents_dense | vector_search MCP tool | -- |
| CAP-7 | Visual multi-vector search via MCP tool | 3.2, 3.5 | documents_multivec | visual_search MCP tool | -- |
| CAP-8 | Knowledge graph search via MCP tool | 3.3, 3.5 | Neo4j graph | graph_search MCP tool | -- |
| CAP-9 | Hybrid search (vector + graph fusion) via MCP tool | 3.4 | Both | hybrid_search MCP tool | -- |
| CAP-10 | Pydantic AI agent selects and calls MCP tools based on query intent | 4.1, 4.2 | -- | All MCP tools | -- |
| CAP-11 | Production error handling, retries, rate limiting | 5.1 | -- | Jina API, all tools | -- |
| CAP-12 | Monitoring and observability | 5.2 | -- | All components | -- |

---

## Phase 1: Foundation

### Overview

Stand up infrastructure, create the Python project skeleton, configure all service connections, provision database schemas, and validate the Jina v4 API integration. Everything in later phases depends on this working correctly. Phase 1 produces no user-facing functionality but ensures all building blocks are solid.

### Task 1.1: Project Scaffold + pyproject.toml

- **Agent:** backend-developer
- **Files to create:**
  - `pyproject.toml`
  - `ingestion/__init__.py`
  - `server/__init__.py`
  - `server/tools/__init__.py`
  - `agent/__init__.py`
  - `config/__init__.py`
  - `tests/__init__.py`
  - `.gitignore`
  - `.env.example`
- **Depends on:** None
- **Parallel with:** 1.2
- **Parallel safe:** Yes -- creates Python project files, no overlap with Docker/infra files
- **Creates:** Installable Python package with all dependencies declared
- **Uses:** Nothing (greenfield)
- **Acceptance criteria:**
  - `pyproject.toml` declares all dependencies: `cocoindex`, `qdrant-client`, `neo4j`, `httpx`, `fastmcp`, `pydantic-ai`, `pydantic-settings`, `pdf2image`, `boto3`, `uvicorn`, `fastapi`
  - Dev dependencies: `pytest`, `pytest-asyncio`, `ruff`, `mypy`
  - Package is installable via `pip install -e .`
  - All `__init__.py` files created for the directory structure from blueprint Section 2
  - `.env.example` contains all variables from blueprint Section 10.2
  - `.gitignore` excludes `.env`, `__pycache__`, `.venv`, `*.pyc`, volume data

### Task 1.2: Docker Compose + Infrastructure Health

- **Agent:** devops-engineer
- **Files to create:**
  - `docker-compose.yml`
  - `scripts/wait-for-services.sh`
- **Depends on:** None
- **Parallel with:** 1.1
- **Parallel safe:** Yes -- Docker/shell files only, no overlap with Python scaffold
- **Creates:** Running Neo4j, Qdrant, PostgreSQL containers
- **Uses:** Nothing (standalone infrastructure)
- **Acceptance criteria:**
  - `docker-compose.yml` per blueprint Section 3, with these changes:
    - Pin specific image versions (not `latest`): `qdrant/qdrant:v1.13.2`, `neo4j:5.26-community`, `postgres:17.2`
    - Neo4j: APOC plugin enabled, password from env var, `dbms.security.procedures.unrestricted=apoc.*`
    - Qdrant: gRPC port exposed alongside REST
    - PostgreSQL: database `cocoindex`, user `cocoindex`
  - `docker compose up -d` brings all three services to healthy state
  - `scripts/wait-for-services.sh` polls health endpoints:
    - Qdrant: `GET http://localhost:6333/healthz`
    - Neo4j: `GET http://localhost:7474` (browser UI) or bolt handshake
    - PostgreSQL: `pg_isready -h localhost -p 5432`
  - Volume mounts for data persistence across restarts

### Task 1.3: Configuration Module (Settings + Constants)

- **Agent:** backend-developer
- **Files to create:**
  - `config/settings.py`
  - `config/constants.py`
- **Depends on:** 1.1 (project must be installable for imports)
- **Parallel with:** None (quick task, not worth parallelizing)
- **Parallel safe:** N/A
- **Creates:** Centralized configuration consumed by all other modules
- **Uses:** `pydantic-settings` library, `.env` file
- **Acceptance criteria:**
  - `config/settings.py` implements `Settings(BaseSettings)` per blueprint Section 10.1 with all fields:
    - Jina v4: `jina_api_key`, `jina_model`, `jina_dense_dimensions`
    - Qdrant: `qdrant_url`, `qdrant_dense_collection`, `qdrant_multivec_collection`
    - Neo4j: `neo4j_uri`, `neo4j_user`, `neo4j_password`
    - PostgreSQL: `database_url`
    - AWS: `s3_bucket_name`, `s3_sqs_queue_url`, `aws_region`
    - LLM: `llm_provider`, `llm_model`, `llm_api_key`
    - MCP: `mcp_transport`, `mcp_port`
  - `config/constants.py` defines:
    - `DENSE_COLLECTION = "documents_dense"`
    - `MULTIVEC_COLLECTION = "documents_multivec"`
    - `DENSE_DIM = 2048`
    - `MULTIVEC_DIM = 128`
    - Entity types: `ENTITY_TYPES = ["PERSON", "ORGANIZATION", "PRODUCT", "TECHNOLOGY", "LOCATION", "CONCEPT", "EVENT"]`
    - Relationship types: `RELATIONSHIP_TYPES = ["WORKS_AT", "PARTNERS_WITH", "PRODUCES", "USES_TECHNOLOGY", "LOCATED_IN", "ACQUIRED", "COMPETES_WITH", "REFERENCES"]`
  - Module-level `settings = Settings()` singleton
  - Settings load from `.env` file and environment variables

### Task 1.4: Qdrant Collection Provisioning

- **Agent:** backend-developer
- **Files to create:**
  - `ingestion/qdrant_setup.py`
  - `tests/test_qdrant_setup.py`
- **Depends on:** 1.2 (Qdrant running), 1.3 (settings + constants)
- **Parallel with:** 1.5
- **Parallel safe:** Yes -- touches Qdrant only, 1.5 touches Neo4j only, no shared files
- **Creates:** Two Qdrant collections ready for ingestion
- **Uses:** `qdrant-client`, `config.settings`, `config.constants`
- **Acceptance criteria:**
  - `create_dense_collection()` creates `documents_dense` per blueprint Section 4.1:
    - Vector params: size=2048, distance=COSINE
    - Payload indexes on `source_file` (KEYWORD), `content_type` (KEYWORD)
  - `create_multivec_collection()` creates `documents_multivec` per blueprint Section 4.2:
    - Named vector `colbert`: size=128, distance=COSINE, `multivector_config={"comparator": "max_sim"}`
    - Payload index on `source_file` (KEYWORD)
  - Both functions are idempotent (skip creation if collection exists)
  - `ensure_collections()` convenience function calls both
  - Test: create collections, verify they exist via `client.get_collection()`, verify vector params match spec
  - Test: idempotency -- calling twice does not error

### Task 1.5: Neo4j Schema Provisioning

- **Agent:** backend-developer
- **Files to create:**
  - `ingestion/neo4j_setup.py`
  - `tests/test_neo4j_setup.py`
- **Depends on:** 1.2 (Neo4j running), 1.3 (settings)
- **Parallel with:** 1.4
- **Parallel safe:** Yes -- touches Neo4j only, 1.4 touches Qdrant only, no shared files
- **Creates:** Neo4j constraints and indexes ready for graph writes
- **Uses:** `neo4j` async driver, `config.settings`
- **Acceptance criteria:**
  - `create_neo4j_schema()` runs the Cypher from blueprint Section 5.1:
    - Uniqueness constraint on `Document.s3_key`
    - Uniqueness constraint on `Entity(name, type)` composite
    - Uniqueness constraint on `Chunk.id`
  - Function is idempotent (uses `IF NOT EXISTS` per blueprint)
  - Verifies APOC plugin is available (call `RETURN apoc.version()`)
  - Test: run schema creation, verify constraints exist via `SHOW CONSTRAINTS`
  - Test: APOC availability confirmed

### Task 1.6: Jina v4 Embedder Wrapper + Tests

- **Agent:** backend-developer
- **Files to create:**
  - `ingestion/embedder.py`
  - `tests/test_embedder.py`
- **Depends on:** 1.3 (settings for API key)
- **Parallel with:** 1.4, 1.5
- **Parallel safe:** Yes -- standalone module, no shared files with DB setup tasks
- **Creates:** `JinaV4Embedder` class -- the single embedding interface for the entire system
- **Uses:** `httpx`, `config.settings`
- **Acceptance criteria:**
  - Implements `JinaV4Embedder` per blueprint Section 6.1 with all five methods:
    - `embed_text(texts, task, dimensions)` -- batch text to single-vector
    - `embed_text_query(query, dimensions)` -- single query to single-vector (uses `retrieval.query` task)
    - `embed_image(image_bytes, media_type)` -- image to single-vector
    - `embed_multi_vector(image_bytes, media_type)` -- image to list of 128d vectors (ColBERT)
    - `embed_query_multi_vector(query)` -- text query to list of 128d vectors (ColBERT)
  - Uses shared `httpx.AsyncClient` (not per-request creation as in blueprint -- optimization)
  - Proper error handling: raise clear errors on HTTP 4xx/5xx with response body context
  - Respects blueprint Section 6.2 notes: task parameter for LoRA adapters, batch input, normalized output
  - **Integration test** (requires live Jina API key, mark with `@pytest.mark.integration`):
    - Embed a short text, verify output is list of 2048 floats
    - Embed same text as query, verify different vector (asymmetric LoRA)
    - Embed a small PNG image, verify output is list of 2048 floats
    - Embed same image in multi-vector mode, verify output is list of lists, inner dim 128
    - Embed a text query in multi-vector mode, verify list of lists structure
  - **Unit test** (mocked HTTP):
    - Verify correct API URL, headers, payload structure for each method
    - Verify `task` parameter is `retrieval.passage` for documents, `retrieval.query` for queries

---

## Phase 2: Ingestion Pipeline

### Overview

Build the document processing pipeline that converts files from S3 into embeddings (Qdrant) and knowledge graph nodes (Neo4j). This is the core data path. Tasks 2.1 and 2.2 are independent processors that can be built in parallel. Task 2.3 depends on 2.2's data models. Task 2.5 wires everything together in a CocoIndex flow, making it the integration point and highest-risk task.

### Task 2.1: File Processor (MIME Classify, PDF-to-Images, Chunking)

- **Agent:** backend-developer
- **Files to create:**
  - `ingestion/file_processor.py`
  - `tests/test_file_processor.py`
  - `tests/fixtures/` (sample PDF, PNG, TXT files for testing)
- **Depends on:** 1.1 (pdf2image dependency available)
- **Parallel with:** 2.2
- **Parallel safe:** Yes -- completely different file domains, no shared state
- **Creates:** Functions `file_to_pages()` and `semantic_chunk()` used by CocoIndex pipeline
- **Uses:** `pdf2image`, `mimetypes`, standard library
- **Acceptance criteria:**
  - `file_to_pages(filename, content)` per blueprint Section 7.1:
    - PDF: renders each page at 300 DPI via `pdf2image.convert_from_bytes`, returns list of `Page` dataclasses with PNG bytes + empty text
    - Image (PNG/JPG/JPEG): returns single `Page` with original bytes
    - Text (MD/TXT): returns single `Page` with text content, empty image bytes
    - PPTX: returns empty list (or basic support -- document assumption if skipping)
    - Unknown MIME: returns empty list, logs warning
  - `semantic_chunk(text, max_chunk_size=512)` per blueprint Section 7.1:
    - Splits text into `TextChunk` dataclasses
    - Preserves paragraph boundaries where possible (improve over blueprint's naive word splitting)
    - Each chunk has `text`, `chunk_index`, `page_number`
  - `Page` and `TextChunk` dataclasses defined per blueprint
  - Tests:
    - PDF with 2 pages produces 2 `Page` objects with non-empty PNG bytes
    - PNG file produces 1 `Page` with same bytes
    - TXT file produces 1 `Page` with text content
    - Chunking: 1500-char text with max_chunk_size=512 produces 3 chunks
    - Unknown file type returns empty list

### Task 2.2: Entity Extractor + Pydantic Models

- **Agent:** backend-developer
- **Files to create:**
  - `ingestion/entity_extractor.py`
  - `tests/test_entity_extractor.py`
- **Depends on:** 1.3 (settings for LLM API key/provider)
- **Parallel with:** 2.1
- **Parallel safe:** Yes -- different files, no shared state
- **Creates:** `extract_entities()` function that takes text and returns structured entities + relationships
- **Uses:** LLM API (via provider abstraction), Pydantic models
- **Acceptance criteria:**
  - Pydantic models per blueprint Section 8.1:
    - `Entity(name, type, description)` -- type constrained to `ENTITY_TYPES` from constants
    - `Relationship(source, target, relation, properties)` -- relation constrained to `RELATIONSHIP_TYPES`
    - `ExtractionResult(entities, relationships)`
  - `extract_entities(text, llm_client)` per blueprint Section 8.1:
    - Sends extraction prompt to LLM with JSON response format
    - Parses response into `ExtractionResult` via Pydantic validation
    - On parse failure: retry once with error feedback, then return empty result + log warning (R3 mitigation)
    - Normalizes entity names: strip whitespace, title case
  - LLM client abstraction:
    - Support at minimum Anthropic (default per settings) and OpenAI
    - Simple interface: `async def chat(messages, response_format) -> response`
  - Tests:
    - **Unit test**: mock LLM to return known JSON, verify Pydantic parsing
    - **Unit test**: mock LLM to return malformed JSON, verify retry + fallback
    - **Unit test**: verify entity name normalization ("  google LLC " becomes "Google Llc" or similar)
    - **Integration test** (mark `@pytest.mark.integration`): real LLM call on sample text, verify non-empty extraction

### Task 2.3: Graph Writer (Neo4j Upserts)

- **Agent:** backend-developer
- **Files to create:**
  - `ingestion/graph_writer.py`
  - `tests/test_graph_writer.py`
- **Depends on:** 1.5 (Neo4j schema), 2.2 (entity models)
- **Parallel with:** None (sequential -- needs entity models from 2.2 and schema from 1.5)
- **Parallel safe:** N/A
- **Creates:** `GraphWriter` class for all Neo4j write operations
- **Uses:** `neo4j` async driver, `config.settings`, entity models from 2.2
- **Acceptance criteria:**
  - `GraphWriter` class per blueprint Section 8.2 with all methods:
    - `upsert_document(doc)` -- MERGE on `s3_key`, set all properties
    - `upsert_chunk(chunk_id, text_preview, page_number, s3_key)` -- MERGE Chunk, link to Document via HAS_CHUNK
    - `upsert_entity(entity)` -- MERGE on `(name, type)`, set description, timestamps
    - `upsert_relationship(source, target, relation, properties)` -- APOC merge relationship
    - `link_chunk_to_entity(chunk_id, entity_name, confidence)` -- MERGE MENTIONS relationship
    - `close()` -- close driver
  - Convenience method: `write_extraction_result(s3_key, chunk_id, extraction_result)` that orchestrates all upserts for a single chunk's entities
  - All operations use async sessions
  - Tests (against running Neo4j from docker-compose):
    - Upsert document, verify node exists with correct properties
    - Upsert chunk, verify HAS_CHUNK relationship to document
    - Upsert entity, verify node with correct type
    - Upsert relationship, verify via APOC
    - Idempotency: upsert same document twice, verify single node
    - `write_extraction_result` end-to-end: creates entities, relationships, and MENTIONS links

### Task 2.4: Custom CocoIndex Jina v4 Ops

- **Agent:** backend-developer
- **Files to create:**
  - `ingestion/jina_cocoindex_ops.py`
  - `tests/test_jina_cocoindex_ops.py`
- **Depends on:** 1.6 (Jina embedder)
- **Parallel with:** 2.1, 2.2
- **Parallel safe:** Yes -- standalone bridge module, no shared files
- **Creates:** CocoIndex-compatible operations wrapping Jina v4 API calls
- **Uses:** `JinaV4Embedder` from 1.6, `cocoindex.op.function` decorator
- **Risk:** R4 -- CocoIndex async op support is uncertain
- **Acceptance criteria:**
  - Three custom CocoIndex ops per blueprint Section 7.3:
    - `jina_embed_text(text: str) -> list[float]` -- single-vector text embedding
    - `jina_embed_image(image_bytes: bytes) -> list[float]` -- single-vector image embedding
    - `jina_embed_image_multivec(image_bytes: bytes) -> list[list[float]]` -- ColBERT image embedding
  - Each op wraps the corresponding `JinaV4Embedder` method
  - Handle async-to-sync bridge if CocoIndex ops don't support async natively (R4 mitigation):
    - Try `@cocoindex.op.function()` with `async def` first
    - Fallback: use `asyncio.run()` or `loop.run_until_complete()` in sync wrapper
  - Embedder instance reuse: create once per op call batch (not per individual call)
  - Tests:
    - **Unit test**: mock `JinaV4Embedder`, verify ops call correct methods with correct args
    - **Integration test**: verify ops produce correct vector dimensions (2048 for text/image, list-of-128d for multivec)

### Task 2.5: CocoIndex Pipeline (Local Files Source)

- **Agent:** backend-developer
- **Files to create:**
  - `ingestion/pipeline.py`
- **Modify:**
  - `tests/test_ingestion.py` (create comprehensive pipeline tests)
- **Depends on:** 2.1 (file processor), 2.3 (graph writer), 2.4 (Jina ops)
- **Parallel with:** None -- this is the integration point, requires all dependencies
- **Parallel safe:** N/A
- **Creates:** Complete `rag_ingestion_flow` CocoIndex pipeline definition
- **Uses:** All ingestion submodules, CocoIndex framework, Qdrant targets
- **Risk:** R4 (async ops), R5 (multi-vector Qdrant export)
- **Acceptance criteria:**
  - `rag_ingestion_flow` per blueprint Section 7.1 with these pipeline stages:
    1. Source: local filesystem initially (S3 added in 2.6)
    2. `file_to_pages` transform: MIME classify, convert to pages
    3. For text pages: `semantic_chunk` then `jina_embed_text` per chunk, collect to dense output
    4. For image/PDF pages: `jina_embed_image` for single-vector, collect to dense output
    5. For image/PDF pages: `jina_embed_image_multivec` for ColBERT, collect to multivec output
    6. Entity extraction on text content, write to Neo4j via graph writer
    7. Export dense output to `documents_dense` Qdrant collection
    8. Export multivec output to `documents_multivec` Qdrant collection
  - Point payloads match schemas from blueprint Sections 4.1 and 4.2
  - Pipeline handles empty/corrupt files gracefully (skip with warning)
  - **R5 mitigation:** If CocoIndex Qdrant target does not support multi-vector named vectors:
    - Use CocoIndex for dense collection export (supported natively)
    - Write multi-vector points directly via `QdrantClient.upsert()` in a custom op
  - Test with local directory containing:
    - 1 PDF (2+ pages) -- verify dense + multivec points created, entities in Neo4j
    - 1 PNG image -- verify dense + multivec points created
    - 1 TXT file -- verify dense points created, entities in Neo4j, no multivec points
  - Verify point counts in Qdrant collections match expected
  - Verify Neo4j Document nodes created with correct s3_key/filename

### Task 2.6: S3+SQS Source Integration

- **Agent:** backend-developer
- **Files to modify:**
  - `ingestion/pipeline.py` (swap source from local to S3)
- **Files to create:**
  - `scripts/setup-localstack.sh` (optional, for local testing)
  - `docs/AWS_SETUP.md` (IAM permissions, S3 event notification, SQS policy)
- **Depends on:** 2.5 (working pipeline with local source)
- **Parallel with:** None
- **Parallel safe:** N/A
- **Creates:** Production-ready S3+SQS source for the pipeline
- **Uses:** CocoIndex `AmazonS3` source, `boto3` for verification
- **Risk:** R6 -- external AWS configuration
- **Acceptance criteria:**
  - Pipeline source switched to `cocoindex.sources.AmazonS3` per blueprint Section 7.1:
    - `bucket_name` from settings
    - `sqs_queue_url` from settings (optional -- works without SQS for batch mode)
    - `included_patterns` for supported file types
    - `binary=True` for raw file content
  - `cocoindex update` processes all current files in bucket
  - `cocoindex server -L` enters live mode, picks up new files via SQS
  - `docs/AWS_SETUP.md` documents:
    - Required IAM permissions for S3 read + SQS receive
    - S3 event notification configuration (ObjectCreated, ObjectRemoved)
    - SQS queue policy allowing S3 to send messages
    - Optional: LocalStack setup for local development
  - Test (can be manual or against LocalStack):
    - Upload file to S3, verify ingestion pipeline processes it

### Task 2.7: Ingestion Integration Tests

- **Agent:** backend-developer
- **Files to create:**
  - `tests/test_integration_ingestion.py`
  - `tests/conftest.py` (shared fixtures: embedder mock, test DB connections)
- **Depends on:** 2.5 (pipeline working)
- **Parallel with:** 2.6 (review required -- both touch pipeline)
- **Parallel safe:** Review required -- tests exercise the pipeline that 2.6 modifies, but test file is separate. Run tests after 2.6 merge.
- **Creates:** Comprehensive integration test suite for the ingestion path
- **Uses:** All ingestion modules, running Docker services
- **Acceptance criteria:**
  - `conftest.py` provides:
    - Qdrant client fixture (clears test collections before/after)
    - Neo4j driver fixture (clears test data before/after)
    - Sample file fixtures (PDF, PNG, TXT)
    - Optional: mock Jina embedder that returns deterministic vectors (for fast tests)
  - Test cases:
    - Full pipeline run with sample PDF: verify Qdrant dense points, Qdrant multivec points, Neo4j Document+Chunk+Entity nodes
    - Full pipeline run with sample image: verify both Qdrant collections populated
    - Full pipeline run with sample text: verify Qdrant dense points, Neo4j entities
    - Incremental processing: run pipeline twice on same file, verify no duplicate points
    - Corrupt/empty file: verify pipeline skips gracefully, no partial state left
  - Tests tagged appropriately: `@pytest.mark.integration` for tests requiring Docker services

---

## Phase 3: MCP Server

### Overview

Build the FastMCP server that exposes the four search tools (vector, visual, graph, hybrid) over MCP protocol. Each tool is an independent module that can be developed in parallel once the server scaffold exists. The hybrid tool depends on vector + graph being complete.

### Task 3.1: MCP Server Scaffold + vector_search Tool

- **Agent:** backend-developer
- **Files to create:**
  - `server/mcp_server.py`
  - `server/tools/vector_search.py`
  - `server/tools/__init__.py`
- **Depends on:** 1.4 (Qdrant collections exist), 1.6 (embedder for query vectors)
- **Parallel with:** None (scaffold must come first)
- **Parallel safe:** N/A
- **Creates:** Running MCP server with one functional tool
- **Uses:** `fastmcp`, `qdrant-client`, `JinaV4Embedder`, `config.settings`
- **Acceptance criteria:**
  - `server/mcp_server.py` per blueprint Section 9.1:
    - Creates `FastMCP` instance with name `"rag-knowledge-base"` and description
    - Registers tools via `mcp.tool()` decorator
    - `if __name__ == "__main__": mcp.run(transport=settings.mcp_transport)`
    - Supports both SSE and stdio transport
  - `vector_search` tool per blueprint Section 9.2:
    - Parameters: `query: str`, `limit: int = 10`, `content_type: str | None`, `source_file: str | None`
    - Embeds query using `embed_text_query` (retrieval.query LoRA)
    - Queries `documents_dense` collection with optional filters
    - Returns list of dicts with: score, text, source_file, page_number, content_type, metadata
    - Proper docstring (MCP tools use docstrings as tool descriptions for the agent)
  - Server starts and responds to MCP protocol handshake
  - Tool is discoverable via MCP client `list_tools()`

### Task 3.2: visual_search Tool (ColBERT Multi-Vector)

- **Agent:** backend-developer
- **Files to create:**
  - `server/tools/visual_search.py`
- **Depends on:** 3.1 (server scaffold exists)
- **Parallel with:** 3.3
- **Parallel safe:** Yes -- separate tool file, no shared state with graph_search
- **Creates:** Visual document search tool using ColBERT-style multi-vector retrieval
- **Uses:** `JinaV4Embedder.embed_query_multi_vector()`, Qdrant `documents_multivec` collection
- **Acceptance criteria:**
  - `visual_search` per blueprint Section 9.2:
    - Parameters: `query: str`, `limit: int = 5`
    - Embeds query using `embed_query_multi_vector` (ColBERT mode)
    - Queries `documents_multivec` with `using="colbert"` named vector
    - Returns list of dicts with: score, source_file, page_number, content_type, s3_key, metadata
    - No text_content in results (visual-only retrieval, per blueprint Section 4.2 note)
  - Registered in `mcp_server.py`
  - Discoverable via MCP client

### Task 3.3: graph_search Tool (Neo4j Cypher)

- **Agent:** backend-developer
- **Files to create:**
  - `server/tools/graph_search.py`
- **Depends on:** 1.5 (Neo4j schema), 3.1 (server scaffold)
- **Parallel with:** 3.2
- **Parallel safe:** Yes -- separate tool file, different DB from visual_search
- **Creates:** Knowledge graph search tool with entity lookup and path finding
- **Uses:** `neo4j` async driver, `config.settings`
- **Acceptance criteria:**
  - `graph_search` per blueprint Section 9.2:
    - Parameters: `query: str`, `search_type: str = "entity"`, `limit: int = 10`
    - `search_type="entity"`: full-text search on entity names/descriptions, returns entity + 1-hop connections + source documents
    - `search_type="path"`: shortest path search between entities matching query, returns path nodes + relations + hop count
    - Uses case-insensitive CONTAINS matching (per blueprint Cypher)
    - Properly closes driver after query
  - Registered in `mcp_server.py`
  - Discoverable via MCP client
  - **Note:** Consider connection pooling instead of creating a new driver per call. Use a module-level or dependency-injected driver.

### Task 3.4: hybrid_search Tool (Vector + Graph Fusion)

- **Agent:** backend-developer
- **Files to create:**
  - `server/tools/hybrid_search.py`
- **Depends on:** 3.2 (visual_search), 3.3 (graph_search) -- also needs vector_search from 3.1
- **Parallel with:** None
- **Parallel safe:** N/A
- **Creates:** Combined search tool that runs vector + graph in parallel
- **Uses:** `vector_search`, `graph_search` functions, `asyncio.gather`
- **Acceptance criteria:**
  - `hybrid_search` per blueprint Section 9.2:
    - Parameters: `query: str`, `limit: int = 10`
    - Runs `vector_search` and `graph_search` concurrently via `asyncio.gather`
    - Returns dict with `vector_results`, `graph_results`, `query`, `strategy`
  - Registered in `mcp_server.py`
  - Handles partial failures gracefully: if one search fails, still return results from the other with an error note

### Task 3.5: Server Data Models + Providers Abstraction

- **Agent:** backend-developer
- **Files to create:**
  - `server/models.py`
  - `server/providers.py`
- **Depends on:** 3.1 (server scaffold)
- **Parallel with:** 3.2, 3.3
- **Parallel safe:** Yes -- separate files, no overlap with tool implementations
- **Creates:** Shared Pydantic models for tool responses, LLM provider abstraction
- **Uses:** `pydantic`, `config.settings`
- **Acceptance criteria:**
  - `server/models.py`:
    - `SearchResult` model: score, text, source_file, page_number, content_type, metadata
    - `VisualSearchResult` model: score, source_file, page_number, content_type, s3_key, metadata
    - `GraphEntity` model: entity, type, description, connections, source_documents
    - `HybridSearchResponse` model: vector_results, graph_results, query, strategy
  - `server/providers.py`:
    - LLM provider abstraction per blueprint reference to Cole Medin's Pydantic AI MCP Agent
    - Support for Anthropic, OpenAI, and Ollama providers
    - Selected via `settings.llm_provider`
    - Used by entity extraction and optionally by agent
  - Models are used by tool functions for type safety (tools can still return dicts for MCP serialization, but internally validate via models)

### Task 3.6: MCP Tool Integration Tests

- **Agent:** backend-developer
- **Files to create:**
  - `tests/test_tools.py`
- **Depends on:** 3.4 (all tools complete)
- **Parallel with:** None
- **Parallel safe:** N/A
- **Creates:** Test suite verifying all MCP tools work correctly
- **Uses:** MCP client library (or direct function calls), test data in Qdrant + Neo4j
- **Acceptance criteria:**
  - Test setup: seed Qdrant with known vectors and payloads, seed Neo4j with known entities/relationships
  - `vector_search` tests:
    - Returns results sorted by score descending
    - `content_type` filter works
    - `source_file` filter works
    - Empty results for unrelated query
  - `visual_search` tests:
    - Returns results from multivec collection
    - Results contain s3_key for image retrieval
  - `graph_search` tests:
    - Entity search finds seeded entity by name
    - Entity search returns connections
    - Path search finds shortest path between two seeded entities
  - `hybrid_search` tests:
    - Returns both vector_results and graph_results
    - Both arrays are non-empty when data exists
  - Test the MCP server can be started and tools discovered via MCP protocol client

---

## Phase 4: Agent Integration

### Overview

Build the Pydantic AI agent that connects to the MCP server and intelligently selects which tool(s) to call based on query intent. Optionally expose via a FastAPI HTTP endpoint. Culminates in an end-to-end test from document upload to agent query response.

### Task 4.1: Pydantic AI Agent with MCP Tool Bindings

- **Agent:** fullstack-developer
- **Files to create:**
  - `agent/agent.py`
  - `tests/test_agent.py`
- **Depends on:** 3.4 (all MCP tools registered and working)
- **Parallel with:** 4.2 (but 4.2 also depends on 4.1 -- sequential in practice)
- **Parallel safe:** N/A
- **Creates:** Agent that can answer questions using the RAG MCP tools
- **Uses:** `pydantic-ai`, MCP client connection, `server/providers.py`
- **Acceptance criteria:**
  - Agent per blueprint:
    - Connects to MCP server via SSE or stdio transport
    - Discovers available tools automatically
    - System prompt guides tool selection:
      - Text/semantic questions -> `vector_search`
      - Visual/layout questions -> `visual_search`
      - Relationship/entity questions -> `graph_search`
      - Complex/multi-faceted questions -> `hybrid_search`
    - Streams responses back to caller
  - Per Cole Medin's Pydantic AI MCP Agent reference:
    - MCP client connection code pattern
    - Provider-agnostic LLM usage (configurable via settings)
  - Tests:
    - Agent initializes and connects to MCP server
    - Agent lists available tools (4 tools)
    - Agent responds to a text query (mock LLM + seeded data)
    - Agent correctly selects `visual_search` for image-related query (verify tool call)
    - Agent correctly selects `graph_search` for relationship query (verify tool call)

### Task 4.2: FastAPI Endpoint (Optional HTTP Access)

- **Agent:** fullstack-developer
- **Files to create:**
  - `agent/api.py`
- **Depends on:** 4.1 (agent module)
- **Parallel with:** None
- **Parallel safe:** N/A
- **Creates:** HTTP endpoint for direct agent access without MCP client
- **Uses:** `fastapi`, `uvicorn`, agent from 4.1
- **Acceptance criteria:**
  - `POST /query` endpoint accepting `{"query": "...", "stream": true/false}`
  - Streaming response support via Server-Sent Events
  - Non-streaming response returns complete answer JSON
  - Health check endpoint `GET /health`
  - CORS middleware configured for local development
  - Basic request logging

### Task 4.3: End-to-End Integration Test

- **Agent:** fullstack-developer
- **Files to create:**
  - `tests/test_e2e.py`
- **Depends on:** 4.1 (agent), 2.6 (S3 pipeline) or 2.5 (local pipeline as fallback)
- **Parallel with:** None
- **Parallel safe:** N/A
- **Creates:** Proof that the full system works end-to-end
- **Uses:** All system components
- **Risk:** R6 -- may need LocalStack for S3 simulation
- **Acceptance criteria:**
  - Full flow test:
    1. Start all Docker services
    2. Provision Qdrant collections and Neo4j schema
    3. Place test documents in local source (or LocalStack S3)
    4. Run ingestion pipeline
    5. Start MCP server
    6. Initialize agent
    7. Query agent with text question -> verify relevant answer referencing ingested document
    8. Query agent with visual question -> verify visual search results
    9. Query agent with relationship question -> verify graph entities
  - Test can run in CI with Docker Compose
  - Test cleanup: remove all test data after run

---

## Phase 5: Production Hardening

### Overview

Add resilience, observability, and optional quality improvements. These tasks are independent of each other and can be parallelized freely.

### Task 5.1: Error Handling, Retries, Rate Limiting

- **Agent:** backend-developer
- **Files to modify:**
  - `ingestion/embedder.py` (add retry logic, rate limit handling)
  - `ingestion/entity_extractor.py` (add retry logic)
  - `ingestion/graph_writer.py` (add retry logic)
  - `server/tools/vector_search.py` (add error handling)
  - `server/tools/visual_search.py` (add error handling)
  - `server/tools/graph_search.py` (add error handling)
  - `server/tools/hybrid_search.py` (add partial failure handling)
- **Depends on:** 3.4 (all tools), 2.6 (pipeline complete)
- **Parallel with:** 5.2
- **Parallel safe:** Yes -- modifies existing modules for resilience, 5.2 adds new monitoring modules
- **Creates:** Production-grade error handling across all components
- **Uses:** `tenacity` (retry library), `asyncio` semaphores
- **Acceptance criteria:**
  - Jina v4 API calls:
    - Retry on 429 (rate limit) with exponential backoff
    - Retry on 5xx with backoff, max 3 attempts
    - Raise clear error on 4xx (bad input)
    - Concurrency limiter: max N concurrent API calls (configurable)
  - LLM entity extraction:
    - Retry on parse failure (re-prompt with error message)
    - Timeout per extraction call (configurable, default 30s)
    - Fallback: return empty `ExtractionResult` after max retries
  - Neo4j writes:
    - Retry on transient errors (connectivity, deadlocks)
    - Log permanent failures with full context
  - MCP tools:
    - Return structured error messages (not raw exceptions)
    - Timeout per tool call (configurable)
    - `hybrid_search`: return partial results if one backend fails

### Task 5.2: Monitoring + Observability Hooks

- **Agent:** devops-engineer
- **Files to create:**
  - `config/logging.py`
  - `config/metrics.py` (optional, if using Prometheus/StatsD)
- **Files to modify:**
  - `ingestion/pipeline.py` (add logging + metrics hooks)
  - `server/mcp_server.py` (add request logging)
- **Depends on:** 3.4 (all tools), 2.6 (pipeline)
- **Parallel with:** 5.1
- **Parallel safe:** Yes -- adds new files + logging calls, 5.1 modifies error handling logic
- **Creates:** Structured logging and optional metrics collection
- **Uses:** `structlog` or stdlib `logging`, optionally `prometheus_client`
- **Acceptance criteria:**
  - Structured logging (JSON format for production):
    - Ingestion: log per-file processing (file name, MIME type, page count, entity count, duration)
    - Embedding: log batch sizes, latency, token counts
    - MCP tools: log query, tool name, result count, latency
    - Errors: log with full context (file, stage, error type)
  - Key metrics to track:
    - Ingestion lag: time from S3 event to Qdrant/Neo4j write
    - Query latency: per tool, p50/p95/p99
    - Embedding API: calls per minute, cost tracking (tokens/images)
    - Error rates: per component
  - Log level configurable via environment variable

### Task 5.3: Optional -- Re-ranking Step

- **Agent:** backend-developer
- **Files to create:**
  - `server/tools/reranker.py`
- **Files to modify:**
  - `server/tools/vector_search.py` (add optional rerank step)
  - `server/tools/hybrid_search.py` (add optional rerank step)
- **Depends on:** 3.4 (search tools working)
- **Parallel with:** 5.4
- **Parallel safe:** Yes -- independent feature, different files from VLM generation
- **Creates:** Optional re-ranking of search results using Jina Reranker or cross-encoder
- **Uses:** Jina Reranker API or cross-encoder model
- **Acceptance criteria:**
  - `rerank(query, results, top_k)` function
  - Integrated into `vector_search` and `hybrid_search` as optional post-processing step
  - Enabled/disabled via settings flag (`rerank_enabled: bool = False`)
  - Preserves original scores alongside rerank scores
  - Measurable improvement in result quality on test queries

### Task 5.4: Optional -- VLM Generation for Visual Results

- **Agent:** backend-developer
- **Files to create:**
  - `server/tools/vlm_generator.py`
- **Files to modify:**
  - `server/tools/visual_search.py` (add optional VLM answer generation)
- **Depends on:** 3.2 (visual_search working)
- **Parallel with:** 5.3
- **Parallel safe:** Yes -- independent feature, different files from reranker
- **Creates:** VLM-powered answer generation from retrieved page images
- **Uses:** VLM API (Qwen3-VL, GPT-4o, or Claude with vision), S3 for image retrieval
- **Acceptance criteria:**
  - After visual_search retrieves relevant pages, optionally:
    1. Fetch page images from S3
    2. Send to VLM with the original query
    3. Return VLM-generated answer alongside visual search results
  - Enabled/disabled via settings flag (`vlm_generation_enabled: bool = False`)
  - VLM provider configurable (Anthropic, OpenAI, etc.)
  - Graceful degradation: if VLM fails, still return raw visual search results

---

## Appendix: Key Technical Decisions

### Decision: Jina v4 via HTTP API (not self-hosted)

- **Choice:** Use Jina's hosted embedding API at `api.jina.ai`
- **Alternatives considered:** Self-hosted Jina v4 via `sentence-transformers` or ONNX
- **Rationale:** Simplifies infrastructure (no GPU management), pay-per-use economics for initial deployment, model updates handled by Jina. Self-hosting becomes relevant at scale where API costs exceed GPU amortization.
- **Reference:** Blueprint Section 6, Jina v4 model card noting API vs self-hosted usage

### Decision: Two Qdrant Collections (dense + multivec) Instead of One

- **Choice:** Separate `documents_dense` (single-vector, 2048d) and `documents_multivec` (ColBERT multi-vector, 128d per token)
- **Alternatives considered:** Single collection with multiple named vectors; single collection with only dense vectors
- **Rationale:** Qdrant's multi-vector (ColBERT/MaxSim) retrieval has fundamentally different query mechanics from single-vector cosine search. Separate collections allow independent scaling, different indexing parameters, and clearer payload schemas. The visual search path (multi-vector) serves a distinct use case from text search (dense).
- **Reference:** Blueprint Sections 4.1, 4.2, 4.3

### Decision: CocoIndex for Pipeline Orchestration

- **Choice:** CocoIndex declarative pipeline with S3/SQS source
- **Alternatives considered:** Custom async pipeline with `asyncio`, Prefect/Dagster, LangChain document loaders
- **Rationale:** CocoIndex provides incremental processing (PostgreSQL state tracking), native S3/SQS support, and declarative Qdrant export. Avoids reinventing change detection and idempotency. The main risk is custom op support for Jina v4 API calls (R4).
- **Reference:** Blueprint Section 7, CocoIndex S3+SQS example

### Decision: APOC for Dynamic Relationship Types in Neo4j

- **Choice:** Use `apoc.merge.relationship()` for entity-to-entity relationships with dynamic types
- **Alternatives considered:** Hardcoded Cypher per relationship type, single generic `RELATED_TO` relationship
- **Rationale:** Entity extraction produces diverse relationship types (WORKS_AT, PARTNERS_WITH, PRODUCES, etc.). APOC's dynamic relationship creation keeps the graph writer generic and extensible. Adding new relationship types requires no code changes.
- **Reference:** Blueprint Section 8.2, Neo4j APOC docs

### Decision: Entity Extraction via LLM (Not NER Model)

- **Choice:** Use general-purpose LLM (Claude/GPT) with structured JSON output for entity + relationship extraction
- **Alternatives considered:** spaCy NER, Hugging Face NER models, LangChain's graph extractors
- **Rationale:** LLM extraction captures relationships (not just entities), understands domain context, and produces normalized names. NER models only find entities without relationships. The tradeoff is cost and latency per chunk, mitigated by batching and caching.
- **Reference:** Blueprint Section 8.1, DeepLearning.AI agentic knowledge graph construction course

### Decision: SSE Transport for MCP Server (Not stdio)

- **Choice:** Default to SSE transport with stdio as fallback
- **Alternatives considered:** stdio only, WebSocket
- **Rationale:** SSE allows the MCP server to run as a standalone service accessible over HTTP, enabling multiple agents/clients to connect. stdio works for single-process local setups but limits deployment flexibility. SSE is the standard MCP transport for networked servers.
- **Reference:** Blueprint Section 9.1, Cole Medin's MCP server template

### Decision: AsyncClient Reuse for Jina API

- **Choice:** Share a single `httpx.AsyncClient` instance within the embedder rather than creating one per request (as shown in blueprint)
- **Alternatives considered:** Per-request client creation (blueprint default)
- **Rationale:** Per-request client creation incurs connection setup overhead for every API call. A shared client with connection pooling improves throughput for batch operations during ingestion.
- **Reference:** httpx documentation on client lifecycle, blueprint Section 6.1 (improved upon)
