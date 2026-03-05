# Spektr — RAG-as-MCP-Server — Technical Specification

| Field | Value |
|-|-|
| Project | Spektr |
| Date | 2026-03-02 |
| Status | Draft |
| Source | braindump_final.md |
| Version | 1.0 |

---

## 1. Executive Summary

Spektr is a background knowledge infrastructure system that automatically syncs documents from an AWS S3 bucket into a dual knowledge store — Qdrant for vector embeddings and Neo4j for a temporal knowledge graph (via Graphiti) — and exposes that knowledge to LLM-based agents through an internet-facing MCP server. It enables AI agents to perform semantic search, entity/relationship traversal, and hybrid queries with full temporal awareness, so they can reason about which information is current vs. historical. V1 delivers the complete ingestion pipeline (CocoIndex + Jina v4 + Graphiti), three core search tools (vector, graph, hybrid), API key authentication, and Docker-based infrastructure for Neo4j, Qdrant, and PostgreSQL.

---

## 2. Problem Statement

LLM-based agents lack access to domain-specific, evolving knowledge locked in heterogeneous document collections. Current approaches suffer from two critical gaps:

1. **Static retrieval:** Traditional RAG treats all retrieved information as equally valid. When a document is updated or superseded, agents receive stale facts alongside current ones with no mechanism to distinguish between them.
2. **No reusable infrastructure:** Teams build bespoke retrieval pipelines per project, duplicating effort across document ingestion, embedding, graph construction, and search API plumbing.

The cost of inaction: agents produce answers based on outdated information without knowing it, and every new agent project reinvents the same retrieval infrastructure.

---

## 3. Target Users

### 3.1 LLM Agent (Primary Consumer)

- **Role:** An LLM-based agent (Pydantic AI, Claude Code, custom frameworks) that needs domain knowledge to accomplish tasks
- **Technical Level:** N/A (machine consumer — operates via MCP protocol)
- **Primary Goal:** Retrieve relevant, temporally-aware knowledge to produce accurate, expert-level responses
- **Usage Frequency:** Continuous — programmatic queries during task execution
- **Key Frustrations:** No standard way to access curated domain knowledge; no temporal context on retrieved information

### 3.2 System Operator (Secondary)

- **Role:** Developer who deploys, configures, and maintains the Spektr stack
- **Technical Level:** Advanced
- **Primary Goal:** Stand up the knowledge infrastructure with minimal friction and keep it running
- **Usage Frequency:** Occasional — setup, monitoring, troubleshooting
- **Key Frustrations:** Complex multi-service deployments; unclear configuration; no visibility into pipeline state

---

## 4. User Stories & Acceptance Criteria

### US-001: Automatic Document Ingestion

**As a** system operator, **I want** documents uploaded to S3 to be automatically ingested into the vector store and knowledge graph, **so that** knowledge is available to agents without manual intervention.

**Acceptance Criteria:**
- [ ] A PDF uploaded to S3 is fully indexed (vectors in Qdrant + episodes in Graphiti) within 5 minutes
- [ ] A Markdown or text file uploaded to S3 is fully indexed within 2 minutes
- [ ] PDFs are rendered to page images at 300 DPI and text is extracted
- [ ] Text content is chunked and embedded via Jina v4 as dense single-vectors (2048-dim)
- [ ] Text chunks are fed to Graphiti as episodes with `reference_time` set to the document's modification timestamp
- [ ] Graphiti extracts entities and relationships and stores them in Neo4j with `inserted_at` timestamps
- [ ] Pipeline runs in live update mode (CocoIndex + SQS), processing events in near-real-time
- [ ] Initial bulk load of ~100 files completes without manual intervention
- [ ] Unsupported file types are skipped with a logged warning (see EC-02)

**Notes:** CocoIndex orchestrates the pipeline. Jina v4 embedding requires a custom CocoIndex operation wrapping the HTTP API. Graphiti replaces any custom entity extraction or graph writing — `addEpisode` handles extraction, deduplication, and graph construction internally.

---

### US-002: Semantic Vector Search

**As an** LLM agent, **I want to** search the document corpus by semantic similarity, **so that** I can find relevant passages for factual lookups.

**Acceptance Criteria:**
- [ ] `vector_search` MCP tool accepts a natural language query string
- [ ] Query is embedded using Jina v4 with `retrieval.query` LoRA adapter
- [ ] Results are returned from Qdrant `documents_dense` collection, ranked by cosine similarity
- [ ] Each result includes: text content, source file path, page number, chunk index, relevance score, and ingestion timestamp
- [ ] Optional `content_type` filter (text_chunk, pdf_page, image) narrows results correctly
- [ ] Optional `source_file` filter narrows results to a specific document
- [ ] Response time is under 3 seconds
- [ ] Empty corpus returns `{"results": [], "query": "...", "message": "No documents indexed yet"}` (see EC-05)

**Notes:** The Jina v4 `retrieval.query` adapter produces asymmetric embeddings optimized for query-document matching. Dimensions are 2048 by default (Matryoshka-truncatable).

---

### US-003: Knowledge Graph Search

**As an** LLM agent, **I want to** search the knowledge graph for entities and their relationships, **so that** I can answer relationship-based and multi-hop questions.

**Acceptance Criteria:**
- [ ] `graph_search` MCP tool accepts a query string and a `search_type` parameter ("entity" or "path")
- [ ] Entity search returns matching entities with their type, description, relationships, connected documents, and temporal metadata (`inserted_at`, `invalid_at`)
- [ ] Path search finds multi-hop connections between entities (up to 4 hops)
- [ ] Both currently-valid and historically-invalidated facts are returned — no server-side temporal filtering
- [ ] Results include sufficient temporal metadata for the agent to reason about information currency
- [ ] Response time is under 3 seconds

**Notes:** Graph queries go through Graphiti's search API, not raw Cypher, to respect Graphiti's internal schema. The decision to return all results (including invalidated) is deliberate — agents reason about temporal validity themselves.

---

### US-004: Hybrid Search (Vector + Graph)

**As an** LLM agent, **I want to** run vector and graph search simultaneously with merged results, **so that** I get comprehensive answers that combine semantic similarity with entity relationships.

**Acceptance Criteria:**
- [ ] `hybrid_search` MCP tool accepts a natural language query
- [ ] Vector search and graph search execute in parallel (not sequential)
- [ ] Results from both sources are returned in a single response
- [ ] Each result is clearly attributed to its source ("vector" or "graph")
- [ ] Total response time is under 3 seconds (parallel execution)
- [ ] Both result sets include temporal metadata

**Notes:** This is a v1 must-have. The hybrid tool is the primary entry point for complex queries. No result re-ranking or fusion scoring in v1 — results from both sources are returned as-is.

---

### US-005: Soft Delete on Document Removal

**As a** system operator, **I want** deleted S3 files to be soft-deleted from the knowledge base, **so that** vector search excludes them but the knowledge graph preserves historical context.

**Acceptance Criteria:**
- [ ] S3 delete event triggers removal of associated vectors from Qdrant (both `documents_dense` and `documents_multivec` if populated)
- [ ] S3 delete event triggers Graphiti episode invalidation (`invalid_at` timestamp set) — not hard deletion
- [ ] After deletion, `vector_search` no longer returns chunks from the deleted file
- [ ] After deletion, `graph_search` still returns entities from that file but with `invalid_at` timestamps
- [ ] Entities that were also mentioned in other (non-deleted) documents remain valid
- [ ] Deletion is processed within 5 minutes of the S3 event

**Notes:** Open question: Graphiti may not natively support bulk invalidation by source document. If unsupported, a source→episode mapping layer is needed (see OQ-1).

---

### US-006: Knowledge Supersession

**As an** LLM agent, **I want** the knowledge graph to reflect that newer documents supersede older ones, **so that** I can distinguish current facts from outdated ones.

**Acceptance Criteria:**
- [ ] When a new document introduces facts that contradict an earlier document, Graphiti marks the old facts with `invalid_at` and the new facts with `inserted_at`
- [ ] Agent queries return both old and new facts with their respective timestamps
- [ ] The agent can determine which information is current by comparing timestamps
- [ ] Supersession is automatic — no manual intervention required to mark old facts as outdated
- [ ] A document re-uploaded with the same S3 key but updated content is treated as a knowledge update, not a duplicate

**Notes:** This is Graphiti's core temporal awareness feature. The `reference_time` on each episode (set to document modification timestamp) enables Graphiti's LLM to detect contradictions and manage temporal validity.

---

### US-007: MCP Server Authentication

**As a** system operator, **I want** the MCP server to require API key authentication, **so that** only authorized agents can query the knowledge base.

**Acceptance Criteria:**
- [ ] MCP server validates a Bearer token on every incoming request
- [ ] Requests without a token receive a 401 Unauthorized response
- [ ] Requests with an invalid token receive a 401 Unauthorized response
- [ ] Valid API keys are configured via environment variable (not hardcoded)
- [ ] Authentication check happens before any search tool execution

**Notes:** Simple API key/Bearer token auth for v1. OAuth2/JWT deferred to v2.

---

### US-008: Infrastructure Deployment

**As a** system operator, **I want to** deploy the full stack with a single `docker-compose up`, **so that** I can get the system running quickly.

**Acceptance Criteria:**
- [ ] `docker-compose up` starts Neo4j, Qdrant, and PostgreSQL
- [ ] All service ports are exposed correctly (Qdrant 6333/6334, Neo4j 7474/7687, Postgres 5432)
- [ ] All configuration is done via a single `.env` file
- [ ] `.env.example` documents every required and optional variable
- [ ] Graphiti connects to Neo4j and the configured LLM provider successfully on startup
- [ ] CocoIndex connects to PostgreSQL for state tracking on startup
- [ ] No secrets are hardcoded in source files

---

### US-009: Visual Document Search (Nice-to-have)

**As an** LLM agent, **I want to** search for visually rich content (diagrams, charts, scanned pages) using multi-vector retrieval, **so that** I can find information encoded in visual layouts, not just text.

**Acceptance Criteria:**
- [ ] `visual_search` MCP tool accepts a natural language query
- [ ] Query is embedded using Jina v4 multi-vector (ColBERT) mode
- [ ] Results are returned from Qdrant `documents_multivec` collection using MaxSim scoring
- [ ] Each result includes: source file path, page number, content type, S3 key for image retrieval, and relevance score
- [ ] Response time is under 3 seconds
- [ ] PDF page images are embedded during ingestion at 300 DPI as multi-vectors (128-dim per patch)

**Notes:** Nice-to-have for v1. Include if the multi-vector pipeline (Jina v4 ColBERT + Qdrant multivec collection) doesn't add significant implementation complexity. Can be deferred without blocking core functionality.

---

## 5. User Journeys

### Journey 1: Initial Bulk Ingestion

**Entry Point:** Operator uploads ~100 files to S3 bucket and starts the pipeline
**Persona:** System Operator

**Happy Path:**
1. Operator configures `.env` with S3 bucket, SQS queue, API keys → System validates config on startup
2. Operator runs `docker-compose up` → Neo4j, Qdrant, PostgreSQL start
3. Operator starts CocoIndex in live update mode (`cocoindex server -L`) → Pipeline connects to SQS
4. S3 fires create events for each file → SQS delivers events to CocoIndex
5. For each file: classify by MIME type → extract text/images → chunk → embed via Jina v4 → store in Qdrant
6. For each text chunk: feed to Graphiti as episode → entities and relationships extracted → Neo4j updated
7. Pipeline completes all files → All knowledge searchable via MCP server

**Error Paths:**
- If Jina v4 API is rate-limited: pipeline retries with backoff, processes other files in the meantime (see EC-01)
- If a PDF is corrupted: file is skipped, error logged, other files continue (see EC-02)
- If Graphiti extraction fails for a chunk: vectors are still stored in Qdrant, graph ingestion retried (see EC-03)

---

### Journey 2: Agent Queries Knowledge Base

**Entry Point:** LLM agent connects to MCP server via SSE with Bearer token
**Persona:** LLM Agent

**Happy Path:**
1. Agent connects to MCP server with valid API key → Server authenticates and exposes tools
2. Agent discovers available tools: `vector_search`, `graph_search`, `hybrid_search` (optionally `visual_search`)
3. Agent receives a user question requiring domain knowledge
4. Agent selects appropriate tool based on query intent:
   - Factual lookup ("What is X?") → `vector_search`
   - Relationship query ("How is X connected to Y?") → `graph_search`
   - Complex query ("Tell me everything about X") → `hybrid_search`
5. MCP server executes search, returns results with temporal metadata
6. Agent evaluates `inserted_at`/`invalid_at` timestamps to determine information currency
7. Agent may issue follow-up queries (e.g., vector search reveals entity name → graph search for relationships)

**Error Paths:**
- If no Bearer token: 401 Unauthorized (see EC-08)
- If knowledge base is empty: empty results with explanatory message (see EC-05)

---

### Journey 3: Knowledge Supersession

**Entry Point:** Operator uploads a newer version of a document to S3
**Persona:** System (automatic)

**Happy Path:**
1. New file lands in S3 → SQS event → CocoIndex processes as standard ingestion
2. New text chunks embedded and stored in Qdrant
3. New chunks fed to Graphiti as episodes with newer `reference_time`
4. Graphiti's LLM detects contradictions between new and existing facts
5. Old facts marked with `invalid_at`; new facts stored with `inserted_at`
6. Next agent query returns both old and new facts with timestamps → agent reasons about currency

**Error Paths:**
- If Graphiti fails to detect a contradiction: both old and new facts remain valid. Agent may receive conflicting information but timestamps still indicate which is newer. [ASSUMPTION — verify Graphiti's contradiction detection reliability]

---

## 6. Feature Scope

### 6.1 In Scope (v1)

| Feature | Description | Priority | Related Stories |
|-|-|-|-|
| CocoIndex ingestion pipeline | S3→SQS→classify→chunk→embed→store, live update mode | Must-have | US-001 |
| Jina v4 dense embeddings | Single-vector text embeddings (2048-dim) via API | Must-have | US-001, US-002 |
| Graphiti temporal graph | Entity extraction + temporal tracking via episodes | Must-have | US-001, US-003, US-005, US-006 |
| Qdrant dense collection | `documents_dense` for semantic vector search | Must-have | US-002 |
| Vector search MCP tool | Dense vector search with optional filters | Must-have | US-002 |
| Graph search MCP tool | Entity/relationship search with temporal metadata | Must-have | US-003 |
| Hybrid search MCP tool | Parallel vector + graph with merged results | Must-have | US-004 |
| Soft delete | Qdrant removal + Graphiti invalidation on S3 delete | Must-have | US-005 |
| Knowledge supersession | Graphiti temporal awareness for evolving facts | Must-have | US-006 |
| API key authentication | Bearer token auth on MCP server | Must-have | US-007 |
| FastMCP server (SSE) | Internet-facing MCP protocol server | Must-have | US-002, US-003, US-004 |
| Docker Compose infrastructure | Neo4j, Qdrant, PostgreSQL containers | Must-have | US-008 |
| Pydantic Settings config | All config via `.env` | Must-have | US-008 |
| Jina v4 multi-vector embeddings | ColBERT-style for visual retrieval | Nice-to-have | US-009 |
| Qdrant multivec collection | `documents_multivec` for visual search | Nice-to-have | US-009 |
| Visual search MCP tool | Multi-vector ColBERT search | Nice-to-have | US-009 |

### 6.2 Out of Scope (Deferred)

| Feature | Reason Deferred | Revisit When |
|-|-|-|
| S3 bucket management / upload UI | Not this system's responsibility | Never (by design) |
| Human-facing UI | Consumers are agents, not humans | v2 if monitoring needs arise |
| Re-ranking (Jina Reranker / cross-encoder) | Need baseline quality first | v2 |
| VLM generation for visual results | Depends on visual search being implemented | v2 |
| Pydantic AI consuming agent | MCP server is the product; consuming agent is the consumer's responsibility | v2 (as reference implementation) |
| Monitoring & observability | Important but not v1 core | v2 |
| Rate limiting on MCP server | Depends on real usage patterns | v2 |
| OAuth2 / JWT authentication | API key sufficient for v1 | v2 if multi-tenant |

---

## 7. Success Metrics

| Metric | Target | Measurement Method |
|-|-|-|
| Ingestion latency (PDF) | < 5 minutes from S3 upload to searchable | Timestamp diff: S3 event time vs. first successful search hit |
| Ingestion latency (text/MD) | < 2 minutes from S3 upload to searchable | Same as above |
| Search response time | < 3 seconds for all tool types | Measure MCP tool response time end-to-end |
| Bulk ingestion reliability | 100 files processed without manual intervention | Run bulk upload test, verify all files indexed |
| Temporal accuracy | Newer facts supersede older facts in Graphiti | Upload contradicting documents, verify `invalid_at` set on old facts |
| Auth effectiveness | 0 unauthorized queries succeed | Send requests without/with invalid tokens, verify 401 |
| Soft delete correctness | Deleted files excluded from vector search, preserved in graph | Delete file from S3, verify Qdrant removal and Graphiti invalidation |

---

## 8. Data Model & State

### Key Entities

**Qdrant Points (documents_dense):**
- ID: UUID v4
- Vector: 2048-dim dense embedding (Jina v4)
- Payload: source_file, content_type (text_chunk | pdf_page | image), page_number, chunk_index, text_content, metadata (mime_type, ingested_at, source_key, char_count)

**Qdrant Points (documents_multivec) [Nice-to-have]:**
- ID: UUID v4
- Vector: list of 128-dim vectors (Jina v4 ColBERT mode)
- Payload: source_file, content_type (pdf_page | image | slide), page_number, metadata (mime_type, ingested_at, source_key, image dimensions)

**Graphiti Entities (Neo4j):**
- Managed by Graphiti internally. Episodes, entity nodes, and edges carry `inserted_at` and `invalid_at` timestamps. Entity types include PERSON, ORGANIZATION, PRODUCT, TECHNOLOGY, LOCATION, CONCEPT, EVENT.

**PostgreSQL (CocoIndex state):**
- Pipeline processing metadata — which files have been processed, incremental state tracking. Schema managed by CocoIndex.

### State Transitions

| Entity | States | Trigger |
|-|-|-|
| Qdrant point | exists → removed | S3 file delete event |
| Graphiti episode | valid (inserted_at set) → invalidated (invalid_at set) | Newer contradicting episode or source file deletion |
| Graphiti entity/edge | valid → invalidated | All source episodes invalidated |
| CocoIndex file record | pending → processed → reprocessing | S3 create/update/delete events |

---

## 9. Edge Cases & Error Handling

| ID | Scenario | Expected Behavior | User Feedback | Recovery |
|-|-|-|-|-|
| EC-01 | Jina v4 API unavailable during ingestion | Retry with exponential backoff (3 attempts). On failure, mark file as failed, continue others. | Error logged with file path and failure reason | Failed files retried on next pipeline run |
| EC-02 | Malformed or corrupted PDF | Skip file, log error, continue pipeline | Error logged with file path and parse error | Operator replaces corrupted file in S3 |
| EC-03 | Graphiti LLM extraction failure | Vectors still stored in Qdrant. Graph ingestion retried. If retry fails, logged as graph-incomplete. | Warning: "Vector indexed but graph extraction failed for chunk X" | Re-ingest file or manually trigger Graphiti episode |
| EC-04 | S3 delete event for unknown file | Ignore — no data to invalidate | Debug-level log | None needed |
| EC-05 | Agent queries empty knowledge base | Return empty results with explanatory message | `{"results": [], "message": "No documents indexed yet"}` | Operator ingests documents |
| EC-06 | Very large PDF (100+ pages) | Process all pages; may take longer due to per-page Jina API calls | Progress logged per page [ASSUMPTION — verify CocoIndex per-page logging] | If timeout, pipeline resumes from last processed page on next run |
| EC-07 | Duplicate file upload (same S3 key, same content) | CocoIndex detects no change, skips reprocessing | Log noting skip | None needed |
| EC-08 | Unauthorized MCP request | Return 401, do not execute any search | 401 Unauthorized response | Agent provides valid API key |
| EC-09 | Concurrent ingestion and query | Queries return results from whatever is indexed so far; partial results possible | Results returned normally | Agent re-queries later for completeness |

---

## 10. Non-Functional Requirements

### Performance

- Search response time: < 3 seconds for all tool types (vector, graph, hybrid)
- Ingestion latency: < 5 minutes (PDF), < 2 minutes (text/MD) from S3 event to searchable
- Hybrid search: vector and graph queries must run in parallel, not sequentially

### Security

- All MCP server requests require Bearer token authentication
- All credentials stored in `.env` file, never hardcoded
- HTTPS recommended for internet-exposed MCP server [ASSUMPTION — verify if TLS termination via reverse proxy or application-level]

### Scalability

- Document corpus: hundreds of files (not thousands in v1)
- Update frequency: 1-5 files per day after initial bulk load
- Concurrent agent queries: low volume in v1 [ASSUMPTION — verify expected concurrent query load]

### Reliability

- Pipeline resilient to transient API failures via retries (Jina v4, LLM provider)
- Partial ingestion failures do not block other files
- Knowledge graph data never hard-deleted — Graphiti invalidation preserves history

---

## 11. Constraints & Dependencies

### Technical Constraints

- **Jina v4 is API-only:** Embedding requires HTTP calls to Jina's API. No local model option. Rate limits apply during bulk ingestion.
- **Graphiti requires an LLM:** Entity extraction and temporal reasoning use an external LLM (Anthropic, OpenAI, or Ollama). Cost scales with document volume.
- **CocoIndex needs custom ops for Jina v4:** CocoIndex doesn't natively support Jina v4 API. Custom `@cocoindex.op.function()` wrappers required.
- **Internet-exposed MCP server:** Requires authentication and ideally TLS. Deployment must account for network security.

### External Dependencies

| Dependency | Type | Risk |
|-|-|-|
| Jina v4 Embedding API | External API | Medium — availability and rate limits during bulk ingestion |
| Graphiti | Open-source library | Medium — relatively new, API may evolve |
| CocoIndex | Open-source framework | Medium — custom ops needed |
| Neo4j 5 Community | Infrastructure | Low — mature |
| Qdrant | Infrastructure | Low — mature, native multi-vector support |
| PostgreSQL 17 | Infrastructure | Low — mature, CocoIndex state only |
| AWS S3 + SQS | Cloud service | Low — well-established |
| LLM provider (Anthropic/OpenAI/Ollama) | External API | Low — used by Graphiti, provider-agnostic |

---

## 12. Open Questions

| # | Question | Context | Owner | Deadline |
|-|-|-|-|-|
| OQ-1 | Does Graphiti support bulk invalidation by source document? | Critical for US-005 (soft delete). If not, need a source→episode mapping layer. | Developer | Phase 1 (before ingestion pipeline) |
| OQ-2 | What is Graphiti's exact Neo4j schema? | Determines if raw Cypher queries are safe alongside Graphiti or if we must use Graphiti's search API exclusively. | Developer | Phase 1 |
| OQ-3 | Should TLS termination be reverse proxy or application-level? | MCP server is internet-exposed. Affects deployment architecture. | Operator | Before deployment |
| OQ-4 | How does CocoIndex handle partial re-ingestion of large PDFs? | If ingestion fails mid-PDF, does it resume from last page or restart? Affects reliability for large files. | Developer | Phase 2 |
| OQ-5 | Which LLM should Graphiti use for entity extraction? | Cost vs. quality tradeoff. Anthropic Claude Sonnet for quality, Ollama for cost. | Developer / Operator | Phase 1 |
| OQ-6 | What is Graphiti's contradiction detection reliability? | If Graphiti misses a contradiction, both old and new facts remain valid. Affects US-006 quality. | Developer | Phase 2 (integration testing) |

---

## 13. Definition of Done

- [ ] All must-have user stories (US-001 through US-008) implemented and passing acceptance criteria
- [ ] Edge cases EC-01 through EC-09 handled as specified
- [ ] `docker-compose up` starts all infrastructure services successfully
- [ ] Initial bulk load of ~100 files completes without manual intervention
- [ ] All three core MCP tools (vector_search, graph_search, hybrid_search) return results with temporal metadata
- [ ] Knowledge supersession verified: uploading contradicting documents results in proper `invalid_at` / `inserted_at` timestamps
- [ ] Soft delete verified: removing a file from S3 removes vectors from Qdrant and invalidates Graphiti episodes
- [ ] Bearer token authentication rejects all unauthorized requests
- [ ] MCP server accessible over the internet via SSE transport
- [ ] All configuration via `.env` file — no hardcoded secrets
- [ ] Open questions OQ-1 and OQ-2 resolved (Graphiti compatibility)
- [ ] Code reviewed and merged
