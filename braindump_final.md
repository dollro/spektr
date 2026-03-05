# Feature Specification: Spektr — RAG-as-MCP-Server

| Field | Value |
|-|-|
| Project | Spektr |
| Date | 2026-03-02 |
| Status | Draft |
| Author | - |
| Version | 1.0 |

---

## 1. Executive Summary

Spektr is a hybrid GraphRAG + multimodal vector search system that automatically syncs documents from an S3 bucket into a dual knowledge store (vector database + temporal knowledge graph) and exposes that knowledge to LLM-based agents via an MCP server. It solves the problem of AI agents lacking access to domain-specific, evolving knowledge locked in heterogeneous document collections. The primary consumers are LLM agents and tools that need fine-grained, expert-level knowledge to perform their tasks. V1 delivers the full ingestion pipeline, three core search tools (vector, graph, hybrid) with temporal awareness, and an internet-facing MCP server with API key authentication.

---

## 2. Problem Statement

**Who is affected:** Developers and teams building LLM-based agents that need to reason over a body of domain-specific documents — technical documentation, research papers, books.

**The current pain:** LLM agents operate with either their training data (stale, generic) or naive RAG implementations that treat all retrieved information as equally current and valid. When a document is updated or superseded by newer information, traditional RAG systems return stale facts alongside current ones with no way to distinguish between them. Agents cannot reason about the temporal validity of knowledge.

**The cost of inaction:** Agents produce answers based on outdated information without knowing it. Teams manually curate knowledge bases or build bespoke retrieval pipelines per project. There's no reusable infrastructure that handles document ingestion, knowledge graph construction with temporal awareness, and agent-accessible search in a unified stack.

**Evidence:** The pattern demonstrated by Cole Medin's agentic RAG knowledge graph demo and the Graphiti library shows that temporal knowledge graphs significantly improve an agent's ability to reason about evolving information — newer facts automatically supersede older ones while preserving historical context.

---

## 3. Target Users & Personas

### Persona 1: LLM Agent (Primary Consumer)

- **Description:** An LLM-based agent (Pydantic AI, Claude Code, custom agent frameworks) that needs to retrieve domain knowledge to accomplish tasks. The agent connects to Spektr's MCP server as a tool.
- **Technical skill level:** N/A (machine consumer — operates via MCP protocol)
- **Frequency of use:** Continuous — queries are made programmatically during agent task execution
- **Key needs:** Fast, relevant search results with temporal metadata so the agent can reason about information currency. Must support semantic search, relationship queries, and hybrid search.

### Persona 2: System Operator (Secondary)

- **Description:** The developer who deploys, configures, and maintains the Spektr stack. Manages the S3 bucket, monitors the ingestion pipeline, and configures agent access.
- **Technical skill level:** Advanced
- **Frequency of use:** Occasional — setup, monitoring, troubleshooting
- **Key needs:** Simple deployment (Docker Compose), clear configuration (.env), observable pipeline state. S3 document management is out of scope — the operator uses their own tools for that.

---

## 4. Core Use Cases

### UC-1: Automatic Document Ingestion

**Actor:** System (triggered by S3 events)
**Trigger:** A file is uploaded to or modified in the S3 bucket
**Preconditions:** CocoIndex pipeline is running in live update mode; SQS queue is configured for S3 event notifications
**Main Flow:**
1. S3 emits a create/update event to the SQS queue
2. CocoIndex detects the event and fetches the file
3. File is classified by MIME type (PDF, Markdown, text, image)
4. PDFs are rendered to page images at 300 DPI; text is extracted
5. Text content is chunked and embedded via Jina v4 (dense single-vector)
6. Page images are embedded via Jina v4 (multi-vector ColBERT mode) [Nice-to-have]
7. Dense vectors are stored in Qdrant `documents_dense` collection
8. Multi-vectors are stored in Qdrant `documents_multivec` collection [Nice-to-have]
9. Text chunks are fed to Graphiti as episodes with `reference_time` set to the document's modification timestamp
10. Graphiti extracts entities and relationships, builds/updates the temporal knowledge graph in Neo4j
**Postconditions:** Document content is searchable via all MCP tools; knowledge graph reflects the new information with temporal metadata
**Acceptance Criteria:**
- [ ] A PDF uploaded to S3 is fully indexed (vectors + graph) within 5 minutes
- [ ] A Markdown file uploaded to S3 is fully indexed within 2 minutes
- [ ] Entities extracted from the new document appear in Neo4j with correct `inserted_at` timestamps
- [ ] Existing entities updated by the new document have proper temporal records (old facts marked with `invalid_at`)

### UC-2: Semantic Vector Search

**Actor:** LLM Agent
**Trigger:** Agent calls the `vector_search` MCP tool
**Preconditions:** MCP server is running; agent has a valid API key; documents have been ingested
**Main Flow:**
1. Agent sends a natural language query with optional filters (content_type, source_file)
2. MCP server embeds the query using Jina v4 with `retrieval.query` LoRA adapter
3. Query vector is searched against Qdrant `documents_dense` collection
4. Results are returned with text content, source file, page number, relevance score, and temporal metadata
**Postconditions:** Agent receives ranked list of relevant document chunks
**Acceptance Criteria:**
- [ ] Search returns results within 3 seconds
- [ ] Results include text content, source file path, page number, and relevance score
- [ ] Optional content_type filter correctly narrows results
- [ ] Optional source_file filter correctly narrows results
- [ ] Results include ingestion timestamp metadata

### UC-3: Knowledge Graph Search

**Actor:** LLM Agent
**Trigger:** Agent calls the `graph_search` MCP tool
**Preconditions:** MCP server is running; agent has a valid API key; entities have been extracted into Neo4j via Graphiti
**Main Flow:**
1. Agent sends a query about entities or relationships (e.g., "What technologies does company X use?")
2. MCP server queries the Graphiti-managed Neo4j graph
3. Results include entities, their relationships, connected documents, and temporal metadata (`inserted_at`, `invalid_at`)
4. Both currently-valid and historical results are returned — the agent decides what's relevant
**Postconditions:** Agent receives entity/relationship data with full temporal context
**Acceptance Criteria:**
- [ ] Entity search by name returns matching entities with their relationships
- [ ] Results include `inserted_at` and `invalid_at` timestamps for temporal reasoning
- [ ] Historical (invalidated) facts are included in results, not filtered out
- [ ] Path search finds multi-hop connections between entities
- [ ] Search returns results within 3 seconds

### UC-4: Hybrid Search (Vector + Graph)

**Actor:** LLM Agent
**Trigger:** Agent calls the `hybrid_search` MCP tool
**Preconditions:** MCP server is running; agent has a valid API key; both Qdrant and Neo4j are populated
**Main Flow:**
1. Agent sends a natural language query
2. MCP server runs vector search and graph search in parallel
3. Results from both sources are merged and returned with source attribution
4. Agent receives a combined result set with both document chunks and entity relationships
**Postconditions:** Agent receives comprehensive results spanning both semantic similarity and graph relationships
**Acceptance Criteria:**
- [ ] Both vector and graph results are returned in a single response
- [ ] Results are clearly attributed to their source (vector vs. graph)
- [ ] Total response time is under 3 seconds (parallel execution)
- [ ] Results include temporal metadata from both sources

### UC-5: Document Deletion (Soft Delete)

**Actor:** System (triggered by S3 events)
**Trigger:** A file is deleted from the S3 bucket
**Preconditions:** The file was previously ingested; CocoIndex pipeline is running
**Main Flow:**
1. S3 emits a delete event to the SQS queue
2. CocoIndex detects the event and identifies the deleted file
3. Qdrant points associated with the deleted file are removed from both collections
4. Graphiti episodes originating from the deleted file are invalidated (marked with `invalid_at` timestamp) — not hard-deleted
5. Entity relationships derived solely from the deleted document are invalidated
**Postconditions:** File's vector data is removed from Qdrant; graph knowledge is preserved as historical with invalidation timestamps
**Acceptance Criteria:**
- [ ] After file deletion, vector search no longer returns chunks from that file
- [ ] After file deletion, graph search still returns entities from that file but with `invalid_at` timestamps
- [ ] Entities that were also mentioned in other documents remain valid
- [ ] The deletion is processed within 5 minutes of the S3 event

---

## 5. Feature Scope

### 5.1 In Scope (v1)

| Feature | Description | Priority | Use Case |
|-|-|-|-|
| CocoIndex ingestion pipeline | S3→SQS→classify→chunk→embed→store, with live update mode | Must-have | UC-1, UC-5 |
| Jina v4 dense embeddings | Single-vector text + image embeddings for Qdrant | Must-have | UC-1, UC-2 |
| Graphiti temporal graph | Entity extraction, relationship building, temporal tracking via Graphiti episodes | Must-have | UC-1, UC-3, UC-5 |
| Qdrant dense collection | `documents_dense` for semantic vector search | Must-have | UC-2 |
| Vector search MCP tool | Dense vector search with optional filters | Must-have | UC-2 |
| Graph search MCP tool | Entity/relationship search with temporal metadata | Must-have | UC-3 |
| Hybrid search MCP tool | Parallel vector + graph search with merged results | Must-have | UC-4 |
| Soft delete via Graphiti | Invalidate graph knowledge on S3 file deletion, remove vectors from Qdrant | Must-have | UC-5 |
| API key authentication | Bearer token auth on the MCP server for internet-exposed deployment | Must-have | All |
| FastMCP server (SSE) | MCP protocol server with SSE transport | Must-have | All |
| Docker Compose infrastructure | Neo4j, Qdrant, PostgreSQL in containers | Must-have | All |
| Pydantic Settings config | All configuration via `.env` file | Must-have | All |
| Jina v4 multi-vector embeddings | ColBERT-style multi-vector for visual document retrieval | Nice-to-have | UC-1 |
| Qdrant multivec collection | `documents_multivec` for visual search | Nice-to-have | UC-1 |
| Visual search MCP tool | Multi-vector ColBERT search for diagrams, charts, scanned pages | Nice-to-have | — |

### 5.2 Explicitly Out of Scope (Deferred)

| Feature | Why Deferred | Revisit |
|-|-|-|
| S3 bucket management / file upload UI | Not this system's responsibility — users manage S3 with their own tools | Never (by design) |
| Human-facing UI (chat, dashboard) | Consumers are agents, not humans | v2 if monitoring needs arise |
| Re-ranking (Jina Reranker / cross-encoder) | Optimization — need baseline quality first | v2 |
| VLM generation for visual results (Qwen3-VL / GPT-4o) | Depends on visual search being implemented first | v2 |
| Pydantic AI agent | The MCP server is the product; building a specific consuming agent is the consumer's responsibility | v2 (as reference implementation) |
| Monitoring & observability | Important but not v1 core functionality | v2 |
| Rate limiting on MCP server | Depends on real usage patterns | v2 |
| OAuth2 / JWT authentication | API key auth is sufficient for v1 | v2 if multi-tenant |

---

## 6. User Journeys & UX Flows

Since the system has no human-facing UI, user journeys describe **system flows** and **agent interaction patterns**.

### Journey 1: Initial Bulk Ingestion

**Entry point:** Operator uploads ~100 files to S3 bucket
**Flow:**
1. S3 fires create events for each file → SQS queue
2. CocoIndex picks up events and processes files sequentially/in batches
3. Each file: classify → extract text/images → chunk → embed via Jina v4 → store in Qdrant
4. Each text chunk: feed to Graphiti as episode → entities and relationships extracted → Neo4j updated
5. Operator can monitor progress via CocoIndex state in PostgreSQL
**Exit point:** All files indexed, knowledge graph populated, MCP server ready for queries

### Journey 2: Agent Queries Knowledge Base

**Entry point:** LLM agent connects to MCP server via SSE with Bearer token
**Flow:**
1. Agent discovers available tools (vector_search, graph_search, hybrid_search, optionally visual_search)
2. Based on query intent, agent selects appropriate tool
3. For factual lookups → `vector_search` (semantic similarity)
4. For relationship questions → `graph_search` (entity traversal with temporal context)
5. For complex questions → `hybrid_search` (parallel vector + graph)
6. Agent receives results with temporal metadata and reasons about information currency
**Decision point:** Agent may issue follow-up queries based on initial results (e.g., vector search finds a relevant entity name → graph search to explore its relationships)
**Exit point:** Agent has sufficient knowledge to complete its task

### Journey 3: Knowledge Update (Supersession)

**Entry point:** Operator uploads a newer version of a document to S3
**Flow:**
1. CocoIndex processes the new file as a standard ingestion
2. Jina v4 embeddings for new content are stored in Qdrant (old vectors from the previous version remain or are updated based on CocoIndex's incremental tracking)
3. Graphiti receives new text chunks as episodes with a newer `reference_time`
4. Graphiti's LLM detects that new facts contradict or update existing ones
5. Old facts are marked with `invalid_at` timestamp; new facts get `inserted_at` timestamp
6. Next agent query returns both old and new facts with timestamps — agent reasons about which is current
**Exit point:** Knowledge graph reflects the evolution of information

---

## 7. Data Model & State Management

### Key Entities and Their Storage

**Qdrant — Vector Store:**
- `documents_dense` collection: dense single-vector embeddings (2048-dim, Jina v4) for text chunks and images. Each point carries payload: source_file, content_type, page_number, chunk_index, text_content, metadata (mime_type, ingested_at, source_key).
- `documents_multivec` collection [Nice-to-have]: multi-vector ColBERT embeddings (128-dim per token) for visually rich document pages.

**Neo4j — Temporal Knowledge Graph (managed by Graphiti):**
- Graphiti manages its own schema internally. Core concepts:
  - **Episodes:** Raw data units (text chunks) with `reference_time`
  - **Entity nodes:** Extracted entities with types, descriptions, temporal metadata
  - **Edge relationships:** Connections between entities with temporal validity
  - All nodes and edges carry `inserted_at` and `invalid_at` timestamps

**PostgreSQL — Pipeline State:**
- CocoIndex state tracking only. Tracks which files have been processed, incremental processing metadata.

### Data Ownership

| Data | Created by | Read by | Updated by | Deleted by |
|-|-|-|-|-|
| S3 files | External (out of scope) | CocoIndex pipeline | External | External |
| Qdrant vectors | CocoIndex + Jina v4 | MCP search tools | CocoIndex (on file update) | CocoIndex (on file delete) |
| Neo4j graph | Graphiti | MCP search tools | Graphiti (on new episodes) | Never hard-deleted; Graphiti invalidates |
| PostgreSQL state | CocoIndex | CocoIndex | CocoIndex | CocoIndex |

### State Transitions

- **S3 file:** exists → deleted (triggers Qdrant removal + Graphiti invalidation)
- **Qdrant point:** created on ingestion → removed on file deletion
- **Graphiti entity/edge:** created (inserted_at) → optionally invalidated (invalid_at set) when superseded by newer information → never deleted
- **CocoIndex state:** tracks per-file processing status for incremental updates

### External Data

- Jina v4 API (external) — embedding generation
- LLM provider (external) — used by Graphiti for entity extraction (Anthropic, OpenAI, or Ollama)
- AWS S3 + SQS (external) — document source and event notification

---

## 8. Edge Cases & Error Handling

### EC-1: Jina v4 API Unavailable During Ingestion

**Scenario:** Jina embedding API returns errors or times out during document processing
**Expected behavior:** CocoIndex retries the embedding call with exponential backoff (up to 3 retries). If all retries fail, the file is marked as failed in pipeline state and skipped.
**User communication:** Error logged with file path and failure reason
**Recovery path:** Failed files are retried on next pipeline run or can be manually re-triggered

### EC-2: Malformed or Corrupted PDF

**Scenario:** A PDF file in S3 cannot be parsed or rendered to images
**Expected behavior:** File is classified as unprocessable. Pipeline logs the error and continues with other files.
**User communication:** Error logged with file path and parse error details
**Recovery path:** Operator replaces the corrupted file in S3; pipeline processes the replacement

### EC-3: Graphiti LLM Extraction Failure

**Scenario:** The LLM used by Graphiti for entity extraction fails (rate limit, timeout, API error)
**Expected behavior:** The text chunk's vector embedding is still stored in Qdrant (vector search works). Graph ingestion for that chunk is retried. If retry fails, chunk is logged as graph-incomplete.
**User communication:** Warning logged — "Vector indexed but graph extraction failed for chunk X"
**Recovery path:** Re-ingest the file or manually trigger Graphiti episode creation for the failed chunks

### EC-4: S3 Delete Event for Unknown File

**Scenario:** A delete event arrives for a file that was never ingested (e.g., temp file, unsupported format)
**Expected behavior:** Pipeline ignores the event — no Qdrant points or Graphiti episodes to invalidate
**User communication:** Debug-level log noting the unknown file
**Recovery path:** None needed

### EC-5: Empty Knowledge Base (First Query)

**Scenario:** An agent queries the MCP server before any documents have been ingested
**Expected behavior:** Search tools return empty result sets with appropriate metadata (e.g., `{"results": [], "query": "...", "message": "No documents indexed yet"}`)
**User communication:** Empty results with explanatory message in the response
**Recovery path:** Agent handles empty results gracefully; operator ingests documents

### EC-6: Very Large PDF (100+ Pages)

**Scenario:** A large book or report PDF is uploaded to S3
**Expected behavior:** Pipeline processes all pages but may take significantly longer due to Jina v4 API calls per page. CocoIndex handles this incrementally.
**User communication:** Progress logged per page [ASSUMPTION — verify CocoIndex provides per-page progress logging]
**Recovery path:** If timeout occurs, pipeline should resume from the last successfully processed page on next run

### EC-7: Duplicate File Upload

**Scenario:** The same file is uploaded to S3 twice with the same key
**Expected behavior:** CocoIndex detects no change (same content hash) and skips reprocessing. If the content differs (same key, new content), it's treated as an update — re-embed and feed new episodes to Graphiti.
**User communication:** Log noting skip (unchanged) or update (changed content)
**Recovery path:** None needed — handled automatically

### EC-8: Unauthorized MCP Request

**Scenario:** An agent sends a request without a valid Bearer token
**Expected behavior:** MCP server returns authentication error; request is not processed
**User communication:** 401 Unauthorized response
**Recovery path:** Agent must provide a valid API key

### EC-9: Concurrent Ingestion and Query

**Scenario:** An agent queries while a large batch ingestion is in progress
**Expected behavior:** Queries return results from whatever has been indexed so far. Partially-ingested documents may appear (some chunks indexed, others pending).
**User communication:** Results returned normally — no indication of ongoing ingestion [ASSUMPTION — verify this is acceptable vs. marking results as "ingestion in progress"]
**Recovery path:** Agent can re-query later for more complete results

---

## 9. Acceptance Criteria

### Ingestion Pipeline

- [ ] PDF files are classified, rendered to images, text extracted, chunked, and embedded
- [ ] Markdown and text files are chunked and embedded
- [ ] All text chunks are stored as dense vectors in Qdrant `documents_dense`
- [ ] All text chunks are fed to Graphiti as episodes with correct `reference_time`
- [ ] Graphiti extracts entities and relationships and stores them in Neo4j
- [ ] S3 file deletion triggers: Qdrant point removal + Graphiti episode invalidation
- [ ] Updated files (same S3 key, new content) are re-processed; Graphiti handles temporal supersession
- [ ] Pipeline runs in live update mode via CocoIndex + SQS
- [ ] Initial bulk load of ~100 files completes without manual intervention

### MCP Server & Search Tools

- [ ] `vector_search` returns ranked text chunks with scores, metadata, and timestamps
- [ ] `graph_search` returns entities with relationships, connected documents, and temporal metadata (`inserted_at`, `invalid_at`)
- [ ] `hybrid_search` runs vector + graph in parallel and returns merged results within 3 seconds
- [ ] All search results include temporal metadata — agents receive both current and historical facts
- [ ] MCP server uses SSE transport and is internet-accessible
- [ ] Bearer token authentication rejects unauthorized requests
- [ ] Empty queries return empty results, not errors

### Infrastructure

- [ ] `docker-compose up` starts Neo4j, Qdrant, and PostgreSQL
- [ ] All configuration via `.env` file (no hardcoded secrets)
- [ ] Graphiti connects to Neo4j and LLM provider successfully on startup

---

## 10. Non-Functional Requirements

### Performance

- Search tool response time: under 3 seconds for first result
- Ingestion latency: files available for search within 5 minutes of S3 upload (PDF), 2 minutes (text/markdown)
- Bulk ingestion: ~100 files processed without manual intervention (total time depends on Jina API throughput)

### Security

- MCP server authentication: API key / Bearer token on all requests
- No secrets in code — all credentials in `.env` file
- MCP server exposed to internet — HTTPS recommended for production [ASSUMPTION — verify if TLS termination is handled by a reverse proxy or the MCP server itself]

### Scalability

- Document corpus: hundreds of files (not thousands in v1)
- Incremental updates: 1-5 files per day typical after initial load
- Concurrent agent queries: low volume expected in v1 [ASSUMPTION — verify expected concurrent query load]

### Reliability

- Pipeline should be resilient to transient API failures (Jina, LLM) via retries
- Partial ingestion failures should not block other files
- Knowledge graph data is never hard-deleted — soft delete via Graphiti invalidation

---

## 11. Risks, Dependencies & Open Questions

### Risks

| Risk | Likelihood | Impact | Mitigation |
|-|-|-|-|
| Graphiti doesn't support bulk invalidation by source document | Medium | High | Verify early; if unsupported, build a thin wrapper that tracks episode-to-source mapping and invalidates individually |
| CocoIndex custom async ops for Jina v4 may need sync wrappers | Medium | Medium | Test early in Phase 1; use `asyncio.run()` wrapper if needed |
| Jina v4 ColBERT multi-vector response format undocumented | Medium | Low | Nice-to-have feature; verify API response shape with a test call before building the pipeline |
| Graphiti's internal Neo4j schema may conflict with custom queries | Low | Medium | Use Graphiti's search API exclusively; avoid raw Cypher on Graphiti-managed data |
| Jina v4 API rate limits during bulk ingestion of ~100 files | Medium | Medium | Implement batching and backoff; consider requesting rate limit increase |
| CocoIndex + Graphiti integration complexity | Medium | High | These are two independent pipeline systems that need to work together; prototype the integration point early |

### Dependencies

| Dependency | Type | Risk Level |
|-|-|-|
| Jina v4 Embedding API | External API | Medium — API availability and rate limits |
| Graphiti library | Open-source library | Medium — relatively new library, API may change |
| CocoIndex | Open-source framework | Medium — custom ops needed for Jina v4 |
| Neo4j 5 Community | Infrastructure | Low — mature, well-documented |
| Qdrant | Infrastructure | Low — mature, native multi-vector support |
| PostgreSQL 17 | Infrastructure | Low — mature, only used for CocoIndex state |
| AWS S3 + SQS | Cloud service | Low — well-established AWS services |
| LLM provider (Anthropic/OpenAI/Ollama) | External API | Low — used by Graphiti, provider-agnostic |

### Open Questions

| Question | Context | Proposed Next Step |
|-|-|-|
| Does Graphiti support invalidating all episodes from a specific source? | Critical for UC-5 (soft delete on S3 file removal). If not, we need a source→episode tracking layer. | Review Graphiti API docs and source code in Phase 1 |
| What is Graphiti's exact Neo4j schema? | Affects whether we can run custom Cypher queries alongside Graphiti or must use its API exclusively. | Read Graphiti docs; if schema is internal, use only Graphiti's search API in the graph_search tool |
| Should the MCP server handle TLS termination? | Server is internet-exposed. Could use a reverse proxy (nginx, Caddy) or handle TLS in the app. | Decide during deployment setup — likely reverse proxy |
| How does CocoIndex handle partial re-ingestion of a large PDF? | If ingestion fails mid-PDF (e.g., Jina API timeout on page 50 of 100), does it resume from page 50 or restart? | Test with CocoIndex during Phase 2 |
| What LLM should Graphiti use for entity extraction? | Graphiti supports OpenAI, Anthropic, Ollama. Cost vs. quality tradeoff. | Start with Anthropic (Claude Sonnet) for quality; consider Ollama for cost reduction later |

---

## 12. Appendix

### Glossary

| Term | Definition |
|-|-|
| MCP | Model Context Protocol — standard protocol for LLM tools and context |
| GraphRAG | RAG approach combining vector retrieval with knowledge graph traversal |
| ColBERT | Multi-vector retrieval model using late interaction (MaxSim scoring) |
| Graphiti | Open-source library for building temporal-aware knowledge graphs on Neo4j |
| Episode | Graphiti's primary data unit — a piece of text with a reference timestamp |
| CocoIndex | Declarative data pipeline framework with incremental processing support |
| Dense vector | Single fixed-dimension embedding (e.g., 2048-dim from Jina v4) |
| Multi-vector | Multiple embeddings per document (e.g., one per image patch), enabling fine-grained matching |
| SSE | Server-Sent Events — transport protocol used by the MCP server |
| Soft delete | Marking data as invalid/archived rather than permanently removing it |

### Key References

- [Architecture Blueprint](rag-mcp-architecture-blueprint.md) — detailed technical design
- [Cole Medin — Agentic RAG Knowledge Graph](https://github.com/coleam00/ottomator-agents/tree/main/agentic-rag-knowledge-graph) — architecture pattern reference
- [Cole Medin — MCP Server Template](https://github.com/coleam00/mcp-mem0) — FastMCP scaffolding
- [CocoIndex — Multi-Format Indexing](https://cocoindex.io/examples/multi_format_index) — pipeline reference
- [CocoIndex — S3 + SQS Pipeline](https://cocoindex.io/examples/s3_sqs_pipeline) — S3 source config
- [Graphiti — Temporal Knowledge Graphs](https://www.youtube.com/watch?v=PxcOIINgiaA) — Cole Medin's demo of temporal knowledge updates
- [Jina v4 Model Card](https://huggingface.co/jinaai/jina-embeddings-v4) — embedding model reference

### Decision Log

| Decision | Rationale |
|-|-|
| Graphiti replaces custom entity extraction + graph writing | Graphiti handles extraction, deduplication, graph construction, and temporal tracking out of the box. Eliminates need for custom `entity_extractor.py` and `graph_writer.py`. |
| Soft delete (invalidation) over hard delete | Preserves historical knowledge. Agents receive temporal metadata and reason about currency themselves. |
| Agent-side temporal reasoning (not server-side filtering) | Maximum flexibility — agents get all results with `inserted_at`/`invalid_at` and decide what's relevant. Supports historical queries. |
| API key / Bearer token auth for v1 | Simple, sufficient for known agent consumers. OAuth2/JWT deferred to v2 if multi-tenant needs arise. |
| Visual search (multi-vector) as nice-to-have | Core value is text semantic + graph search. Multi-vector ColBERT adds complexity; include if straightforward, defer if not. |
| No human-facing UI | System is pure infrastructure for agents. No chat interface, no dashboard in v1. |
| Standard .env for secrets | Sufficient for v1 scale. AWS Secrets Manager deferred to production hardening. |
