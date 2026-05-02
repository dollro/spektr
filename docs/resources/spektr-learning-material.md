# Spektr Deep Dive — Understanding Every Layer of the Stack

*A 2-hour reading guide for understanding what we're building and why each piece exists.*

---

## Table of Contents

1. The Problem We're Solving
2. RAG Fundamentals — What Happens Under the Hood
3. Why Embeddings Matter — From Words to Vectors
4. Jina v4 — One Model to Rule Them All
5. Single-Vector vs Multi-Vector — The ColBERT Revolution
6. Qdrant — Why a Dedicated Vector Database
7. Knowledge Graphs — Why Vectors Aren't Enough
8. Neo4j — Graph Database Mechanics
9. GraphRAG — Combining Graphs and Vectors
10. CocoIndex — Declarative Data Pipelines
11. S3 + SQS — Event-Driven Ingestion
12. MCP Protocol — Making RAG a Tool for Agents
13. FastMCP — The Server Implementation
14. Pydantic AI — The Agent Framework
15. How It All Connects — End-to-End Request Flow
16. Production Considerations

---

## 1. The Problem We're Solving

Imagine you have a growing collection of documents: quarterly financial reports as PDFs, scanned invoices, architecture diagrams, project notes in markdown, presentation slides. Some are text-heavy, some are scanned images with tables and charts that no OCR tool can reliably parse.

You want an AI agent to be able to answer questions like:

- "What were our Q3 revenue figures?" → needs to find a text passage in a PDF
- "Show me the org chart from the onboarding deck" → needs to find a specific diagram in a slide deck
- "Which suppliers does Acme Corp work with, and what products do they provide?" → needs to traverse entity relationships across multiple documents
- "Find the table comparing cloud providers from last month's architecture review" → needs to understand visual layout, not just text

Traditional RAG (stick text chunks in a vector database, search by similarity) handles the first query well but fails at the rest. You need three capabilities working together:

**Vector search** for semantic similarity ("what text passages are relevant?"), **visual document retrieval** for layout-aware search ("which page has that specific chart?"), and **knowledge graph traversal** for relationship queries ("how are these entities connected across documents?").

That's Spektr: a system that combines all three, exposed as an MCP server so any AI agent can use it as a tool.

---

## 2. RAG Fundamentals — What Happens Under the Hood

RAG stands for Retrieval-Augmented Generation. The idea is simple: instead of relying on an LLM's training data alone, you first *retrieve* relevant context from your own documents, then *augment* the LLM's prompt with that context, and let it *generate* an answer grounded in your actual data.

### The Standard RAG Pipeline

**Ingestion time** (happens once per document, then incrementally):

```
Document → Split into chunks → Embed each chunk → Store vectors + metadata
```

**Query time** (happens per user question):

```
User question → Embed the question → Find similar vectors → 
Feed top-K chunks to LLM → LLM generates answer with citations
```

### Why Chunking Matters

LLMs have context windows (how much text they can process at once). You can't feed an entire 200-page PDF into a prompt. So you split documents into chunks — typically 200–1000 tokens each — and only retrieve the most relevant chunks for a given query.

The art is in *how* you chunk. Naive chunking (split every N characters) breaks mid-sentence and loses context. Better approaches:

- **Recursive character splitting**: split on paragraph breaks, then sentences, then characters as fallback
- **Semantic chunking**: use embedding similarity to detect topic boundaries — when the embedding of consecutive sentences suddenly shifts, that's a natural chunk boundary
- **Context-aware chunking**: preserve document structure (keep a heading with its paragraphs, keep a table row intact)

For Spektr, we chunk text content for the dense vector collection. But for visual content, we don't chunk at all — we embed entire page images, which is one of the key insights of the architecture.

### The Embedding Step

This is where the magic happens. An embedding model converts text (or images) into a dense vector — a list of numbers, typically 128 to 2048 dimensions. These vectors capture *meaning*, not just keywords.

"The company's revenue increased by 23%" and "Sales grew almost a quarter" would have very similar vectors despite sharing almost no words. That's the power of semantic embedding.

---

## 3. Why Embeddings Matter — From Words to Vectors

### What's Actually in a Vector?

When a model embeds the sentence "Kubernetes orchestrates container workloads", it produces something like:

```
[0.023, -0.147, 0.891, 0.034, ..., -0.256]  # 2048 numbers
```

Each dimension doesn't correspond to a human-readable concept. The model learned during training that arranging meanings in this high-dimensional space — where "Kubernetes" is close to "Docker" and "container orchestration" but far from "French cooking" — produces useful retrieval results.

### Cosine Similarity

To compare two vectors, we use cosine similarity — the cosine of the angle between them. It ranges from -1 (opposite) to 1 (identical). In practice, for normalized vectors (unit length), this is just the dot product.

```
similarity = dot(vector_a, vector_b)
# 0.95 = very similar meaning
# 0.50 = somewhat related
# 0.10 = unrelated
```

This is what Qdrant does millions of times per query — compute the similarity between your query vector and every stored document vector, then return the top-K most similar.

### The Modality Gap Problem

Here's a subtle but important issue. Traditional multimodal models like CLIP use two separate encoders — one for text, one for images. They're trained to put matching text-image pairs close together in vector space, but they don't share internal representations.

The result is a "modality gap": text vectors cluster in one region of the space, image vectors cluster in another. A text about a sunset might be at cosine similarity 0.85 with the same text rephrased, but only 0.65 with a photo of that exact sunset. Semantically unrelated text can be *more similar* to a query than a perfectly matching image, just because they're in the same modality cluster.

This matters for our use case. When someone asks "show me the revenue table", we need the text query to find the right PDF page image. If there's a big modality gap, our retrieval quality suffers.

Jina v4 solves this with a fundamentally different architecture.

---

## 4. Jina v4 — One Model to Rule Them All

### Architecture: Single-Stream, Not Dual-Encoder

Jina v4 is built on Qwen2.5-VL-3B-Instruct (a 3.8 billion parameter vision-language model). Unlike CLIP-style dual encoders, it processes text and images through a **single unified pathway**:

1. Images are first converted into "visual tokens" by a vision encoder
2. These visual tokens are concatenated with text tokens
3. Both go through the same transformer decoder with shared attention layers
4. The output is pooled (or projected) into embedding vectors

Because text and images are processed *together* through shared attention, the model learns a truly unified representation. The modality gap that plagues CLIP is dramatically reduced — from a gap score of 0.15 (OpenAI CLIP) to 0.71 (Jina v4), meaning cross-modal retrieval is far more reliable.

### Task-Specific LoRA Adapters

Jina v4 has three lightweight LoRA adapters (about 60M parameters each, less than 2% of the model) that specialize the frozen backbone for different tasks:

- **`retrieval.query`** / **`retrieval.passage`**: Asymmetric retrieval — queries and documents get different optimizations. Use `retrieval.query` when embedding a search query, `retrieval.passage` when embedding a document chunk or page image. This asymmetry is key: queries are short and intent-focused, documents are long and information-dense. Optimizing each separately produces better retrieval than treating them symmetrically.

- **`text-matching`**: Symmetric similarity — both inputs are treated equally. Use for duplicate detection, semantic textual similarity, or clustering.

- **`code`**: Optimized for code retrieval tasks.

You select the adapter at inference time via the `task` parameter in the API call. The frozen backbone stays the same — only the lightweight adapter changes.

### Matryoshka Representation Learning

The default embedding dimension is 2048, but Jina v4 was trained with Matryoshka Representation Learning (MRL). This means you can truncate vectors to 128, 256, 512, or 1024 dimensions with minimal quality loss.

Why does this work? During training, the model is optimized so that the first N dimensions carry the most important information. Think of it like a progressive JPEG — the first bytes give you a blurry version, more bytes add detail. The first 256 dimensions capture most of the semantic meaning; dimensions 257–2048 add finer distinctions.

This is valuable for production: storing 256-dimensional vectors uses 8x less memory than 2048-dimensional ones. You might use 2048 for your primary collection and 256 for a cheaper, faster pre-filtering step.

### Dual Output Modes

This is critical for Spektr. Jina v4 can produce two fundamentally different types of output:

**Single-vector (dense)**: One vector per input (default 2048 dimensions). The entire text or image is compressed into a single point in vector space. Fast storage, fast search, good for general semantic retrieval.

**Multi-vector (ColBERT-style)**: Multiple vectors per input — one per token (text) or per patch (image), each 128 dimensions. This preserves fine-grained, token-level semantics. Slower and uses more storage, but dramatically better for visually complex documents.

We use both in Spektr — dense for text chunks, multi-vector for page images.

---

## 5. Single-Vector vs Multi-Vector — The ColBERT Revolution

### The Single-Vector Limitation

When you compress an entire document page into one 2048-dimensional vector, you're asking a lot. A page might contain a title, three paragraphs of text, a data table, and a bar chart. All of that gets squeezed into 2048 numbers. The vector captures the *general topic* well, but loses spatial details — where on the page is the table? What do the chart axes say?

For text-only retrieval, this is usually fine. But for visually rich documents — scanned invoices, architectural diagrams, slide decks — the single vector loses too much.

### How ColBERT Works

ColBERT (Contextualized Late Interaction over BERT) introduced a different approach. Instead of one vector per document, you keep *one vector per token*:

```
Document: "Revenue grew 23% in Q3"
Single-vector: [0.23, -0.14, 0.89, ...]           # 1 × 2048

Multi-vector:  [[0.12, 0.34, ...],    # "Revenue"   # token 1 × 128
                [0.56, 0.78, ...],    # "grew"       # token 2 × 128
                [0.91, 0.23, ...],    # "23%"        # token 3 × 128
                [0.45, 0.67, ...],    # "in"         # token 4 × 128
                [0.89, 0.12, ...]]    # "Q3"         # token 5 × 128
```

For images, it works similarly — the image is divided into patches (small rectangular regions), and each patch gets its own embedding vector. A 300 DPI A4 page might produce hundreds of patch vectors, each capturing the semantics of that specific region.

### MaxSim Scoring

The similarity between a multi-vector query and a multi-vector document is computed using **MaxSim** (Maximum Similarity):

1. For each query token vector, find the document token vector it's most similar to
2. Take the maximum similarity score
3. Sum across all query tokens

```
Score = Σ (for each query token q) max(for each doc token d) similarity(q, d)
```

This is powerful because it allows *partial matching*. If you search for "Q3 revenue table", the word "table" can match a patch that contains table structure, "revenue" can match a different patch with revenue text, and "Q3" can match yet another patch. Each query term independently finds its best match in the document.

This is why multi-vector retrieval is so effective for visually rich documents — it can simultaneously match text content, layout structure, and visual elements across different regions of a page.

### The Storage Trade-off

Multi-vector uses significantly more storage. A single-vector document page takes 2048 × 4 bytes = 8 KB. A multi-vector page with 500 patches takes 500 × 128 × 4 bytes = 256 KB. That's 32x more storage per page.

For Spektr, we use both collections:
- `documents_dense` (single-vector): fast, efficient text search
- `documents_multivec` (multi-vector): precise visual document retrieval

The MCP server exposes both as separate tools, and the agent decides which to use based on the query.

---

## 6. Qdrant — Why a Dedicated Vector Database

### Why Not Just PostgreSQL + pgvector?

PostgreSQL with the pgvector extension can do vector similarity search. We're already running PostgreSQL for CocoIndex. So why add another database?

The answer is **multi-vector support**. pgvector stores one vector per row and computes cosine similarity between single vectors. It has no concept of multi-vector documents or MaxSim scoring. Qdrant does.

Qdrant's `multivector_config` with `comparator: "max_sim"` implements exactly the ColBERT MaxSim scoring described above. When you query the `documents_multivec` collection, Qdrant:

1. Takes your list of query token vectors
2. For each stored document (which has its own list of patch vectors)
3. Computes MaxSim between query and document
4. Returns the top-K highest-scoring documents

This is a specialized operation that pgvector simply cannot do.

### How Qdrant Stores Data

Each entry in Qdrant is called a **point**. A point has:

- **ID**: unique identifier (we use UUID v4)
- **Vector(s)**: the embedding(s) — either a single vector or a named vector collection
- **Payload**: arbitrary JSON metadata (source file, page number, content type, etc.)

Points are organized into **collections**. Each collection has a fixed vector configuration (dimension, distance metric, whether it's multi-vector). We use two collections:

```
documents_dense:
  vectors: 2048-dim, cosine distance
  payload indexes: source_file (keyword), content_type (keyword)
  
documents_multivec:
  vectors: "colbert" named vector, 128-dim per token, max_sim comparator
  payload indexes: source_file (keyword)
```

### HNSW Index

Qdrant uses HNSW (Hierarchical Navigable Small World) graphs for approximate nearest neighbor search. Instead of comparing your query against every stored vector (which would be O(n)), HNSW builds a multi-layered graph structure that allows O(log n) search with high recall.

The key parameters are:
- **`m`**: how many connections each node has (more = better recall but slower indexing)
- **`ef_construct`**: how many candidates to consider during index building

For most use cases, the defaults work well. You'd tune these if you have millions of documents and need to balance speed vs accuracy.

### Payload Filtering

One of Qdrant's strengths is combining vector search with metadata filters. When the agent searches for "Q3 revenue" but only in PDF files, we can:

```python
client.query_points(
    collection_name="documents_dense",
    query=query_vector,
    query_filter=Filter(must=[
        FieldCondition(key="content_type", match=MatchValue(value="text_chunk")),
    ]),
    limit=10,
)
```

This first narrows the search space by metadata, then does vector similarity within that subset. It's much more efficient than retrieving top-K and then filtering in Python.

---

## 7. Knowledge Graphs — Why Vectors Aren't Enough

### The Relationship Blindness Problem

Vector search finds *similar content*, but it can't answer *relationship questions*. Consider:

"Which companies are suppliers to Acme Corp, and what technologies do they use?"

This requires:
1. Find the entity "Acme Corp"
2. Traverse SUPPLIED_BY relationships to find supplier entities
3. From each supplier, traverse USES_TECHNOLOGY relationships
4. Return the results

No amount of embedding similarity will solve this. The answer might be spread across 15 different documents, with no single chunk containing all the information. You need a structured representation of *entities and their relationships* — a knowledge graph.

### What's in a Knowledge Graph?

A knowledge graph consists of:

- **Nodes** (entities): things like people, organizations, products, technologies, locations
- **Edges** (relationships): typed connections between entities — "works at", "produces", "uses", "acquired"
- **Properties**: attributes on both nodes and edges — a person has a role, a relationship has a start date

The graph structure allows multi-hop queries: "Find all people who work at companies that use Kubernetes and are located in Berlin." Each hop traverses one relationship type, and the query combines them.

### How We Build the Graph

During ingestion, after extracting text from documents, we use an LLM to extract entities and relationships:

```
Input text: "Sarah Chen, CTO of Acme Corp, announced a partnership
with DataFlow Inc to integrate their Kubernetes-based platform."

Extracted:
  Entities:
    - Sarah Chen (PERSON)
    - Acme Corp (ORGANIZATION)
    - DataFlow Inc (ORGANIZATION)
    - Kubernetes (TECHNOLOGY)
  
  Relationships:
    - Sarah Chen -[WORKS_AT {role: "CTO"}]-> Acme Corp
    - Acme Corp -[PARTNERS_WITH]-> DataFlow Inc
    - DataFlow Inc -[USES_TECHNOLOGY]-> Kubernetes
```

This extraction runs on every text chunk during ingestion. Over time, as more documents are processed, the graph grows and becomes increasingly valuable — entities from different documents get connected, revealing relationships that no single document contains.

### Graph vs Vector: When to Use Which

| Query Type | Best Approach | Why |
|------------|--------------|-----|
| "What does our Q3 report say about margins?" | Vector search | Semantic similarity finds relevant passages |
| "Who is the CEO of Acme Corp?" | Graph search | Direct entity lookup, no similarity needed |
| "How is Acme connected to DataFlow?" | Graph search (path) | Multi-hop traversal between entities |
| "Find documents about companies using Kubernetes" | Hybrid (graph + vector) | Graph finds the entities, vector enriches with context |
| "Show me the architecture diagram from the platform review" | Visual search (multi-vector) | Needs visual layout understanding |

The power of Spektr is that the agent has all three tools and can choose — or combine — them per query.

---

## 8. Neo4j — Graph Database Mechanics

### Why Neo4j?

Neo4j is the most widely used graph database and has the richest ecosystem for our use case:

- **Cypher query language**: intuitive, SQL-like syntax for graph traversal
- **APOC plugin**: hundreds of utility procedures (dynamic relationship creation, full-text search, path algorithms)
- **Community edition**: free, open source, sufficient for our needs
- **Pydantic AI integration**: the Cole Medin reference architecture uses it, so there's proven integration patterns

### Cypher — The Graph Query Language

Cypher uses ASCII art patterns to express graph queries. Nodes are `()`, relationships are `-[]->`:

```cypher
// Find a person and where they work
MATCH (p:Person {name: "Sarah Chen"})-[:WORKS_AT]->(org:Organization)
RETURN p.name, org.name

// Find all technologies used by partners of Acme
MATCH (acme:Organization {name: "Acme Corp"})
      -[:PARTNERS_WITH]->(partner:Organization)
      -[:USES_TECHNOLOGY]->(tech:Technology)
RETURN partner.name, tech.name

// Shortest path between two entities
MATCH path = shortestPath(
    (a:Entity {name: "Sarah Chen"})-[*..6]-(b:Entity {name: "Kubernetes"})
)
RETURN path
```

The `*..6` means "up to 6 hops." Without a limit, pathfinding can explore the entire graph.

### Our Schema Design

We use three node types and several relationship types:

**Nodes:**
- `:Document` — one per source file, identified by S3 key
- `:Chunk` — one per text chunk, ID matches the Qdrant point ID (this is the bridge between vector search and graph search)
- `:Entity` — extracted entities with a name, type, and optional description

**Key relationships:**
- `(:Document)-[:HAS_CHUNK]->(:Chunk)` — document structure
- `(:Chunk)-[:MENTIONS]->(:Entity)` — which chunks mention which entities
- `(:Entity)-[:WORKS_AT|PARTNERS_WITH|PRODUCES|...]->(:Entity)` — entity relationships

The `MENTIONS` relationship is especially important. When vector search returns a chunk, we can follow `MENTIONS` to find which entities it references, then traverse further to find related entities and documents. This is the graph-vector bridge.

### MERGE vs CREATE

We use `MERGE` instead of `CREATE` for all writes. `MERGE` is an idempotent "find or create" — it first checks if a matching node/relationship exists. If yes, it updates properties. If no, it creates a new one.

This is critical for incremental ingestion. When CocoIndex re-processes a modified document, the entity extraction might produce the same entities again. With `MERGE`, they get updated rather than duplicated.

```cypher
-- This is safe to run repeatedly:
MERGE (e:Entity {name: "Acme Corp", type: "ORGANIZATION"})
SET e.description = "Manufacturing company",
    e.last_seen = datetime()
ON CREATE SET e.first_seen = datetime()
```

---

## 9. GraphRAG — Combining Graphs and Vectors

### The Hybrid Search Pattern

GraphRAG is the pattern of using both graph and vector retrieval in the same system. There are several strategies:

**Parallel fusion** (what Spektr uses for `hybrid_search`): Run vector search and graph search simultaneously, then merge results. Fast, simple, and lets the agent see both types of results.

```python
vector_results, graph_results = await asyncio.gather(
    vector_search(query, limit=10),
    graph_search(query, search_type="entity", limit=10),
)
```

**Graph-guided retrieval**: Use the graph to find relevant entities first, then use those entities as filters for vector search. More precise but slower.

**Vector-guided graph expansion**: Use vector search to find initial chunks, then follow `MENTIONS` relationships to find connected entities, then traverse the graph for additional context.

### Why This Outperforms Pure Vector RAG

Research from Microsoft (the GraphRAG paper) and Neo4j shows that GraphRAG consistently outperforms pure vector RAG on:

- **Multi-hop questions**: "What technologies do Acme's partners use?" requires traversing relationships
- **Aggregation queries**: "How many suppliers are connected to this product line?" requires structured counting
- **Precise entity lookups**: "What is Sarah Chen's role?" is a direct property access, not a similarity problem
- **Cross-document synthesis**: Connecting facts from different documents through shared entities

Pure vector RAG might find a chunk mentioning Sarah Chen's role, but only if the chunk happens to contain that information near other relevant text. Graph search finds it by structure, regardless of semantic similarity.

---

## 10. CocoIndex — Declarative Data Pipelines

### What CocoIndex Does

CocoIndex is a framework for building data transformation pipelines — specifically designed for AI/ML workloads like building vector indexes and knowledge bases. Think of it as "ETL for embeddings."

The key differentiator is **incremental processing**. When a file in your S3 bucket changes, CocoIndex doesn't re-process your entire corpus. It detects which files changed, re-processes only those, and updates only the affected vectors and metadata.

### Declarative vs Imperative

Traditional approach (imperative): you write Python scripts that loop over files, call APIs, handle retries, track which files were already processed, manage state...

CocoIndex approach (declarative): you define *what* transformations should happen, and CocoIndex handles the *how* — change detection, parallelization, caching, retries, state tracking.

```python
@cocoindex.flow_def(name="RAGIngestion")
def rag_ingestion_flow(flow_builder, data_scope):
    # Declare the source
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.AmazonS3(bucket_name="my-bucket", binary=True)
    )
    
    # Declare transformations
    with data_scope["documents"].row() as doc:
        doc["pages"] = flow_builder.transform(file_to_pages, ...)
        with doc["pages"].row() as page:
            page["embedding"] = page["image"].transform(embed_function)
    
    # Declare where output goes
    output.export("collection", cocoindex.targets.Qdrant(...))
```

You never write a loop. CocoIndex figures out what to process based on what changed since the last run.

### State Tracking via PostgreSQL

CocoIndex uses PostgreSQL to track pipeline state — which files have been processed, what the hash of each file was, which chunks were produced. This is why we have PostgreSQL in the stack even though we use Qdrant for vectors.

When a file changes in S3:
1. SQS delivers the change event
2. CocoIndex compares the new file hash against the stored hash
3. If different, it re-runs the pipeline for that file only
4. It deletes old vectors/graph nodes for that file
5. It creates new vectors/graph nodes
6. It updates the state in PostgreSQL

This is the "smart caching" that makes incremental updates efficient.

### Custom Operations

CocoIndex has built-in operations like `SentenceTransformerEmbed` and `ColPaliEmbedImage`. Since we're using Jina v4 via API, we need custom operations:

```python
@cocoindex.op.function()
async def jina_embed_text(text: str) -> list[float]:
    embedder = JinaV4Embedder(api_key=os.environ["JINA_API_KEY"])
    results = await embedder.embed_text([text], task="retrieval.passage")
    return results[0]
```

The `@cocoindex.op.function()` decorator registers this as a CocoIndex operation that can be used in the pipeline's `transform` calls. CocoIndex handles batching and caching — if the same text was already embedded and hasn't changed, it skips the API call.

---

## 11. S3 + SQS — Event-Driven Ingestion

### The Auto-Ingestion Pattern

This is one of the most valuable parts of the architecture. Instead of manually triggering ingestion when new documents arrive, the system watches the S3 bucket and automatically processes new files.

The flow:

```
1. Someone uploads "q4-report.pdf" to the S3 bucket
2. S3 sends an event notification to the SQS queue:
   {
     "eventName": "ObjectCreated:Put",
     "s3": {
       "bucket": {"name": "spektr-docs"},
       "object": {"key": "reports/q4-report.pdf"}
     }
   }
3. CocoIndex (running in live mode) receives the SQS message
4. CocoIndex processes only this new file:
   - Converts pages to images
   - Embeds text and images via Jina v4
   - Extracts entities and writes to Neo4j
   - Stores vectors in Qdrant
5. Within seconds, the new document is searchable
```

### Why SQS and Not Direct Polling?

S3 doesn't have a "watch" or "subscribe" API. You could poll S3 every few seconds to check for new files, but:

- Polling wastes API calls (and costs money) when nothing changes
- Polling has latency — you miss files between polls
- Polling is unreliable if your process crashes mid-poll

SQS solves all of these. S3 *pushes* events to SQS, and SQS *queues* them reliably. If your ingestion pipeline is down, messages wait in the queue (up to 14 days). When it comes back up, it processes all queued messages.

### Setting It Up (AWS Side)

The AWS configuration involves:

1. **Create an SQS queue** with a policy allowing S3 to send messages
2. **Configure S3 event notifications** to send `ObjectCreated:*` and `ObjectRemoved:*` events to the queue
3. **Set IAM permissions** for CocoIndex to read from S3 and receive/delete SQS messages

CocoIndex's S3 source supports SQS natively:

```python
cocoindex.sources.AmazonS3(
    bucket_name="spektr-docs",
    sqs_queue_url="https://sqs.eu-central-1.amazonaws.com/123456789/spektr-events",
    included_patterns=["*.pdf", "*.png", "*.md"],
    binary=True,
)
```

### File Delete Handling

When a file is deleted from S3, the SQS event includes `ObjectRemoved:Delete`. CocoIndex handles this by:

1. Finding all chunks/pages that were produced from this file (tracked in PostgreSQL state)
2. Deleting the corresponding points from Qdrant
3. Deleting the corresponding nodes and relationships from Neo4j
4. Updating its internal state

This means the search index is always in sync with the bucket contents.

---

## 12. MCP Protocol — Making RAG a Tool for Agents

### What is MCP?

MCP (Model Context Protocol) is an open standard by Anthropic for connecting AI models to external tools and data sources. Think of it as a USB-C for AI — a standardized interface that any agent can use to interact with any tool.

Before MCP, every AI tool integration was custom. You'd write a specific function for each LLM framework. With MCP, you write the tool once as an MCP server, and any MCP-compatible agent can use it.

### MCP Architecture

An MCP system has three parts:

- **MCP Server**: exposes tools (functions the agent can call), resources (data the agent can read), and prompts (templates the agent can use)
- **MCP Client**: built into the agent framework, connects to servers and makes tools available
- **Transport**: how they communicate — SSE (Server-Sent Events over HTTP) for remote, stdio for local

### Why Expose RAG as an MCP Server?

Instead of hardcoding RAG into your agent, you make it a separate service that any agent can discover and use. Benefits:

- **Multiple agents can share one RAG server** — your coding assistant, your chat agent, and your email summarizer all use the same knowledge base
- **The RAG server evolves independently** — upgrade your embedding model or add a new collection without touching agent code
- **Tool descriptions guide the agent** — the agent reads the tool documentation and decides when/how to use each search tool

### Our MCP Tools

Spektr exposes five tools:

```
vector_search    → Dense semantic search over text chunks
visual_search    → Multi-vector search over page images (ColBERT-style)
graph_search     → Neo4j entity/relationship queries
hybrid_search    → Parallel vector + graph, merged results
ingest_url       → On-demand ingestion of a URL (future)
```

Each tool has a description, parameter schema, and return type. When an agent connects, it receives these descriptions and can reason about which tool to call:

- User asks "What were Q3 margins?" → agent calls `vector_search`
- User asks "Find the architecture diagram" → agent calls `visual_search`
- User asks "Who does Acme partner with?" → agent calls `graph_search`
- User asks a complex question → agent calls `hybrid_search` or multiple tools in sequence

### SSE Transport

SSE (Server-Sent Events) is an HTTP-based protocol where the server pushes events to the client over a long-lived connection. For MCP, this means:

1. Agent connects to `http://localhost:8080/sse`
2. Server sends tool definitions as events
3. Agent sends tool calls as HTTP POST requests
4. Server streams results back as SSE events

SSE is simpler than WebSockets (no binary frames, works through HTTP proxies) and sufficient for our use case since the agent initiates all interactions.

---

## 13. FastMCP — The Server Implementation

### What is FastMCP?

FastMCP is the official Python SDK for building MCP servers. It provides decorators to register tools, handles the protocol details, and supports both SSE and stdio transports.

### Tool Registration

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="spektr",
    description="Multimodal RAG with knowledge graph"
)

@mcp.tool()
async def vector_search(query: str, limit: int = 10) -> list[dict]:
    """Semantic search over document chunks using dense vectors.
    
    Best for: text-based queries, finding relevant passages.
    """
    # Implementation...
```

The docstring becomes the tool's description. FastMCP automatically extracts parameter types and generates the JSON schema that agents use to understand how to call the tool.

### Type Annotations Matter

FastMCP uses Python type hints to generate the tool schema:

```python
async def vector_search(
    query: str,                          # required string
    limit: int = 10,                     # optional int, default 10
    content_type: str | None = None,     # optional string, nullable
) -> list[dict]:
```

This generates an MCP tool schema that tells the agent: "this tool takes a required query string, an optional limit integer, and an optional content_type string." The agent uses this to construct valid tool calls.

### Running the Server

```python
if __name__ == "__main__":
    mcp.run(transport="sse")  # Remote access via HTTP
    # or
    mcp.run(transport="stdio")  # Local access via stdin/stdout
```

SSE for when the agent runs in a separate process or machine. stdio for when the agent spawns the server as a subprocess (common in desktop AI tools like Claude Desktop).

---

## 14. Pydantic AI — The Agent Framework

### Why Pydantic AI?

Pydantic AI is the agent framework built by the Pydantic team (the people behind Pydantic, FastAPI's validation layer). It's designed for production use with:

- **Provider-agnostic**: works with OpenAI, Anthropic, Ollama, Google, etc.
- **MCP integration**: built-in support for connecting to MCP servers
- **Streaming**: real-time response streaming
- **Type safety**: uses Pydantic models for tool inputs/outputs

The Cole Medin reference architecture uses Pydantic AI, which is why we chose it — proven patterns exist for connecting it to Neo4j + vector search.

### How the Agent Uses MCP Tools

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

# Connect to the Spektr MCP server
spektr = MCPServerSSE(url="http://localhost:8080/sse")

agent = Agent(
    model="anthropic:claude-sonnet-4-20250514",
    mcp_servers=[spektr],
    system_prompt="""You have access to a knowledge base via MCP tools.
    
    Use vector_search for text-based semantic queries.
    Use visual_search for finding charts, diagrams, tables.
    Use graph_search for entity relationships and connections.
    Use hybrid_search for complex questions needing both.""",
)

# The agent now automatically has access to all Spektr tools
result = await agent.run("What technologies does Acme Corp use?")
```

The agent's LLM reads the tool descriptions, decides which tool(s) to call, constructs the arguments, calls the tool via MCP, reads the results, and generates a natural language answer incorporating the retrieved information.

### The Agent Routing Decision

This is where the magic happens. The agent doesn't just blindly search — it *reasons* about which tool is best for each query. The system prompt guides this, but the LLM's own judgment plays a role too.

A well-designed agent might even call multiple tools sequentially:

1. Call `graph_search` to find entities related to "Acme Corp"
2. Use the results to formulate a more specific query
3. Call `vector_search` with that refined query to find detailed passages
4. Synthesize everything into an answer

This multi-step reasoning is what makes agentic RAG more powerful than simple retrieve-and-generate.

---

## 15. How It All Connects — End-to-End Request Flow

### Ingestion Flow (Document Arrives)

```
1. User uploads "q4-report.pdf" to S3 bucket

2. S3 → SQS event notification
   {"eventName": "ObjectCreated:Put", "key": "reports/q4-report.pdf"}

3. CocoIndex receives SQS message, downloads file from S3

4. file_to_pages: MIME detection → PDF → 12 page images at 300 DPI

5. For each page:
   a. Jina v4 API (retrieval.passage, single-vector, 2048d)
      → Store in Qdrant documents_dense with text metadata
   
   b. Jina v4 API (retrieval.passage, multi-vector, 128d per patch)  
      → Store in Qdrant documents_multivec with page metadata
   
   c. If text was extracted:
      - Chunk the text (semantic chunking)
      - Jina v4 API (retrieval.passage, single-vector) per chunk
      → Store each chunk in Qdrant documents_dense
      
      - LLM entity extraction on each chunk
      → Write entities + relationships to Neo4j
      → Link chunks to entities via MENTIONS

6. CocoIndex updates PostgreSQL state (file hash, chunk IDs, timestamps)

7. Document is now searchable across all three retrieval modes
```

### Query Flow (Agent Answers a Question)

```
1. User: "What suppliers does Acme Corp work with, and show me any 
          comparison tables from the procurement review"

2. Agent LLM reads the question, decides to use two tools:
   
   Step A: graph_search(query="Acme Corp", search_type="entity")
   → Neo4j returns:
     {entity: "Acme Corp", connections: [
       {name: "DataFlow Inc", relation: "PARTNERS_WITH"},
       {name: "CloudBase", relation: "SUPPLIED_BY"},
     ]}
   
   Step B: visual_search(query="comparison table procurement suppliers")
   → Qdrant multi-vector search returns:
     [{source_file: "procurement-review-2024.pdf", page: 7, score: 0.87},
      {source_file: "procurement-review-2024.pdf", page: 12, score: 0.72}]

3. Agent LLM synthesizes results:
   "Based on the knowledge graph, Acme Corp works with two suppliers:
   DataFlow Inc (partner) and CloudBase (supplier). The procurement
   review contains comparison tables on pages 7 and 12 — page 7 shows
   pricing comparisons and page 12 shows feature comparisons."
```

---

## 16. Production Considerations

### Rate Limiting on Jina API

Jina v4 API has rate limits. During initial bulk ingestion of many documents, you'll hit them. Strategies:

- **Batch embeddings**: send multiple texts/images per API call (the API batches internally)
- **Exponential backoff**: on 429 (rate limited), wait 1s, then 2s, then 4s...
- **Queue processing**: process pages through a work queue with configurable concurrency
- **Dimension reduction**: use 512 or 1024 dimensions instead of 2048 for initial ingestion (Matryoshka lets you do this with minimal quality loss), then re-embed at full resolution for important documents

### Monitoring

Key metrics to track:

- **Ingestion lag**: time between file upload to S3 and searchability in Qdrant
- **Query latency**: time from MCP tool call to results (broken down by tool)
- **Embedding API cost**: Jina charges per token — track monthly spend
- **Qdrant memory usage**: multi-vector collection grows fast
- **Neo4j entity count**: watch for entity duplication (imperfect extraction)

### Re-ranking (Future Enhancement)

After initial retrieval (top-50 candidates from Qdrant), a re-ranker can re-score results for better precision. Jina offers a Reranker model, or you can use a cross-encoder:

1. Retrieve top-50 from vector search
2. Re-rank with a cross-encoder (query + each result scored together)
3. Return top-10 after re-ranking

This adds latency but significantly improves precision, especially for ambiguous queries.

### VLM Answer Generation (Future Enhancement)

For visual search results, instead of just returning "page 7 of procurement-review.pdf", you could:

1. Retrieve the page image from S3
2. Send it to a Vision Language Model (Qwen3-VL, GPT-4o)
3. Ask the VLM to extract specific information from the image
4. Return a text answer with the visual context

This turns "I found this page" into "The table on page 7 shows CloudBase at $42/month vs DataFlow at $38/month for the standard tier."

---

## Appendix A: Key Terminology Quick Reference

| Term | Meaning |
|------|---------|
| **Dense vector** | Single fixed-size vector representing an entire text/image (e.g., 2048 dims) |
| **Multi-vector** | Multiple vectors per document, one per token/patch (ColBERT-style) |
| **MaxSim** | Scoring method for multi-vector: sum of max similarities between query and doc tokens |
| **LoRA** | Low-Rank Adaptation — small adapter layers that specialize a frozen model for specific tasks |
| **Matryoshka** | Training technique allowing vector truncation with minimal quality loss |
| **Modality gap** | Phenomenon where text and image vectors cluster in separate regions despite semantic matches |
| **HNSW** | Hierarchical Navigable Small World — graph-based index for approximate nearest neighbor search |
| **Cypher** | Neo4j's graph query language using ASCII art patterns: `()-[]->()` |
| **MERGE** | Neo4j's idempotent "find or create" operation |
| **SSE** | Server-Sent Events — HTTP-based one-way streaming protocol used by MCP |
| **MCP** | Model Context Protocol — standard for connecting AI agents to tools |
| **CocoIndex** | Declarative data pipeline framework with incremental processing |
| **ColBERT** | Contextualized Late Interaction over BERT — multi-vector retrieval with MaxSim scoring |
| **RAG** | Retrieval-Augmented Generation — retrieve context, then generate answers |
| **GraphRAG** | RAG enhanced with knowledge graph traversal for relationship-aware retrieval |

## Appendix B: API Calls Quick Reference

### Jina v4 — Single-Vector Text Embedding

```bash
curl https://api.jina.ai/v1/embeddings \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "task": "retrieval.passage",
    "dimensions": 2048,
    "normalized": true,
    "input": [{"text": "Your document text here"}]
  }'
```

### Jina v4 — Multi-Vector Image Embedding

```bash
curl https://api.jina.ai/v1/embeddings \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "task": "retrieval.passage",
    "normalized": true,
    "input": [{"image": "data:image/png;base64,iVBOR..."}],
    "embedding_type_params": {"output_type": "colbert"}
  }'
```

### Jina v4 — Query Embedding (Note Different Task)

```bash
curl https://api.jina.ai/v1/embeddings \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "task": "retrieval.query",
    "dimensions": 2048,
    "normalized": true,
    "input": [{"text": "What were Q3 revenue figures?"}]
  }'
```

### Qdrant — Create Dense Collection

```bash
curl -X PUT http://localhost:6333/collections/documents_dense \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {"size": 2048, "distance": "Cosine"}
  }'
```

### Qdrant — Create Multi-Vector Collection

```bash
curl -X PUT http://localhost:6333/collections/documents_multivec \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "colbert": {
        "size": 128,
        "distance": "Cosine",
        "multivector_config": {"comparator": "max_sim"}
      }
    }
  }'
```

### Neo4j — Create Constraints

```cypher
CREATE CONSTRAINT doc_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.s3_key IS UNIQUE;

CREATE CONSTRAINT entity_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE;

CREATE CONSTRAINT chunk_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
```

---

*End of deep dive. Next step: implementation Phase 1 — docker-compose up and testing each component independently.*

---

## Appendix C: Divergences from Current Implementation (as of 2026-04)

!!! warning "This document is a learning resource, not source of truth"
    The body above was written early in the project. Several design decisions evolved during implementation. For authoritative, up-to-date details, see `docs/resources/rag-mcp-architecture-blueprint.md` and the module-specific pages under `Ingestion` / `MCP Server` / `Configuration`. The deltas below summarize what has changed.

### C.1 Entity & Relationship Schema

- **Entity types:** now 14 lowercase types (not 7 uppercase). See `config/constants.py :: ENTITY_TYPES`.
  `person, organization, location, date_time, monetary_value, document, product, technology, metric, event, legal_term, role, concept, requirement`
- **Entity property:** `type` (scalar) → **`types`** (array — GLiNER2 supports multi-label).
- **Uniqueness constraint:** `(name, type)` → **`(name)`** only.
- **Document key property:** `s3_key` → **`source_key`** (source can be local filesystem or S3).
- **Relationship types:** the hardcoded business relations in the body (`WORKS_AT`, `PARTNERS_WITH`, `PRODUCES`, `USES_TECHNOLOGY`, `LOCATED_IN`, `ACQUIRED`, `COMPETES_WITH`) are replaced by 12 generic, domain-agnostic relations with domain/range constraints:
  `created_by, owned_by, uses, part_of, measured_by, requires, applies_to, succeeds, conflicts_with, valued_at, scheduled_for, mentions`
- `related_to` was explicitly removed as an anti-pattern (noisy super-edges); `mentions` replaces it with honest textual co-occurrence semantics.
- Triples violating `RELATION_CONSTRAINTS` are dropped during GLiNER ingestion.

### C.2 Graph Engine — Dual Architecture

The body describes LLM-based entity extraction only. The current system supports **two pluggable graph engines**, selected via `GRAPH_ENGINE` env var:

| Engine | Trigger | Mechanism | Use case |
|-|-|-|-|
| `graphiti` (default) | Bulk ingestion + live streaming | LLM-based temporal episodic graph, schema-free | Rich, time-aware knowledge graph |
| `gliner` | Bulk ingestion only | Local CPU GLiNER2 model, schema-driven (14 types) | Fast, API-cost-free, deterministic |

See `ingestion/graph_engine.py` for the `GraphEngine` protocol and factory.

### C.3 Dual-Path Ingestion

The body covers only **Path A** (bulk files from local/S3 via CocoIndex). The current system adds **Path B**:

- **Path B — Live streaming:** FastAPI endpoint (`ingestion/live_ingest.py`) accepts streaming text via `POST /ingest/chunk`, written into Qdrant with `session_id` and `is_live` payload tags, and into Neo4j via Graphiti temporal episodes.
- Session lifecycle: `POST /session/start` → `/ingest/chunk` → `/session/end`.
- Both paths write to the same `documents_dense` collection; live content is distinguished by payload filters.
- MCP search tools are **session-aware** — they can scope or prioritize results by `session_id`.

### C.4 Authentication (not in body)

Two independent opt-in auth layers, both disabled when the corresponding key is empty:

- **MCP server:** `MCP_API_KEY` — Bearer token middleware on all tool calls.
- **Live ingest:** two-layer auth — `INGEST_API_KEY` gates `/session/start`, which returns an **ephemeral per-session token** required for `/ingest/chunk` and `/session/end`.

### C.5 Embedding Provider Abstraction

The body covers Jina v4 exclusively. The current system supports two providers via `EMBEDDING_PROVIDER`:

- **Jina v4** (default) — text + image + ColBERT multi-vector, 512-dim default (Matryoshka truncation from 2048).
- **Voyage AI** — `voyage-4-large` for text + `voyage-multimodal-3.5` for images, 1024-dim. **No ColBERT multi-vector support.**

Switching providers requires re-ingestion (vectors are incompatible across providers).

### C.6 Default Dense Dimension

Body says 2048. Current default is **512** (`JINA_DENSE_DIMENSIONS=512`), leveraging Matryoshka for 4× storage savings at ~95% quality retention.

### C.7 MCP Tools — Actual Surface

Body lists `ingest_url` as a planned tool — **not implemented**. Actual tool surface under `server/tools/`:

| Tool | Purpose |
|-|-|
| `vector_search` | Dense semantic search (session-aware) |
| `visual_search` | ColBERT multi-vector (requires `MULTIVEC_ENABLED=true`) |
| `graph_search` | Neo4j / Graphiti traversal |
| `hybrid_search` | Parallel vector + graph fusion |
| `reranker` | Cross-encoder re-ranking (feature-flagged) |
| `vlm_generator` | Vision-language model answer synthesis from retrieved images (feature-flagged) |

### C.8 File Processing

Body implies `pdf2image` + `mimetypes` only. Current pipeline uses **Docling** (layout-aware, detects figures/tables/formulas) with **PyMuPDF fallback**. The `IMAGE_EMBED_STRATEGY=smart` setting uses Docling layout analysis to embed only visually meaningful pages — a significant cost saver over embedding every page.

### C.9 Per-Document Schema Induction

Not covered in body. `ingestion/schema_inducer.py` can propose document-specific entity/relation schemas via LLM before GLiNER extraction — useful when a document's domain is too narrow for the generic 14-type schema. See `docs/ingestion/knowledge-graph.md` for the bootstrapping strategy.

### C.10 Infrastructure Pinning

`docker-compose.yml` pins specific versions for reproducibility (not shown in body):

- `qdrant/qdrant:v1.17.0`
- `neo4j:5.26-community`
- `postgres:17.2`

