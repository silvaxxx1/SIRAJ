# VECTOR DATABASE BATTLE CARD
## Al Rajhi Bank — Senior AI Engineer Interview

---

# 1. HOW VECTOR DATABASES WORK UNDER THE HOOD

Before the trade-offs, understand what every vector DB is actually doing.

## The Core Problem
Given a query vector Q and 10M stored vectors, find the top-K most similar.
Brute force = compute cosine similarity against all 10M = O(n). Too slow.
Solution = Approximate Nearest Neighbor (ANN) indexing.

## Index Types — Know These Cold

### HNSW (Hierarchical Navigable Small World)
Used by: Qdrant, Weaviate, pgvector, Milvus (optional)

```
Layer 2 (sparse):    1 ──── 5
                     |       |
Layer 1 (medium):   1 ─ 3 ─ 5 ─ 8
                    |   |   |   |
Layer 0 (dense):   1-2-3-4-5-6-7-8-9  ← all vectors
```

- Entry point at top layer → greedy search down layers
- Each vector has links to M nearest neighbors per layer
- Search: O(log n) — extremely fast
- Build: O(n log n) — slow for large datasets
- Memory: ~100 bytes/vector overhead
- Recall: 95-99% at ef=100 (tunable)
- **Best for: < 50M vectors, latency-critical workloads**

Key params:
| Param | Default | Effect |
|-------|---------|--------|
| `M` | 16 | Connections per node. Higher = better recall, more memory |
| `ef_construction` | 200 | Build quality. Higher = better index, slower build |
| `ef` (search) | 100 | Search quality. Higher = better recall, slower query |

### IVF (Inverted File Index)
Used by: Milvus, FAISS, pgvector

```
Cluster centroids:  C1  C2  C3  C4  C5
                    |   |   |   |   |
Vectors:           [v1][v2][v3][v4][v5]  ← vectors assigned to nearest centroid
```

- K-means clusters vectors at build time (nlist clusters)
- Query: find nearest centroid → search only that cluster's vectors
- `nprobe`: how many clusters to search (recall vs speed trade-off)
- **Best for: 1M-1B vectors, batch workloads, GPU acceleration**

Variants:
- `IVF_FLAT`: exact within clusters, high recall
- `IVF_SQ8`: scalar quantized (8-bit), 4x smaller, slight quality loss
- `IVF_PQ`: product quantized, maximum compression, lower recall

### DiskANN
Used by: Milvus (optional), Azure AI Search

- Graph-based like HNSW but designed for SSD storage
- Most vectors on disk, hot vectors in memory
- **Best for: datasets too large for RAM (100M+ vectors)**
- Higher latency than pure HNSW (disk I/O)

### FLAT (Brute Force)
Used by: All DBs as fallback

- Exact search, 100% recall
- O(n) — only viable for < 100K vectors
- Use for: testing, small collections, when recall must be 100%

---

# 2. THE BATTLE CARD

## Complete Trade-off Matrix

| Capability | Weaviate | Qdrant | Milvus | pgvector | Pinecone | FAISS |
|------------|----------|--------|--------|----------|----------|-------|
| **Native Hybrid Search** | ✅ BM25+vector, one query | ✅ (v1.10+, separate queries) | ✅ (v2.4+) | ❌ (manual + pg_trgm) | ❌ | ❌ |
| **Default Index** | HNSW | HNSW | HNSW / IVF / DiskANN | HNSW | Proprietary | FLAT / IVF / HNSW |
| **GPU Acceleration** | ❌ | ❌ | ✅ (IVF_GPU) | ❌ | ❌ | ✅ (FAISS-GPU) |
| **Max Practical Scale** | 100M vectors | 100M vectors | 1B+ vectors | 1-5M vectors | 10M+ (managed) | Unlimited (library) |
| **Filtering Speed** | Good (inverted index) | **Best (SIMD pre-filter)** | Good | Good (SQL WHERE) | Poor (post-filter) | None |
| **On-Prem** | ✅ Docker/K8s | ✅ Single binary | ✅ K8s (complex) | ✅ PostgreSQL extension | ❌ | ✅ Library |
| **Ops Complexity** | Medium | **Low** | **High** | **Lowest** | None (managed) | N/A (no server) |
| **Query Latency P95** | ~50ms | ~40ms | ~60ms | ~100ms (degrades at scale) | ~30ms | Varies |
| **Arabic / Unicode** | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| **Multi-tenancy** | ✅ (classes/tenants) | ✅ (collections) | ✅ (partition keys) | ✅ (schemas) | ✅ (namespaces) |
| **Storage Backend** | Custom (Go) | Rust + RocksDB | S3/MinIO + etcd | PostgreSQL | Proprietary | In-memory / mmap |
| **SAMA Data Residency** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Production References** | Netflix, Verizon | Uber, Discord | Salesforce, Airbnb | Supabase, Vercel | OpenAI, Notion | Meta, Microsoft |
| **Written In** | Go | **Rust** | C++ / Go | C | Proprietary | C++ |
| **GraphQL API** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **gRPC API** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Module/Plugin System** | ✅ (vectorizers, QnA) | Limited | ✅ | ❌ | ❌ | ❌ |
| **Sparse Vector Support** | ✅ BM25 built-in | ✅ (sparse vectors) | ✅ | ❌ | ✅ | ❌ |
| **Cost (self-hosted)** | Free | Free | Free | Free | N/A | Free |
| **Cost (managed)** | Weaviate Cloud | Qdrant Cloud | Zilliz Cloud | Supabase | $70/mo+ | N/A |

---

# 3. DEEP DIVE PER DATABASE

## Weaviate

**Architecture:**
- Go-based, monolithic-ish (easier to deploy than Milvus)
- Each "class" (collection) has its own HNSW index
- BM25 inverted index built alongside vector index automatically
- Module system: plug in vectorizers (BGE-M3, OpenAI, etc.) at the DB level
- Tenant isolation: separate storage per tenant in multi-tenant mode

**Unique strengths:**
- Native hybrid search: single API call runs BM25 + vector + RRF fusion internally
- GraphQL API: flexible queries, easy joins across collections
- Module ecosystem: run embedding model as sidecar, auto-vectorize on insert
- Generative module: plug LLM directly into retrieval query

**Weaknesses:**
- Higher memory usage than Qdrant
- Slower than Qdrant for pure vector search
- Less performant filtering than Qdrant's SIMD implementation
- Slower to reach production maturity than Qdrant

**When to choose Weaviate:**
RAG systems with hybrid search where you want one clean API, no custom fusion code, and a rich module ecosystem.

---

## Qdrant

**Architecture:**
- Rust-based — memory safety, no GC pauses
- Single binary deployment — no Kubernetes required
- RocksDB for persistent storage
- HNSW with payload-aware filtering (SIMD-optimized)
- Collections = separate indexes, fully isolated

**Unique strengths:**
- **Fastest payload filtering** — pre-filters using SIMD before vector search (not post-filter)
- Lowest operational overhead — download binary, run, done
- Built-in sparse vector support (for hybrid search)
- Consistent sub-50ms P99 latency

**The Qdrant filtering advantage (critical to understand):**

```
Post-filter (most DBs):
  Search all vectors → get top 1000 → apply filter → return 5
  Problem: may scan all 10M vectors to find 5 that pass filter

Pre-filter (Qdrant):
  Build index of filterable payloads → apply filter first → search only matching vectors
  Result: ~10ms vs ~100ms for heavy filters
```

**For banking:** User can only see their department's docs. Filter = `department = "retail"`. Qdrant's pre-filter makes this essentially free.

**Weaknesses:**
- Hybrid search requires two separate queries + manual RRF (v1.10 added hybrid but less integrated than Weaviate)
- No native module system
- Smaller ecosystem than Weaviate or Milvus

**When to choose Qdrant:**
Pure vector workloads with heavy metadata filtering. Simple ops, high performance. If hybrid search is less critical.

---

## Milvus

**Architecture:**
- Decoupled storage and compute (cloud-native design)
- Multiple microservices: proxy, query nodes, data nodes, index nodes, etcd, MinIO
- Supports multiple index types: HNSW, IVF_FLAT, IVF_SQ8, IVF_PQ, DiskANN, GPU indexes
- Segment-based storage: growing segments (RAM) → sealed segments (disk/S3)
- GPU acceleration for IVF indexes

**Unique strengths:**
- **Designed for 1B+ vectors** — the only true billion-scale open-source option
- GPU-accelerated index building and search
- Multiple index types per collection (can mix strategies)
- Mature: 5+ years production use, largest community

**Weaknesses:**
- Operationally complex: 8+ microservices, requires Kubernetes + etcd + MinIO
- Overkill for < 10M vectors
- Higher latency than Qdrant for small datasets (overhead of distributed system)
- Hybrid search less native than Weaviate

**When to choose Milvus:**
100M+ vectors, large team with K8s expertise, need GPU-accelerated indexing, or have specific requirements for DiskANN (RAM-efficient large-scale search).

**Migration path from Weaviate:**
```python
# Abstract interface — swap when scale demands it
class VectorStore(ABC):
    def hybrid_search(self, query, vector, filters, limit): ...

class WeaviateStore(VectorStore): ...   # Phase 1: 0-100M docs
class MilvusStore(VectorStore): ...     # Phase 2: 100M+ docs
```

---

## pgvector

**Architecture:**
- PostgreSQL extension — adds `vector` data type and `<=>` operator
- HNSW and IVF_FLAT index support (added HNSW in v0.5.0)
- Full SQL joins, transactions, ACID
- No separate service — lives inside your existing PostgreSQL

**Unique strengths:**
- Zero additional infrastructure if you already run PostgreSQL
- Full SQL: `SELECT * FROM docs WHERE department='retail' ORDER BY embedding <=> $1 LIMIT 10`
- ACID transactions: update vector and metadata atomically
- Familiar tooling: pgAdmin, existing DBA skills

**Weaknesses:**
- **Practical limit: ~1-5M vectors** — HNSW degrades, vacuum/analyze issues at scale
- No native sparse vectors (no BM25 hybrid without pg_trgm hack)
- Single-node by default (no built-in sharding)
- Memory pressure: all HNSW indexes must fit in shared_buffers

**When to choose pgvector:**
< 1M vectors, existing PostgreSQL infrastructure, team prefers SQL, strong transactional requirements. Good for prototyping, not for production RAG at bank scale.

---

## FAISS

**What it actually is:** A library, not a database. No server, no API, no persistence layer.

- Facebook AI Similarity Search
- Runs in-process (Python/C++)
- Fastest raw ANN search available
- Supports every index type: Flat, IVF, HNSW, PQ, ScaNN
- GPU support via FAISS-GPU

**Why it's on your CV:** You used it in early RAG prototyping before moving to a production vector DB.

**When to use FAISS:**
Research, prototyping, or when you're building a custom vector DB layer and need raw ANN primitives. Not suitable for production multi-tenant serving.

---

# 4. THE HYBRID SEARCH INTERNALS

## How BM25 Works (for the interview)

```
BM25 score(doc, query) = Σ IDF(term) × (TF × (k1+1)) / (TF + k1 × (1 - b + b × |doc|/avgdl))

Where:
  IDF = log((N - df + 0.5) / (df + 0.5))  ← rare terms score higher
  TF  = term frequency in document
  k1  = 1.5 (term frequency saturation)
  b   = 0.75 (document length normalization)
  avgdl = average document length in corpus
```

Plain English: rare terms that appear often in a short document score highest. Handles exact keyword matches that semantic search misses.

## RRF Fusion

```python
def rrf(dense_results, sparse_results, k=60):
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

Why k=60: smoothing constant. A document ranked 1st gets 1/61 ≈ 0.016. A document ranked 60th gets 1/120 ≈ 0.008. Prevents any single first-place result from dominating. No tuning needed — works consistently across domains.

---

# 5. DECISION FLOWCHART

```
START: Choose a vector DB
           │
           ▼
     SAMA data residency
     required?
      Yes │  No
          │   └──────────────────► Pinecone (managed, fastest)
          ▼
     Expected vector count?
          │
    ┌─────┼──────────────┐
    │     │              │
  <1M   1-100M        >100M
    │     │              │
    ▼     ▼              ▼
pgvector  │           Milvus
(simple)  │           (only option)
          │
     Hybrid search
     critical?
      Yes │  No
          │   └──────────────► Qdrant
          ▼                    (best filtering, simplest ops)
       Weaviate
    (native hybrid,
     one query API)
```

---

# 6. WHAT THEY WILL ASK

**"Why Weaviate over Qdrant for Al Rajhi?"**
> "Two reasons. First, native hybrid search: Weaviate runs BM25 and vector search in a single query with RRF fusion built in. Qdrant requires two separate queries plus custom RRF code — more surface area for bugs. Second, for a banking RAG system with a shared codebase, one clean GraphQL API is easier to maintain. The trade-off is Qdrant's filtering is faster — but at our scale of 10M documents, the difference is under 10ms per query, which is within our latency budget. If we were at 100M+ documents we'd revisit."

**"What happens when you outgrow Weaviate?"**
> "We abstracted the vector store behind an interface from day one. Migration to Milvus is a swap of the implementation — the query interface stays identical. We'd trigger that migration when we cross 100M vectors or when Weaviate's single-node HNSW becomes a bottleneck. Milvus's DiskANN index is specifically built for that scale."

**"Why not pgvector — you already use PostgreSQL?"**
> "pgvector is excellent up to about 1M vectors. Beyond that, HNSW query performance degrades and vacuum operations become problematic. We're targeting 10M+ documents at Al Rajhi, which is 30M+ chunks. pgvector would fail at that scale. We keep PostgreSQL for metadata and audit logs — the right tool for relational data — and Weaviate for vectors."

**"What is HNSW and why is it the default everywhere?"**
> "HNSW builds a layered graph — sparse at the top, dense at the bottom. Search starts at the top and greedily follows the closest neighbors downward through layers. It gives O(log n) search time with 95-99% recall at typical ef settings. It's the default because it needs no training data unlike IVF, it handles dynamic inserts well, and it works from 10K to 50M vectors without rethinking the architecture. IVF is better for batch workloads or GPU acceleration at billion scale, but HNSW is the right default for our use case."

---

# 7. QUICK REFERENCE — ONE LINE PER DB

| DB | One line |
|----|----------|
| **Weaviate** | Native hybrid search in one query, module ecosystem, choose for RAG |
| **Qdrant** | Fastest filtering (SIMD pre-filter), simplest ops, choose for pure vector + heavy filtering |
| **Milvus** | Only open-source option at 100M-1B+ vectors, needs K8s |
| **pgvector** | Zero extra infra, SQL joins, max 1-5M vectors before degrading |
| **Pinecone** | Best managed service, data leaves KSA — SAMA violation |
| **FAISS** | Library not a DB, use for prototyping or custom ANN primitives |
