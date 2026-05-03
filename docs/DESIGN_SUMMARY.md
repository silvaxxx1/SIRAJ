# AL RAJHI BANK — PRODUCTION RAG PIPELINE
## Technical Design Document (Final — Modified Version)
### Senior AI Engineer Interview — April 26, 2026

---

# 1. CORE DECISIONS MATRIX

| Component | Choice | Rationale | Alternatives Rejected |
|-----------|--------|-----------|----------------------|
| **Parser** | PyMuPDF + Mistral OCR fallback | Speed + Arabic OCR quality | Tesseract (poor Arabic), Azure (data residency) |
| **Chunking** | Parent-child (128 child / 512 parent) | Precision + context — no trade-off | Fixed-size (can't have both), Semantic (too slow) |
| **Embedding** | BGE-M3 (multi-vector OFF) | Dense+sparse in one model, on-prem, bilingual | OpenAI (data residency), Jina (no sparse) |
| **Vector DB** | **Weaviate** | Native hybrid search, single query, GraphQL | Qdrant (separate queries + manual RRF) |
| **Hybrid Search** | Weaviate native + RRF | Integrated, no tuning, k=60 constant | Weighted sum (tuning hell) |
| **Reranker** | BGE-reranker-v2-m3 | On-prem, Arabic-optimized, free | Cohere (data leaves KSA) |
| **LLM Serving** | **SGLang** | RadixAttention caches system prompt across all requests | vLLM (no cross-request caching), TGI (lower throughput) |
| **LLM Model** | Llama 3.1 70B Instruct | Bilingual reasoning, 128K context, on-prem | Jais-30B (8K context, poor English), GPT-4 (data residency) |
| **Validation Judge** | Llama 3.1 8B (separate endpoint) | 12.5% compute overhead vs 2x with 70B | NLI model (cheaper, lower quality) |
| **Caching L1** | Redis (exact query) | Sub-ms, 5min TTL, 15-20% hit rate | None |
| **Caching L2** | Redis + RedisVL (semantic similar) | Vector similarity search in same store, 1hr TTL | Memcached (can't do similarity) |
| **Caching L3** | pgvector (embedding) | 24hr TTL, 30-40% hit rate | None |
| **Caching L4** | SGLang RadixAttention (KV cache) | GPU memory, session duration, 95%+ for system prompt | None |
| **Async Queue** | Kafka (priority topics) | Disk persistence, replayable, 7-day retention | Redis Streams (RAM-limited, loses data on OOM) |
| **Deployment** | Blue-green + canary + warmup | Zero downtime, instant rollback, SGLang cache preload | Rolling (risky) |
| **Auth** | OIDC + service account | Production security, no anonymous access | None |
| **Audit** | Hash-chained immutable logs | SAMA compliance, 7-year retention | Plain logs (tamperable) |

---

# 2. ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION (Async)                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Documents ──▶ Kafka (priority) ──▶ Workers ──▶ PyMuPDF/Mistral ──▶ Parent-Child   │
│                                                    └─┬─┘              └──────┬──────┘ │
│                                                      │                        │       │
│                                                      ▼                        ▼       │
│                                              BGE-M3 (dense+sparse)    Weaviate + S3   │
│                                              multi-vector OFF                         │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              QUERY (Sync/FastAPI)                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  User ──▶ Auth(OIDC) ──▶ PII redact ──▶ Language detect ──▶ Query rewrite          │
│                                              │                                       │
│                                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  CACHE L1 (Redis exact) ──miss──▶ CACHE L2 (RedisVL semantic)               │    │
│  │         │                              │                                     │    │
│  │      hit                             hit                                    │    │
│  │         │                              │                                     │    │
│  │      return ◀─────────────────── skip retrieval, use cached chunks          │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                              │ miss                                  │
│                                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  WEAVIATE HYBRID (native)                                                   │    │
│  │  Dense(BGE-M3) + BM25 ──RRF(k=60)──▶ Top20                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                              │                                       │
│                                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  BGE-RERANKER-V2-M3 (cross-encoder) ──▶ Top5                                │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                              │                                       │
│                                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  PARENT EXPANSION (child ID → parent chunk from S3)                         │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                              │                                       │
│                                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  SGLANG + LLAMA 3.1 70B (4× A100)                                           │    │
│  │  RadixAttention caches system prompt KV across ALL requests                 │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                              │                                       │
│                                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  STRUCTURED OUTPUT (Pydantic + Instructor)                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                              │                                       │
│                                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  VALIDATION                                                                 │    │
│  │  Faithfulness (Llama 8B judge, separate 1× A100) ──▶ Citation check        │    │
│  │  PII output scan ──▶ Confidence gate ──▶ Guardrails                         │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 3. COMPONENT BREAKDOWN

## 3.1 Parsing

```python
# Decision: PyMuPDF default, Mistral fallback for scans
if doc.is_native_pdf:
    result = PyMuPDF.parse(doc)
elif doc.confidentiality == 'HIGH':
    result = Tesseract.parse(doc, lang='ara+eng')  # on-prem
else:
    result = MistralOCR.parse(doc, language='ara+eng')  # higher quality
```

## 3.2 Chunking

```python
# Decision: Parent-child (128 token child, 512 token parent)
class ParentChildChunker:
    def chunk(self, doc):
        parent_chunks = split_by_tokens(doc, size=512)  # stored, returned to LLM
        for parent in parent_chunks:
            child_chunks = split_by_tokens(parent, size=128, overlap=15)  # indexed
            child.parent_id = parent.id
```

## 3.2.1 Weaviate Object Schema & Metadata Indexing

```python
from weaviate.classes.config import Configure, Property, DataType, Tokenization, VectorDistances

# Child collection — searched, never returned directly to LLM
client.collections.create(
    name="Chunks",
    vectorizer_config=Configure.Vectorizer.none(),       # embed externally with BGE-M3
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE,
        ef_construction=200,   # build quality (set once)
        max_connections=64,    # M param — good for 1024-dim BGE-M3 vectors
        ef=100,                # query recall vs latency (tunable)
    ),
    inverted_index_config=Configure.inverted_index(
        bm25_b=0.75,           # length normalization
        bm25_k1=1.2,           # TF saturation
    ),
    properties=[
        # BM25-searchable fields
        Property(name="content",      data_type=DataType.TEXT,
                 tokenization=Tokenization.WHITESPACE,   # Arabic-safe
                 index_searchable=True,  index_filterable=False),
        Property(name="header",       data_type=DataType.TEXT,
                 tokenization=Tokenization.WHITESPACE,
                 index_searchable=True,  index_filterable=False),

        # Filterable fields — pre-filter HNSW search space
        Property(name="doc_type",     data_type=DataType.TEXT,
                 index_filterable=True,  index_searchable=False),
        Property(name="language",     data_type=DataType.TEXT,
                 index_filterable=True,  index_searchable=False),
        Property(name="department",   data_type=DataType.TEXT,
                 index_filterable=True,  index_searchable=False),
        Property(name="clearance",    data_type=DataType.INT,
                 index_filterable=True),
        Property(name="page_number",  data_type=DataType.INT,
                 index_filterable=True),
        Property(name="ingested_at",  data_type=DataType.DATE,
                 index_filterable=True),

        # Stored only — no index, just returned with results
        Property(name="parent_id",    data_type=DataType.TEXT,
                 index_filterable=False, index_searchable=False),
        Property(name="doc_id",       data_type=DataType.TEXT,
                 index_filterable=False, index_searchable=False),
        Property(name="source_file",  data_type=DataType.TEXT,
                 index_filterable=False, index_searchable=False),
        Property(name="chunk_index",  data_type=DataType.INT,
                 index_filterable=False),
        Property(name="token_count",  data_type=DataType.INT,
                 index_filterable=False),
        Property(name="ocr_used",     data_type=DataType.BOOL,
                 index_filterable=False),
    ]
)

# Parent collection — returned to LLM, never searched directly
client.collections.create(
    name="Documents",
    vector_index_config=Configure.VectorIndex.hnsw(skip=True),  # no vector index
    properties=[
        Property(name="content",     data_type=DataType.TEXT,
                 index_searchable=False, index_filterable=False),
        Property(name="doc_id",      data_type=DataType.TEXT,
                 index_filterable=True,  index_searchable=False),
        Property(name="page_number", data_type=DataType.INT,
                 index_filterable=True),
        Property(name="header",      data_type=DataType.TEXT,
                 index_searchable=False, index_filterable=False),
    ]
)

# Indexing rule: index_searchable=True  → enters BM25 inverted index
#               index_filterable=True   → enters inverted index for pre-filtering
#               both False              → stored only, zero index overhead
```

**Metadata indexing principle:** index only what you filter or search on. Indexing everything wastes memory and slows ingestion.

## 3.3 Embedding

```python
# Decision: BGE-M3, multi-vector OFF
class BGEM3Embedder:
    use_dense = True
    use_sparse = True
    use_multi_vector = False  # OFF: 10x storage, 5x latency for marginal gain
    # Re-enable only if retrieval recall < 0.80 on golden dataset
    
    def embed(self, text):
        return {dense: vector, sparse: lexical_weights}
```

## 3.4 Vector Database

```python
# Decision: WEAVIATE (not Qdrant)
# Reason: Native hybrid search with RRF in ONE query
# Qdrant requires: dense search → sparse search → manual RRF merge

client.query.get("Document").with_hybrid(
    query=user_query,
    alpha=0.5,
    vector=query_embedding,
    fusion_type='rrf',
    fusion_k=60
).with_where({
    "operator": "And",
    "operands": [
        {"path": ["department"], "operator": "Equal", "value": user.department},
        {"path": ["clearance"], "operator": "LessThanEqual", "value": user.clearance}
    ]
}).with_limit(50).do()

# Authentication: OIDC + service account (no anonymous access)
```

## 3.5 Reranking

```python
# Decision: BGE-reranker-v2-m3 (on-prem, free, Arabic-optimized)
# NOT Cohere: data leaves KSA (SAMA violation)

class BGEReranker:
    def rerank(self, query, documents):
        pairs = [[query, doc.text] for doc in documents]
        scores = cross_encoder(pairs)  # ~150-300ms for 20 docs
        return top_k(documents, scores, k=5)
```

## 3.6 LLM Serving (Inference Engine)

```python
# Decision: SGLANG (not vLLM)
# Reason: RadixAttention caches system prompt KV across ALL requests
# For Al Rajhi: system prompt = 2000 tokens (65% of input)
# vLLM: processes 2000 tokens per request → 2B tokens/day
# SGLang: processes 2000 tokens once → cached for all subsequent requests

sglang.launch_server(
    model="Llama-3.1-70B-Instruct",
    tp=4,  # 4× A100
    radix_cache_size=10_000_000,  # Cache 10M unique prefixes
    disable_radix_cache=False  # CRITICAL: this is our advantage
)
```

## 3.7 LLM Model

```python
# Decision: LLAMA 3.1 70B (not Jais-30B)
# Jais-30B: better Arabic (95% vs 85%), but:
#   - 8K context (cuts off large policies)
#   - Poor English reasoning (bank has bilingual documents)
#   - Poor structured output (JSON mode broken)

# Llama 3.1 70B: 128K context, bilingual, structured output native
# Fallback: Llama 3.1 8B (quantized) for overload scenarios
```

## 3.8 Caching (4-Layer)

```python
# L1: Redis exact query (5 min TTL)
cache_key = f"l1:{user.department}:{hash(query)}"

# L2: Redis + RedisVL semantic similarity (1 hr TTL)
# Why RedisVL not Memcached: Memcached cannot do similarity search
semantic_index.search(query_embedding, threshold=0.95)

# L3: pgvector embedding cache (24 hr TTL)
SELECT embedding FROM query_embeddings WHERE query_hash = $1

# L4: SGLang RadixAttention (KV cache, GPU memory)
# Handled automatically by SGLang, no code required
```

## 3.9 Validation Judge

```python
# Decision: LLAMA 3.1 8B (separate endpoint, 1× A100)
# NOT same 70B model: would double compute cost

class FaithfulnessJudge:
    def __init__(self):
        self.judge = SGLangClient(endpoint="http://judge:30001")  # 8B model
    
    def check(self, answer, context):
        # Generator (70B): 120ms
        # Judge (8B): 15ms → 12.5% overhead, acceptable
        score = self.judge.generate(f"Score faithfulness 0-1: {answer}\nContext: {context}")
        return float(score)

# Fallback for high load: NLI model (DeBERTa, 5ms, CPU)
```

## 3.10 Async Ingestion

```python
# Decision: KAFKA (not Redis Streams)
# Reason: Disk persistence (7 days), replayable, no data loss on OOM

producer.send(f'ingestion.{priority}', value={doc_id, s3_path})  # priority: high/medium/low

# Worker:
consumer = KafkaConsumer(f'ingestion.{priority}', max_poll_records=10)
for msg in consumer:
    process(msg)
    consumer.commit()  # manual commit after success
```

## 3.11 Deployment

```python
# Blue-green + canary + warmup

# Step 1: Deploy green environment
# Step 2: Canary 5% traffic for 30 min
# Step 3: If RAGAS metrics hold → 50% → 100%
# Step 4: Keep old blue for 1 hour (instant rollback)

# CRITICAL: SGLang warmup before taking live traffic
def warmup_sglang():
    top_queries = fetch_top_100_queries_last_7_days()
    for q in top_queries:
        sglang.generate(q, max_tokens=1)  # Populates RadixAttention cache
    # Without this: first 100 users get 800ms instead of 250ms
```

## 3.11b Document Updates & Deduplication

**Document update:**
- Standard docs (product, FAQ): delete old chunks by `doc_id` → ingest new version → ~2 min stale window
- Regulatory docs (SAMA circulars): shadow index → atomic swap → zero stale window

**Deduplication on ingest:**
- Exact: MD5 hash check — skip if identical file already indexed
- Near-duplicate: MinHash LSH (threshold 0.85) — version management, supersede old doc
- Chunk-level: hash each chunk — skip duplicate paragraphs that appear across multiple docs

## 3.12 Observability

**Prometheus metrics (Grafana dashboards):**
- `rag_query_latency_seconds` — P95 target: 2.5s, broken down per stage
- `rag_faithfulness_score` — target: >0.85, alert if drops >5% below baseline
- `retrieval_hit_rate` — alert if <0.90 (index issue)
- `pii_detection_count` — labeled by type and severity

**SAMA-compliant audit (hash-chained, 7-year retention, object-locked S3):**
```python
audit_log.write({
    'timestamp': utcnow(),
    'user_id': user.id,
    'query': query,
    'retrieved_chunk_ids': [...],
    'answer': answer,
    'hash': sha256(previous_hash + event_string)  # tamper-proof chain
})
```

**Evaluation:**
- Offline: 500-query golden dataset → RAGAS nightly (faithfulness, context precision, answer relevancy, recall)
- Online signals: escalation rate, citation click-through, query reformulation rate
- Alert: any metric drops >5% from baseline → page on-call, consider rollback
- Component isolation: retrieval metrics tracked separately from generation — know exactly which stage degraded

---

# 4. TRADE-OFF SUMMARY

| Decision | Chosen Over | Why |
|----------|-------------|-----|
| **Weaviate** | Qdrant | Native hybrid + RRF in one query (not separate + manual merge) |
| **SGLang** | vLLM | RadixAttention caches system prompt across ALL requests (3.2x throughput) |
| **BGE-M3 (multi-vector OFF)** | multi-vector ON | 10x storage, 5x latency for marginal gain. Re-enable only if recall <0.80 |
| **Llama 70B** | Jais-30B | Jais has 8K context (cuts policies) + poor English reasoning |
| **Llama 8B judge** | Same 70B judge | 12.5% compute overhead vs 100% (separate endpoint) |
| **RedisVL L2** | Memcached L2 | Memcached cannot do similarity search (fundamental requirement) |
| **Kafka** | Redis Streams | Disk persistence (no data loss on OOM), replayable |
| **SGLang warmup** | None | Without it: first 100 users get 800ms instead of 250ms |
| **OIDC + service account** | Anonymous access | Bank security requirement |

---

# 5. SLAs & TARGETS

| Metric | Target |
|--------|--------|
| P95 end-to-end latency | <2.5 seconds (first token at 900ms) |
| New document ingestion | P95 <90 seconds |
| Faithfulness score | >0.85 |
| Context precision | >0.75 |
| Cache hit rate (L1+L2) | 25-35% |
| Availability | 99.9% |
| RTO / RPO | 4 hours / 15 minutes |

---

**END OF DOCUMENT**