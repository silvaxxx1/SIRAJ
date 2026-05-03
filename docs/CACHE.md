# Caching Layer — Complete Reference

---

## 0. Core Principle

> **Generation is the bottleneck. Cache anything that avoids it.**

Full RAG request cost breakdown:

| Step | Latency | Cacheable? |
|---|---|---|
| BGE-M3 embedding | ~10ms | Yes — L3 |
| Weaviate retrieval | ~10ms | Partially |
| Reranker | ~50ms | Partially |
| Llama 70B generation | ~2-5s | Yes — L1, L2 |
| System prompt KV | ~200ms | Yes — L4 |

**Target:** 30-35% of requests never reach the LLM.

---

## 1. Architecture Overview

```
Query
  ↓
L1 Redis exact ──────────────────────────────→ HIT: return <1ms
  ↓ miss
L2 RedisVL semantic ─────────────────────────→ HIT: return ~5ms
  ↓ miss
L3 pgvector embedding cache
  ├── HIT:  skip BGE-M3 → go to retrieval
  └── MISS: run BGE-M3 → store embedding
  ↓
Weaviate retrieval
  ↓
Reranker (BGE reranker-v2-m3)
  ↓
L4 SGLang RadixAttention (system prompt KV always cached)
  ↓
Llama 70B generates answer
  ↓
8B judge validates
  ↓
Write to L1 + L2
  ↓
Return answer (~2-5s on full miss)
```

---

## 2. L1 — Redis Exact Match

### What it caches
Complete answers for **identical** queries.

### How it works

```python
import hashlib
import redis
import json

r = redis.Redis(host='localhost', port=6379)

def l1_lookup(query: str) -> str | None:
    key = f"l1:{hashlib.md5(query.encode()).hexdigest()}"
    cached = r.get(key)
    return json.loads(cached) if cached else None

def l1_store(query: str, answer: dict, ttl: int = 300):
    key = f"l1:{hashlib.md5(query.encode()).hexdigest()}"
    r.setex(key, ttl, json.dumps(answer))
```

### Parameters

| Parameter | Value | Reason |
|---|---|---|
| Key | MD5(query_string) | Fast, deterministic, collision-safe |
| TTL | 5 minutes | Financial data changes — stale rate answers are compliance risk |
| Hit rate | 15-20% | Repeated identical queries in customer service patterns |
| Latency | <1ms | Pure in-memory key lookup |

### Why 5min TTL (not longer)

A cached answer to "current mortgage rate" that is 2 hours old is a **compliance risk** — rates, policies, and balances change. Staleness outweighs cache efficiency for financial data.

### Why Redis (not Memcached)

Redis is chosen here for consistency — same store used for L2. Also supports atomic operations, persistence, and pub/sub for cache invalidation.

---

## 3. L2 — RedisVL Semantic Cache

### What it caches
Complete answers for **semantically similar** queries — not identical, but close enough to return the same answer.

### The Problem it Solves

```
Cached:  "ما هي شروط التمويل العقاري؟"
New:     "ما هو معدل تمويل المنازل؟"

Exact match: MISS
Semantic:    cosine similarity = 0.94 → HIT
```

### How it works

```python
from redisvl.extensions.llmcache import SemanticCache

cache = SemanticCache(
    name="rag_semantic_cache",
    redis_url="redis://localhost:6379",
    distance_threshold=0.08,   # cosine distance < 0.08 = similarity > 0.92
    ttl=3600                   # 1 hour
)

def l2_lookup(query: str, embedding: list[float]) -> str | None:
    results = cache.check(prompt=query, vector=embedding)
    return results[0]["response"] if results else None

def l2_store(query: str, embedding: list[float], answer: dict):
    cache.store(
        prompt=query,
        response=json.dumps(answer),
        vector=embedding
    )
```

### Parameters

| Parameter | Value | Reason |
|---|---|---|
| Similarity threshold | 0.92 cosine | Below this, queries are too different — wrong answer risk |
| TTL | 1 hour | Similar queries can drift in meaning over time |
| Hit rate | +10-15% on top of L1 | Near-duplicate customer questions |
| Latency | ~5ms | Embedding lookup in Redis vector index |

### Threshold Tuning

```
threshold too high (0.99) → almost no hits, behaves like L1
threshold too low  (0.80) → wrong answers returned for different questions

For financial/legal domain: err toward 0.92-0.95 (conservative)
For general chitchat: 0.85-0.90 is acceptable
```

### Why Not Memcached

Memcached is pure key-value — no vector operations, no similarity search. Semantic caching is architecturally impossible with Memcached.

### Why Same Redis Instance as L1

- Single connection pool
- Single TTL management system
- RedisVL is a Redis extension — runs natively on the same server
- No extra infrastructure

---

## 4. L3 — pgvector Embedding Cache

### What it caches
**Embeddings** — not answers. Skips BGE-M3 inference on repeated queries.

### Why Different from L1/L2

L1 and L2 cache **answers**. L3 caches **intermediate computation**.

```
L1/L2 hit → skip everything (BGE-M3 + retrieval + reranker + LLM)
L3 hit    → skip BGE-M3 only → still runs retrieval, reranker, LLM
```

### How it works

```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://localhost/ragdb")
register_vector(conn)

def l3_lookup(query: str) -> list[float] | None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT embedding FROM embedding_cache
            WHERE query_hash = %s
              AND created_at > NOW() - INTERVAL '24 hours'
        """, (hashlib.md5(query.encode()).hexdigest(),))
        row = cur.fetchone()
        return row[0] if row else None

def l3_store(query: str, embedding: list[float]):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO embedding_cache (query_hash, query_text, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (query_hash) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                created_at = NOW()
        """, (hashlib.md5(query.encode()).hexdigest(), query, embedding))
    conn.commit()
```

### Schema

```sql
CREATE TABLE embedding_cache (
    id          SERIAL PRIMARY KEY,
    query_hash  VARCHAR(32) UNIQUE NOT NULL,
    query_text  TEXT NOT NULL,
    embedding   vector(1024),         -- BGE-M3 dense dimension
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON embedding_cache USING ivfflat (embedding vector_cosine_ops);
```

### Parameters

| Parameter | Value | Reason |
|---|---|---|
| TTL | 24 hours | Embeddings don't change unless model changes |
| Hit rate | 30-40% | Common banking terms appear repeatedly |
| Latency saved | ~10ms | BGE-M3 inference skipped |
| Key | MD5(query_string) | Exact match only — no fuzzy embedding lookup here |

### Why pgvector (not Redis)

- Longer TTL requires persistence — Redis can be volatile
- pgvector integrates with existing Postgres infrastructure
- SQL queries allow analytics on cached queries (what's being asked most?)
- ACID guarantees on cache writes

---

## 5. L4 — SGLang RadixAttention

### What it caches
**KV cache** (Key-Value attention pairs) for shared token prefixes — specifically the system prompt.

### Why This is Different

L1-L3 are application-level caches. L4 is a **GPU-level cache** built into the inference engine.

### How it works

```
System prompt (2000 tokens):
[t1, t2, t3, ... t2000]
        ↓
Stored in radix tree on GPU

Request 1: [system_prompt][user_query_A] → cache hit on 2000 tokens
Request 2: [system_prompt][user_query_B] → cache hit on 2000 tokens
Request 3: [system_prompt][user_query_C] → cache hit on 2000 tokens

Each request skips 2000 tokens of attention computation.
```

### The Math

```
Without RadixAttention:
  Cost per request = (system_prompt_tokens + query_tokens) × forward_pass_cost

With RadixAttention:
  Cost per request = query_tokens × forward_pass_cost
  System prompt computed ONCE, shared across all requests

At 1000 requests/day with 2000-token system prompt:
  Saved = 1000 × 2000 tokens = 2B tokens of wasted computation
```

### Parameters

| Parameter | Value |
|---|---|
| Scope | Cross-request, GPU memory |
| TTL | Until GPU memory pressure evicts it |
| Hit rate | ~100% on system prompt prefix |
| Latency saved | ~200ms per request |

### Why SGLang Over vLLM

vLLM's KV cache is per-request only — system prompt recomputed every time.
SGLang's RadixAttention persists KV across requests — system prompt computed once.

**vLLM wins when:** diverse system prompts per user/tenant — no shared prefix benefit.
**SGLang wins when:** fixed large system prompt shared across all requests — your case.

---

## 6. Hit Rate Economics

| Layer | Hit Rate | Latency Saved | What is Avoided |
|---|---|---|---|
| L1 exact | 15-20% | ~3s | Everything |
| L2 semantic | +10-15% | ~2.5s | Generation |
| L3 embedding | 30-40% | ~10ms | BGE-M3 inference |
| L4 RadixAttention | ~100% | ~200ms | System prompt KV recompute |

**Combined L1+L2:** ~30-35% of requests never reach Llama 70B.

---

## 7. Cache Invalidation

### When to Invalidate

| Event | Cache Layer | Action |
|---|---|---|
| Policy/rate change | L1, L2 | Flush affected keys or wait for TTL |
| Model upgrade (BGE-M3) | L3 | Full flush — embeddings changed |
| System prompt change | L4 | SGLang restart — radix tree cleared |
| Incorrect answer found | L1, L2 | Delete specific key |

### Targeted Invalidation

```python
def invalidate_query(query: str):
    key = f"l1:{hashlib.md5(query.encode()).hexdigest()}"
    r.delete(key)
    # L2 requires vector search to find similar keys — use TTL instead

def invalidate_pattern(pattern: str):
    # e.g. invalidate all murabaha-related answers after rate change
    keys = r.scan_iter(f"l1:*")
    # requires storing original query alongside hash for pattern matching
```

### Emergency Full Flush

```bash
# L1 + L2 (Redis)
redis-cli FLUSHDB

# L3 (pgvector)
psql -c "TRUNCATE embedding_cache;"

# L4 (SGLang)
systemctl restart sglang
```

---

## 8. TTL Strategy Summary

| Layer | TTL | Reason |
|---|---|---|
| L1 Redis exact | 5 minutes | Financial rates/policies change frequently |
| L2 RedisVL semantic | 1 hour | Similar queries can drift in meaning |
| L3 pgvector embedding | 24 hours | Embeddings stable unless model changes |
| L4 SGLang RadixAttention | GPU memory | Evicted only under memory pressure |

**Rule:** shorter TTL for answers, longer TTL for embeddings. Answers carry compliance risk, embeddings do not.

---

## 9. Full Request Flow with Cache

```python
async def rag_request(query: str) -> dict:

    # L1 — exact match
    if answer := l1_lookup(query):
        return {"answer": answer, "cache": "L1"}

    # L2 — semantic match (needs embedding first)
    embedding = l3_lookup(query)           # L3 check
    if embedding is None:
        embedding = bge_m3.embed(query)    # BGE-M3 inference
        l3_store(query, embedding)         # store for next time

    if answer := l2_lookup(query, embedding):
        return {"answer": answer, "cache": "L2"}

    # Full pipeline
    chunks = weaviate.hybrid_search(query, embedding)
    reranked = bge_reranker.rerank(query, chunks)
    answer = llama_70b.generate(query, reranked)  # L4 active here

    # Validate
    verdict = llama_8b.judge(query, answer, reranked)
    if not verdict.passed:
        return fallback_response(query)

    # Write to L1 + L2
    l1_store(query, answer)
    l2_store(query, embedding, answer)

    return {"answer": answer, "cache": "MISS"}
```

---

## 10. Monitoring

Key metrics to track:

```
cache_hit_rate_l1        → target: 15-20%
cache_hit_rate_l2        → target: 10-15%
cache_hit_rate_l3        → target: 30-40%
cache_miss_total         → full pipeline invocations
cache_staleness_events   → TTL expiry before use (too long TTL signal)
l1_memory_usage_mb       → Redis memory pressure
l3_table_size_rows       → pgvector table growth
sglang_prefix_hit_rate   → RadixAttention effectiveness
```

---

## 11. Interview One-Liners

**On cache design philosophy:**
> "Four layers, each targeting a different cost. L1 skips everything, L2 skips generation on near-duplicates, L3 skips embedding inference, L4 skips system prompt recomputation on the GPU. Combined they eliminate ~30% of full pipeline calls."

**On TTL choices:**
> "TTLs are short on answers, long on embeddings. A stale answer on a financial rate is a compliance risk. A stale embedding is just a minor quality issue — same semantic meaning, slightly different vector."

**On L2 threshold:**
> "The 0.92 similarity threshold is conservative for financial domain. A chatbot can afford 0.85. A bank cannot — wrong answers on similar-but-different queries carry legal risk."

**On why pgvector for L3:**
> "Embeddings need 24-hour persistence — Redis is volatile under memory pressure. pgvector gives me persistence, SQL analytics on what's being cached, and it reuses existing Postgres infrastructure."

**On SGLang L4:**
> "RadixAttention is not a cache I manage — it's built into the inference engine. My system prompt is 2000 tokens, shared across every request. SGLang computes it once and keeps the KV pairs in GPU memory. At scale that's the difference between 1x and 2x throughput on the same hardware."
