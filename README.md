```
  ███████╗██╗██████╗  █████╗      ██╗
  ██╔════╝██║██╔══██╗██╔══██╗     ██║
  ███████╗██║██████╔╝███████║     ██║
  ╚════██║██║██╔══██╗██╔══██║██   ██║
  ███████║██║██║  ██║██║  ██║╚█████╔╝
  ╚══════╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚════╝

  سراج — Production RAG · On-Prem · Bilingual (AR/EN)
  ─────────────────────────────────────────────────────
  Forked from the RagApp open-source skeleton
```

**Siraj** (سراج — *lantern*) is a production-grade, open-source enterprise RAG system built for fully on-prem deployment. It supports multilingual documents (Arabic + English), hybrid retrieval, multi-layer caching, and a full validation pipeline — designed around SAMA data-residency constraints.

---

## Architecture

### Ingestion Pipeline (Async)

```
Documents ──▶ Kafka (priority topics)
                    │
                    ▼
             Workers (async)
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   PyMuPDF (native PDF)   Mistral OCR (scanned)
          └─────────┬──────────┘
                    ▼
       Parent-Child Chunker
       512-token parent / 128-token child
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   BGE-M3 (dense+sparse)   Parent stored in S3/PG
   embed child chunks
          │
          ▼
   Weaviate (child chunks + metadata)
```

### Query Pipeline (Sync / FastAPI)

```
User ──▶ OIDC Auth ──▶ PII Redact ──▶ Language Detect ──▶ Query Rewrite
                                                                  │
                              ┌───────────────────────────────────┘
                              ▼
                  ┌─ Cache L1: Redis exact (5 min TTL) ─ hit ──▶ return
                  │
                  └─ Cache L2: RedisVL semantic (1 hr TTL) ─ hit ──▶ return
                              │ miss
                              ▼
                  Weaviate Hybrid Search
                  BGE-M3 dense + BM25 ──▶ RRF (k=60) ──▶ Top 20
                              │
                              ▼
                  BGE-reranker-v2-m3 (cross-encoder) ──▶ Top 5
                              │
                              ▼
                  Parent Expansion (child ID ──▶ parent chunk)
                              │
                              ▼
                  SGLang + Llama 3.1 70B (4× A100)
                  RadixAttention caches system prompt KV across ALL requests
                              │
                              ▼
                  Structured Output (Pydantic + Instructor)
                              │
                              ▼
                  Validation
                  ├── Faithfulness judge (Llama 8B, separate endpoint)
                  ├── PII output scan
                  └── Confidence gate ──▶ Guardrails
```

---

## Design Decisions

Every component was chosen for on-prem deployment under SAMA data-residency requirements.

| Component | Choice | Why | Rejected |
|-----------|--------|-----|----------|
| **Parser** | PyMuPDF + Mistral OCR fallback | Speed + Arabic OCR quality | Tesseract (poor Arabic), Azure (data leaves KSA) |
| **Chunking** | Parent-child (128 child / 512 parent tokens) | Precision on retrieval, full context on generation — no trade-off | Fixed-size (can't have both), Semantic (too slow) |
| **Embedding** | BGE-M3, multi-vector OFF | Dense+sparse in one model, on-prem, bilingual; multi-vector is 10× storage for marginal gain | OpenAI (data residency), Jina (no sparse) |
| **Vector DB** | Weaviate | Native hybrid search + RRF in one query | Qdrant (two separate queries + manual RRF merge) |
| **Hybrid Search** | Dense + BM25 → RRF (k=60) | No weight tuning, constant k=60 works across domains | Weighted sum (requires per-domain tuning) |
| **Reranker** | BGE-reranker-v2-m3 | On-prem, Arabic-optimized, free | Cohere (data leaves KSA) |
| **LLM Serving** | SGLang | RadixAttention caches system prompt KV across all requests → 3.2× throughput | vLLM (no cross-request KV caching), TGI (lower throughput) |
| **LLM Model** | Llama 3.1 70B Instruct | Bilingual, 128K context, structured output native | Jais-30B (8K context, poor English reasoning) |
| **Validation Judge** | Llama 3.1 8B (separate endpoint) | 12.5% compute overhead vs 100% if using same 70B | NLI model (cheaper but lower quality) |
| **Cache L1** | Redis exact match | Sub-ms, 5 min TTL, 15–20% hit rate | — |
| **Cache L2** | Redis + RedisVL semantic | Vector similarity search in same store, 1 hr TTL | Memcached (cannot do similarity search) |
| **Cache L3** | pgvector embedding cache | 24 hr TTL, 30–40% hit rate | — |
| **Cache L4** | SGLang RadixAttention | GPU memory, session duration, 95%+ hit on system prompt | — |
| **Async Queue** | Kafka (priority topics) | Disk persistence, 7-day retention, replayable | Redis Streams (RAM-limited, data loss on OOM) |
| **Deployment** | Blue-green + canary + SGLang warmup | Zero downtime, instant rollback, pre-populated KV cache | Rolling (risky with stateful cache) |
| **Auth** | OIDC + service accounts | Bank security requirement | Anonymous access |
| **Audit** | Hash-chained immutable logs | SAMA compliance, 7-year tamper-proof retention | Plain logs (tamperable) |

---

## SLA Targets

| Metric | Target |
|--------|--------|
| P95 end-to-end latency | < 2.5 s (first token at 900 ms) |
| New document ingestion | P95 < 90 s |
| Faithfulness score | > 0.85 |
| Context precision | > 0.75 |
| Cache hit rate (L1 + L2) | 25–35% |
| Availability | 99.9% |
| RTO / RPO | 4 h / 15 min |

---

## Current State vs. Target

Everything is built additively — existing providers (PGVector, Qdrant, OpenAI, Cohere) are never removed. New providers are added through the same interface and activated via `.env`.

| Component | Built (keep) | Adding |
|-----------|--------------------|--------|
| Parser | PyMuPDF via LangChain loader | Mistral OCR fallback for scanned docs |
| Chunking | Custom line-based splitter | Parent-child (128 / 512 tokens) alongside existing |
| Embedding | OpenAI, Cohere, `e5-large-v2` | + BGE-M3 provider (dense+sparse, on-prem) |
| Vector DB | PGVector ✅  Qdrant ✅ | + Weaviate provider (native hybrid search) |
| Hybrid search | Dense-only (PGVector / Qdrant) | + Dense + BM25 → RRF (k=60) → Top 20 via Weaviate |
| Reranker | Not implemented | + BGE-reranker-v2-m3 → Top 5 |
| Parent expansion | Not implemented | + child ID → parent chunk lookup |
| Cache | Not implemented | + 4-layer (Redis / RedisVL / pgvector / SGLang) |
| LLM serving | OpenAI ✅  Cohere ✅ | + SGLang provider (Llama 3.1 70B, on-prem) |
| Structured output | Not implemented | + Pydantic + Instructor |
| Validation | Not implemented | + Faithfulness judge + PII scan + confidence gate |
| Auth | Not implemented | + OIDC + service accounts |
| Audit | Not implemented | + Hash-chained logs, 7-year retention |

---

## Quickstart

### Local dev (limited resources, no GPU required)

Uses a minimal Docker stack (~300 MB RAM) and a small CPU-friendly embedding model.

```bash
# 1. Install deps
uv sync

# 2. Env — pre-configured for CPU-only dev
cp src/.env.local.example src/.env
# Edit src/.env: set POSTGRES_PASSWORD and OPENAI_API_KEY
# (or switch to Ollama — see comments inside the file)

# 3. Postgres + Qdrant only (skip Nginx / Prometheus / Grafana)
cp docker/env/.env.example.postgres docker/env/.env.postgres
# Edit docker/env/.env.postgres: set same password as above
docker compose -f docker/docker-compose.local.yml up -d

# 4. Migrations
cd src/models/db_schemes/RagApp && alembic upgrade head && cd -

# 5. Run
cd src && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI → http://localhost:8000/docs

**Local dev defaults** (`src/.env.local.example`):

| Setting | Value | Why |
|---------|-------|-----|
| `EMBEDDING_MODEL_ID` | `BAAI/bge-small-en-v1.5` | 384-dim, ~130 MB, fast on CPU |
| `EMBEDDING_MODEL_SIZE` | `384` | matches above |
| `GENERATION_BACKEND` | `openai` (gpt-4o-mini) | no local GPU needed |
| `VECTOR_DB_BACKEND` | `PGVECTOR` | already in Docker, zero extra setup |

**Fully offline? Use Ollama:**
```bash
# Install Ollama, then:
ollama pull gemma2:2b   # ~1.5 GB RAM
# In src/.env, set:
# OPENAI_API_URL=http://localhost:11434/v1/
# GENERATION_MODEL_ID=gemma2:2b
# OPENAI_API_KEY=ollama
```

---

### Production / full stack

```bash
# All services (Nginx, Prometheus, Grafana, exporters)
cp docker/env/.env.example.app               docker/env/.env.app
cp docker/env/.env.example.postgres          docker/env/.env.postgres
cp docker/env/.env.example.grafana           docker/env/.env.grafana
cp docker/env/.env.example.postgres-exporter docker/env/.env.postgres-exporter
docker compose -f docker/docker-compose.yml up -d

cd src/models/db_schemes/RagApp && alembic upgrade head && cd -
cd src && uvicorn main:app --host 0.0.0.0 --port 8000
```

Swagger UI → http://localhost:8000/docs

---

## Switching backends

All backends swap via `.env` — no code changes needed:

```env
VECTOR_DB_BACKEND=PGVECTOR       # PGVECTOR | QDRANT | WEAVIATE (once added)
GENERATION_BACKEND=openai        # openai | cohere | sglang (once added)
EMBEDDING_BACKEND=open_source_embeddings   # open_source_embeddings | openai | cohere | bge_m3 (once added)
PRIMARY_LANG=en                  # en | ar
```

---

## Project Structure

```
src/
├── main.py                  # FastAPI entry point + startup wiring
├── controllers/             # NLP pipeline, ingestion, processing, project CRUD
├── stores/
│   ├── llm/providers/       # OpenAI, Cohere, sentence-transformers
│   └── vectordb/providers/  # Qdrant, PGVector
├── models/db_schemes/       # SQLAlchemy models + Alembic migrations
├── routes/                  # FastAPI routers + Pydantic schemas
└── utils/metrics.py         # Prometheus instrumentation
docker/
├── docker-compose.yml
└── env/                     # One .env.example.* per service
docs/                        # Full technical design (start with DESIGN_SUMMARY.md)
```

---

## Observability

Prometheus metrics exposed at `/metrics`, Grafana dashboards in `docker/`:

| Metric | Alert threshold |
|--------|----------------|
| `rag_query_latency_seconds` (P95) | > 2.5 s |
| `rag_faithfulness_score` | drops > 5% below baseline |
| `retrieval_hit_rate` | < 0.90 |
| `pii_detection_count` | any spike |

---

## Roadmap

**Phase 1 — Ingestion**
- [ ] Parent-child chunker (128 child / 512 parent tokens, token-based)
- [ ] PyMuPDF + Mistral OCR fallback parser
- [ ] Kafka async ingestion queue (priority topics)

**Phase 2 — Retrieval**
- [ ] Weaviate provider (added alongside PGVector + Qdrant, not replacing them)
- [ ] BGE-M3 embedding provider (dense+sparse, added alongside existing providers)
- [ ] Hybrid search: dense + BM25 → RRF (k=60) → Top 20 (Weaviate backend)
- [ ] BGE-reranker-v2-m3 → Top 5
- [ ] Parent expansion (child ID → parent chunk)

**Phase 3 — Cache**
- [ ] Redis L1 (exact query, 5 min TTL)
- [ ] RedisVL L2 (semantic similarity, 1 hr TTL)
- [ ] pgvector L3 (embedding cache, 24 hr TTL)

**Phase 4 — Validation**
- [ ] Pydantic + Instructor structured output
- [ ] Llama 3.1 8B faithfulness judge (separate SGLang endpoint)
- [ ] PII output scan + confidence gate

**Phase 5 — Ops**
- [ ] OIDC middleware + service accounts
- [ ] Hash-chained immutable audit logs (SAMA compliance)
- [ ] RAGAS evaluation harness (nightly golden dataset)

---

## License

MIT — see [LICENSE](./LICENSE)
