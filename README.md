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

**Siraj** (سراج — *lantern*) is a production-grade, open-source enterprise RAG system built for on-prem or cloud deployment. It supports multilingual documents (Arabic + English), hybrid retrieval, multi-layer caching, and a full validation pipeline.

---

## Tech Stack

| Layer | Current | Target |
|-------|---------|--------|
| API | FastAPI + Uvicorn | — |
| Vector DB | PGVector (primary) + Qdrant | Weaviate (native hybrid search) |
| Embeddings | sentence-transformers (`e5-large-v2`) | BGE-M3 (dense+sparse, on-prem) |
| LLM | OpenAI / Cohere API | SGLang + Llama 3.1 70B (on-prem) |
| Chunking | Custom line-based splitter | Parent-child (128 child / 512 parent tokens) |
| Database | PostgreSQL + pgvector | — |
| Observability | Prometheus + Grafana | — |
| Queue | — | Kafka (priority topics) |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/silvaxxx1/SIRAJ.git
cd SIRAJ
uv sync
```

### 2. Configure environment

```bash
# Local dev
cp src/.env.example src/.env

# Docker (one file per service)
cp docker/env/.env.example.app               docker/env/.env.app
cp docker/env/.env.example.postgres          docker/env/.env.postgres
cp docker/env/.env.example.grafana           docker/env/.env.grafana
cp docker/env/.env.example.postgres-exporter docker/env/.env.postgres-exporter
```

Fill in credentials and choose your backends — see comments inside each file.

### 3. Start services

```bash
cd docker && docker compose up -d
```

Starts: PostgreSQL + pgvector `5432`, Qdrant `6333`, Nginx `80`, Prometheus `9090`, Grafana `3000`.

### 4. Run migrations

```bash
cd src/models/db_schemes/RagApp
alembic upgrade head
```

### 5. Start the API

```bash
cd src && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI → http://localhost:8000/docs

---

## Switching backends

All backends swap via `.env` — no code changes needed:

```env
VECTOR_DB_BACKEND=PGVECTOR       # PGVECTOR | QDRANT
GENERATION_BACKEND=openai        # openai | cohere
EMBEDDING_BACKEND=open_source_embeddings   # open_source_embeddings | openai | cohere
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

## Linting

```bash
ruff check src/
black src/
```

---

## Roadmap

- [x] PostgreSQL + PGVector (MongoDB removed)
- [x] Dual vector DB support (PGVector + Qdrant) via unified interface
- [x] Custom chunking + async ingestion pipeline
- [x] Prometheus metrics + Grafana dashboards
- [ ] Parent-child chunking (128 child / 512 parent tokens)
- [ ] BGE-M3 embeddings (dense+sparse, on-prem)
- [ ] Weaviate provider + native hybrid search (dense + BM25 → RRF)
- [ ] BGE reranker + parent expansion
- [ ] 4-layer cache (Redis L1 → RedisVL L2 → pgvector L3 → SGLang L4)
- [ ] SGLang serving + Llama 3.1 70B (on-prem)
- [ ] Faithfulness judge + PII scan + structured output
- [ ] OIDC auth + hash-chained audit logs
- [ ] RAGAS evaluation harness

---

## License

MIT — see [LICENSE](./LICENSE)
