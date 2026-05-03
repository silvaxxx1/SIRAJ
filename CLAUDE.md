# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

RagApp is a production-grade enterprise RAG system. It supports multilingual documents, hybrid retrieval, multi-layer caching, and a full validation pipeline. Target deployment is fully on-prem.

The `docs/` folder contains the full technical design. **Read `docs/DESIGN_SUMMARY.md` first** before touching any code.

## Commands

All commands run from the repo root. The project uses `uv` for dependency management.

```bash
# Install deps
uv sync

# Run dev server (from repo root, with src/ on path)
cd src && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Lint / format
ruff check src/
black src/

# Docker stack (Postgres + pgvector, Qdrant, Nginx, Prometheus, Grafana)
cd docker && docker compose up -d
cd docker && docker compose down

# Alembic migrations (run from inside src/models/db_schemes/RagApp/)
alembic upgrade head
alembic revision --autogenerate -m "description"
```

Env vars live in `docker/env/`. Copy and fill `.env.app` before running locally — it maps to `src/.env` via `pydantic-settings`.

## Architecture

### Request flow

```
HTTP request
  → FastAPI router (routes/)
  → Controller (controllers/)
      → stores/llm/  (embedding / generation)
      → stores/vectordb/  (vector search)
      → models/db_schemes/  (Postgres via SQLAlchemy async)
```

### Provider pattern

Every backend is behind an abstract interface + factory so it's swappable via `.env`:

| Layer | Interface | Factory | Current providers |
|-------|-----------|---------|-------------------|
| LLM / Embeddings | `LLMInterface` | `LLMProviderFactory` | OpenAI, Cohere, OpenSourceEmbeddings (sentence-transformers) |
| Vector DB | `VectorDBInterface` | `VectorDBProvidersFactory` | Qdrant, PGVector |

To add a new provider: implement the interface, register it in the factory, add the enum value.

### App startup (`main.py`)

`startup_span()` wires everything together and attaches clients to `app.*`:
- `app.db_client` — SQLAlchemy `AsyncSession` sessionmaker (Postgres)
- `app.generation_client` / `app.embedding_client` — LLM provider instances
- `app.vectordb_client` — vector DB provider instance
- `app.template_parser` — locale-aware prompt template loader

Controllers receive these via dependency injection from `request.app.*`.

### Key controllers

- **`NLPController`** — the core RAG pipeline: embed query → vector search → build prompt from locale templates → generate answer
- **`ProcessController`** — file loading (PyMuPDF / TextLoader) + chunking (custom line-based splitter; LangChain `RecursiveCharacterTextSplitter` is available but commented out)
- **`DataController`** — file upload validation, path generation, filename sanitization
- **`ProjectController`** — project CRUD + filesystem path helpers

### Database

SQLAlchemy async models (`DataChunk`, `Project`, `Asset`) live in `src/models/db_schemes/RagApp/schemes/`. Alembic migrations are in `src/models/db_schemes/RagApp/alembic/versions/`. The model layer (`ProjectModel`, `ChunkModel`, `AssetModel`) wraps SQLAlchemy with async query helpers.

### Prompt templates

`TemplateParser` loads `string.Template` values from `stores/llm/templates/locales/<lang>/<group>.py`. Currently `en/` and `ar/` locales exist. Group `rag` holds `system_prompt`, `document_prompt`, and `footer_prompt`.

### Collection naming

Vector DB collections are named `collection_{embedding_size}_{project_id}` — this means changing the embedding model or size invalidates all existing collections.

## Target Architecture vs. Current State

| Component | Target | Current |
|-----------|--------|---------|
| Parser | PyMuPDF + Mistral OCR fallback | PyMuPDF via LangChain loader |
| Chunking | Parent-child (128 child / 512 parent tokens) | Custom line-based fixed-size |
| Embedding | BGE-M3 on-prem (dense+sparse) | Generic sentence-transformers |
| Vector DB | Weaviate (native hybrid search) | Qdrant / PGVector (dense-only) |
| Hybrid search | Dense + BM25 → RRF (k=60) → Top20 | Not implemented |
| Reranker | BGE-reranker-v2-m3 → Top5 | Not implemented |
| Cache | 4-layer (Redis / RedisVL / pgvector / SGLang) | Not implemented |
| LLM serving | SGLang + Llama 3.1 70B | OpenAI / Cohere API calls |
| Validation | Faithfulness judge + PII scan | Not implemented |
| Auth | OIDC + service accounts | Not implemented |

## Implementation Phases

1. **Phase 1 — Ingestion**: Parent-child chunking, PyMuPDF parser, Kafka async queue
2. **Phase 2 — Retrieval**: Weaviate provider, BGE-M3 embeddings, BGE reranker, parent expansion
3. **Phase 3 — Cache**: 4-layer cache module (`stores/cache/`)
4. **Phase 4 — Validation**: Instructor structured output, faithfulness judge, PII scan
5. **Phase 5 — Ops**: OIDC middleware, hash-chained audit logs, RAGAS eval harness

## Docs Reference

| Doc | Use it when... |
|-----|----------------|
| `docs/DESIGN_SUMMARY.md` | Full architecture, every decision with rationale |
| `docs/DESIGN.md` | Deep implementation details per component |
| `docs/VECTOR_DB_BATTLE_CARD.md` | Building the Weaviate provider, HNSW config |
| `docs/CACHE.md` | Building the 4-layer cache |
| `docs/RAG_EVAL.md` | RAGAS evaluation harness |
| `docs/GENAI_LLM_DEPTH.md` | SGLang inference, KV cache, RadixAttention |

## Known Quirks

- `INPUT_DAFAULT_MAX_CHARACTERS` and `GENERATION_DAFAULT_MAX_TOKENS` intentionally keep the typo — `LLMProviderFactory` reads these exact keys from config.
- `app.on_event("startup")` is deprecated in newer FastAPI; migrate to `lifespan` context manager when refactoring `main.py`.
- `disconnect()` in `shutdown_span` is called without `await` — check each provider's implementation before adding async there.
