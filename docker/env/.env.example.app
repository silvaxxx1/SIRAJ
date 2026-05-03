# ======================== App Info ========================
APP_NAME="Siraj"
APP_VERSION="3.0.0"

# ======================== File Handling ===================
FILE_ALLOWED_TYPES=["text/plain", "application/pdf"]
FILE_MAX_SIZE=10                  # MB
FILE_DEFAULT_CHUNK_SIZE=512000    # bytes (512 KB)

# ======================== Postgres Config =================
# Must match .env.postgres values below
POSTGRES_USERNAME="postgres"
POSTGRES_PASSWORD="your_postgres_password"
POSTGRES_HOST="pgvector"          # Docker service name
POSTGRES_PORT=5432
POSTGRES_MAIN_DATABASE="RagApp"

# ======================== LLM Config =====================
# Generation backend: openai | cohere
GENERATION_BACKEND="openai"
GENERATION_MODEL_ID="gpt-4o-mini"
GENERATION_MODEL_ID_LITERAL=["gpt-4o-mini", "gpt-4o"]

# Embedding backend: open_source_embeddings | openai | cohere
EMBEDDING_BACKEND="open_source_embeddings"
EMBEDDING_MODEL_ID="intfloat/e5-large-v2"
EMBEDDING_MODEL_SIZE=1024

# Provider credentials (only fill what you use)
OPENAI_API_KEY="your_openai_api_key"
OPENAI_API_URL="https://api.openai.com/v1/"   # override for local proxies / Ollama
COHERE_API_KEY="your_cohere_api_key"

# ⚠ Keep these typos — LLMProviderFactory reads these exact key names
INPUT_DAFAULT_MAX_CHARACTERS=1024
GENERATION_DAFAULT_MAX_TOKENS=200
GENERATION_DAFAULT_TEMPERATURE=0.1

# ======================== Vector DB Config ================
# Backend: QDRANT | PGVECTOR
VECTOR_DB_BACKEND="PGVECTOR"
VECTOR_DB_BACKEND_LITERAL=["QDRANT", "PGVECTOR"]
VECTOR_DB_PATH="qdrant_db"              # only used when VECTOR_DB_BACKEND=QDRANT
VECTOR_DB_METHOD="cosine"              # cosine | dot
VECTOR_DB_PGVEC_INDEX_THRESHOLD=200    # row count before HNSW index is built

# ======================== Template Config =================
DEFAULT_LANG="en"
PRIMARY_LANG="en"                      # en | ar
