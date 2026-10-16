"""Environment-driven settings shared by the gateway, the worker and the UI.

Secrets have no defaults: services fail fast at startup if they are missing.
See `.env.example` for every variable.
"""

import os
import tempfile


class ConfigError(RuntimeError):
    pass


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Environment variable {name} is required (see .env.example).")
    return value


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# Object storage
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
# Host used in presigned URLs handed to browsers/CLI clients (signatures are host-bound).
MINIO_PUBLIC_ENDPOINT = os.getenv("MINIO_PUBLIC_ENDPOINT", "http://localhost:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "dam-media")
PRESIGNED_URL_TTL_SECONDS = _int("PRESIGNED_URL_TTL_SECONDS", 3600)

# Vector store
QDRANT_HOST = os.getenv("QDRANT_HOST", "vector-db")
QDRANT_PORT = _int("QDRANT_PORT", 6333)

# Messaging
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis-cache:6379/0")
INGESTION_QUEUE = os.getenv("INGESTION_QUEUE", "ingestion")
LLM_QUEUE = os.getenv("LLM_QUEUE", "llm")  # summaries: long LLM jobs, separate from ingestion

# LLM / VLM runtime
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama-engine:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
# Context window requested from Ollama; its default (2048 tokens) silently truncates long prompts.
OLLAMA_NUM_CTX = _int("OLLAMA_NUM_CTX", 8192)
# Character budget for context placed in one prompt (~4 chars/token, leaving room for the answer).
LLM_CONTEXT_CHARS = _int("LLM_CONTEXT_CHARS", 16000)
LLM_TIMEOUT_SECONDS = _int("LLM_TIMEOUT_SECONDS", 180)

# Embedding models
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "all-MiniLM-L6-v2")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")
TEXT_VECTOR_SIZE = 384
IMAGE_VECTOR_SIZE = 512

# Ingestion tuning
CHUNK_SIZE_CHARS = _int("CHUNK_SIZE_CHARS", 500)
CHUNK_OVERLAP_CHARS = _int("CHUNK_OVERLAP_CHARS", 100)
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 32)
UPSERT_BATCH_SIZE = _int("UPSERT_BATCH_SIZE", 256)
FRAME_SAMPLE_SECONDS = float(os.getenv("FRAME_SAMPLE_SECONDS", "2.0"))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
CAPTION_IMAGES = _bool("CAPTION_IMAGES", True)
WORK_DIR = os.getenv("WORK_DIR", os.path.join(tempfile.gettempdir(), "dam-work"))

# Upload limits
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 200)  # multipart through the API / UI
MAX_DIRECT_UPLOAD_MB = _int("MAX_DIRECT_UPLOAD_MB", 5120)  # presigned direct-to-storage uploads
DIRECT_UPLOAD_PART_MB = _int("DIRECT_UPLOAD_PART_MB", 16)

# Auth
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = _int("JWT_EXPIRE_MINUTES", 120)
CORS_ORIGINS = _list("CORS_ORIGINS", "http://localhost:8501")


def jwt_secret() -> str:
    secret = _required("JWT_SECRET")
    if len(secret) < 32:
        raise ConfigError("JWT_SECRET must be at least 32 characters (try: openssl rand -hex 32).")
    return secret


def database_url() -> str:
    return _required("DATABASE_URL")


def celery_broker_url() -> str:
    return _required("CELERY_BROKER_URL")


def minio_credentials() -> tuple[str, str]:
    return _required("MINIO_ACCESS_KEY"), _required("MINIO_SECRET_KEY")
