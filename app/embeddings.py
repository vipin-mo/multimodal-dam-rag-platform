"""Lazily loaded embedding models shared by the gateway (queries) and the worker (ingestion).

Both processes load the models locally; nothing is sent over the network for embedding.
"""

import logging
import threading

from app import config

logger = logging.getLogger("dam.embeddings")

_models: dict = {}
_lock = threading.Lock()


def _load(name: str):
    if name not in _models:
        with _lock:
            if name not in _models:
                from sentence_transformers import SentenceTransformer  # heavy import, deferred

                logger.info("Loading embedding model %s", name)
                _models[name] = SentenceTransformer(name, device="cpu")
    return _models[name]


def warm_up() -> None:
    _load(config.TEXT_MODEL_NAME)
    _load(config.CLIP_MODEL_NAME)


def loaded() -> bool:
    return config.TEXT_MODEL_NAME in _models and config.CLIP_MODEL_NAME in _models


def embed_texts(texts: list[str]) -> list[list[float]]:
    """384-d MiniLM vectors for a batch of passages (or a single query)."""
    if not texts:
        return []
    vectors = _load(config.TEXT_MODEL_NAME).encode(
        texts, batch_size=config.EMBED_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def embed_images(images: list) -> list[list[float]]:
    """512-d CLIP vectors for a batch of PIL images."""
    if not images:
        return []
    vectors = _load(config.CLIP_MODEL_NAME).encode(
        images, batch_size=config.EMBED_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()


def embed_clip_text(text: str) -> list[float]:
    """CLIP text tower: places a text query in the same space as images."""
    vector = _load(config.CLIP_MODEL_NAME).encode([text], normalize_embeddings=True, show_progress_bar=False)
    return vector[0].tolist()
