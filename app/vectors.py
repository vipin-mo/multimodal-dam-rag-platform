"""Qdrant collections, payload schema and helpers.

Two collections, because the embedding spaces differ:
  text_chunks     384-d all-MiniLM-L6-v2   PDF chunks, transcript windows, image captions
  image_features  512-d CLIP ViT-B/32      still images and sampled video frames

Every point carries `asset_id`, `filename`, `media_type` and `kind`, plus location fields
(`page` for documents, `start`/`end` seconds for media).
"""

import logging
import uuid
from collections.abc import Iterator

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from app import config

logger = logging.getLogger("dam.vectors")

TEXT = "text_chunks"
IMAGE = "image_features"
COLLECTIONS = {TEXT: config.TEXT_VECTOR_SIZE, IMAGE: config.IMAGE_VECTOR_SIZE}
INDEXED_FIELDS = ("asset_id", "media_type", "kind")

qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=30)


def ensure_collections() -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    for name, size in COLLECTIONS.items():
        if name not in existing:
            logger.info("Creating collection %s (%d-d, cosine)", name, size)
            try:
                qdrant.create_collection(name, vectors_config=qm.VectorParams(size=size, distance=qm.Distance.COSINE))
            except UnexpectedResponse as exc:  # created concurrently by another replica
                if exc.status_code != 409:
                    raise
        for field in INDEXED_FIELDS:
            qdrant.create_payload_index(name, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD)


def reset_collections() -> None:
    for name in COLLECTIONS:
        qdrant.delete_collection(name)
    ensure_collections()


def point_id(asset_id: str, kind: str, index: int) -> str:
    """Deterministic ids: re-running ingestion overwrites points instead of duplicating them."""
    return str(uuid.uuid5(uuid.UUID(asset_id), f"{kind}:{index}"))


def asset_filter(asset_id: str) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key="asset_id", match=qm.MatchValue(value=asset_id))])


def delete_asset(asset_id: str) -> None:
    for name in COLLECTIONS:
        qdrant.delete(name, points_selector=qm.FilterSelector(filter=asset_filter(asset_id)), wait=True)


def search_filter(media_type: str | None = None, asset_id: str | None = None) -> qm.Filter | None:
    must = []
    if media_type:
        must.append(qm.FieldCondition(key="media_type", match=qm.MatchValue(value=media_type)))
    if asset_id:
        must.append(qm.FieldCondition(key="asset_id", match=qm.MatchValue(value=asset_id)))
    return qm.Filter(must=must) if must else None


def search(
    collection: str,
    vector: list[float],
    limit: int,
    query_filter: qm.Filter | None = None,
    score_threshold: float | None = None,
) -> list[qm.ScoredPoint]:
    return qdrant.search(
        collection,
        query_vector=vector,
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )


def iter_asset_points(collection: str, asset_id: str, page_size: int = 256) -> Iterator[qm.Record]:
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection,
            scroll_filter=asset_filter(asset_id),
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        yield from points
        if offset is None:
            return
