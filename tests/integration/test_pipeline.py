"""End-to-end against real Qdrant, Postgres and MinIO with the real embedding and Whisper models.

Run with scripts/run-integration-tests.sh (starts docker-compose.test.yml). Ingestion runs inline
via Celery's eager apply(); only the LLM is faked, because Ollama is not part of this setup.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client.http import models as qm

pytestmark = pytest.mark.integration

CORPUS = Path(__file__).resolve().parents[2] / "samples" / "corpus"


@pytest.fixture(scope="module")
def live():
    from app import main, processor

    with pytest.MonkeyPatch.context() as mp:
        # Run the Celery task in-process instead of dispatching to a broker.
        mp.setattr(main, "enqueue_ingestion", lambda asset_id: processor.ingest_asset.apply(args=[asset_id]).id)
        with TestClient(main.app) as client:  # runs the lifespan: tables, bucket, collections
            res = client.post("/api/token", data={"username": "test_admin", "password": "admin-pass-123"})
            client.headers["Authorization"] = f"Bearer {res.json()['access_token']}"
            client.post("/api/admin/reset").raise_for_status()
            yield client


def upload(client, name: str) -> dict:
    with (CORPUS / name).open("rb") as fh:
        res = client.post("/api/upload", files={"file": (name, fh)})
    assert res.status_code == 202, res.text
    asset = client.get(f"/api/assets/{res.json()['asset_id']}").json()
    assert asset["status"] == "ready", asset
    return asset


def count_points(asset_id: str) -> dict:
    from app import vectors

    flt = vectors.asset_filter(asset_id)
    return {c: vectors.qdrant.count(c, count_filter=flt, exact=True).count for c in vectors.COLLECTIONS}


@pytest.fixture(scope="module")
def assets(live):
    return {
        name: upload(live, name)
        for name in ("retention_policy.pdf", "sunset.jpg", "bar_chart.png", "customer_call.wav")
    }


def test_documents_are_chunked_and_searchable(live, assets):
    assert assets["retention_policy.pdf"]["stats"]["pages"] == 3
    hits = live.get("/api/search", params={"query": "Who is the data protection officer?"}).json()["text_matches"]
    assert any(h["filename"] == "retention_policy.pdf" and h["page"] == 3 and "Okafor" in h["text"] for h in hits[:3])


def test_text_query_finds_matching_image(live, assets):
    top = live.get("/api/search", params={"query": "a sunset over the ocean"}).json()["visual_matches"][0]
    assert top["filename"] == "sunset.jpg"
    chart = live.get("/api/search", params={"query": "a bar chart", "media_type": "image"}).json()
    assert chart["visual_matches"][0]["filename"] == "bar_chart.png"


def test_speech_is_transcribed_with_timestamps(live, assets):
    stats = assets["customer_call.wav"]["stats"]
    assert stats["language"] == "en" and stats["transcript_windows"] >= 1
    hit = live.get("/api/search", params={"query": "When is the renewal due?", "media_type": "audio"}).json()
    top = hit["text_matches"][0]
    assert top["filename"] == "customer_call.wav" and top["start"] is not None
    assert "december" in top["text"].lower()


def test_reindex_does_not_duplicate_vectors(live, assets):
    asset_id = assets["retention_policy.pdf"]["id"]
    before = count_points(asset_id)
    assert live.post(f"/api/assets/{asset_id}/reindex").status_code == 202
    assert count_points(asset_id) == before


def test_rag_cites_retrieved_page(live, assets, monkeypatch):
    from app import llm

    monkeypatch.setattr(llm, "chat", AsyncMock(return_value="The DPO is Miriam Okafor [1]."))
    body = live.post("/api/rag", json={"question": "Who handles GDPR deletion requests?"}).json()
    assert body["cited"] == [1]
    assert body["sources"][0]["label"] == "retention_policy.pdf, page 3"


def test_direct_presigned_upload_roundtrip(live):
    data = (CORPUS / "incident_postmortem.pdf").read_bytes()
    plan = live.post("/api/uploads/init", json={"filename": "postmortem.pdf", "size_bytes": len(data)}).json()
    assert len(plan["parts"]) == 1
    put = httpx.put(plan["parts"][0]["url"], content=data)  # straight to MinIO, not via the API
    put.raise_for_status()
    done = live.post(
        f"/api/uploads/{plan['asset_id']}/complete",
        json={"upload_id": plan["upload_id"], "parts": [{"part_number": 1, "etag": put.headers["ETag"]}]},
    )
    assert done.status_code == 202, done.text
    assert live.get(f"/api/assets/{plan['asset_id']}").json()["status"] == "ready"


def test_delete_removes_object_and_vectors(live, assets):
    from app import storage

    asset = assets["bar_chart.png"]
    assert live.delete(f"/api/assets/{asset['id']}").status_code == 204
    assert sum(count_points(asset["id"]).values()) == 0
    with pytest.raises(storage.ClientError):
        storage.head(f"{asset['id']}/bar_chart.png")
    assert live.get(f"/api/assets/{asset['id']}").status_code == 404


def test_presigned_media_url_streams_from_object_store(live, assets):
    url = live.get(f"/api/assets/{assets['sunset.jpg']['id']}/url").json()["url"]
    res = httpx.get(url)
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    assert res.content[:3] == b"\xff\xd8\xff"


def test_payload_indexes_exist(live):
    from app import vectors

    schema = vectors.qdrant.get_collection(vectors.TEXT).payload_schema
    assert {"asset_id", "media_type", "kind"} <= schema.keys()
    assert schema["asset_id"].data_type == qm.PayloadSchemaType.KEYWORD
