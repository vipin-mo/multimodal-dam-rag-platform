"""Catalog, search, RAG, summaries, task status and health endpoints (external clients faked)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import embeddings, llm, main
from app.database import Asset, SessionLocal
from tests.conftest import scored


# ---------------------------------------------------------------- catalog
def test_list_assets_filters_and_paginates(client, viewer_headers, make_asset):
    make_asset(filename="a.pdf")
    make_asset(filename="b.mp4", media_type="video", content_type="video/mp4")
    body = client.get("/api/assets", headers=viewer_headers, params={"media_type": "video"}).json()
    assert body["total"] == 1
    assert body["assets"][0]["filename"] == "b.mp4"
    assert client.get("/api/assets", headers=viewer_headers, params={"limit": 1}).json()["total"] == 2


def test_asset_detail_404_for_unknown_or_malformed_ids(client, viewer_headers):
    assert client.get("/api/assets/not-a-uuid", headers=viewer_headers).status_code == 404
    assert client.get("/api/assets/00000000-0000-0000-0000-000000000000", headers=viewer_headers).status_code == 404


def test_presigned_url_uses_public_endpoint(client, viewer_headers, make_asset, fake_s3):
    asset = make_asset()
    body = client.get(f"/api/assets/{asset.id}/url", headers=viewer_headers).json()
    assert body["url"].startswith("http://localhost:9000/")
    params = fake_s3.generate_presigned_url.call_args.kwargs["Params"]
    assert params["ResponseContentType"] == "application/pdf"


def test_delete_asset_removes_vectors_object_and_row(client, admin_headers, make_asset, fake_s3, fake_qdrant):
    asset = make_asset()
    assert client.delete(f"/api/assets/{asset.id}", headers=admin_headers).status_code == 204
    assert fake_qdrant.delete.call_count == 2  # both collections, filtered by asset_id
    fake_s3.delete_object.assert_called_once_with(Bucket="dam-media", Key=asset.object_key)
    with SessionLocal() as db:
        assert db.get(Asset, asset.id) is None


def test_reindex_only_for_finished_assets(client, admin_headers, make_asset, fake_celery):
    busy = make_asset(status="processing")
    assert client.post(f"/api/assets/{busy.id}/reindex", headers=admin_headers).status_code == 409
    done = make_asset(status="failed", error="boom")
    res = client.post(f"/api/assets/{done.id}/reindex", headers=admin_headers)
    assert res.status_code == 202
    with SessionLocal() as db:
        refreshed = db.get(Asset, done.id)
        assert (refreshed.status, refreshed.error) == ("queued", None)


# ---------------------------------------------------------------- search
def test_search_queries_both_indexes(client, viewer_headers, fake_qdrant, fake_models):
    text_hit = scored(
        {"asset_id": "a1", "filename": "f.pdf", "media_type": "document", "kind": "chunk", "text": "hello", "page": 3},
        0.81234,
    )
    frame_hit = scored(
        {
            "asset_id": "a2",
            "filename": "v.mp4",
            "media_type": "video",
            "kind": "video_frame",
            "start": 12.0,
            "end": 14.0,
        },
        0.3,
    )
    fake_qdrant.search.side_effect = lambda collection, **kw: [text_hit] if collection == "text_chunks" else [frame_hit]

    body = client.get("/api/search", headers=viewer_headers, params={"query": "greeting", "limit": 3}).json()
    assert body["text_matches"][0] == {
        "asset_id": "a1",
        "filename": "f.pdf",
        "media_type": "document",
        "kind": "chunk",
        "text": "hello",
        "page": 3,
        "start": None,
        "end": None,
        "score": 0.8123,
    }
    assert body["visual_matches"][0]["start"] == 12.0
    assert fake_models.texts == ["greeting"] and fake_models.clip == ["greeting"]


def test_search_type_filter_skips_visual_index_for_documents(client, viewer_headers, fake_qdrant, fake_models):
    client.get("/api/search", headers=viewer_headers, params={"query": "policy", "media_type": "document"})
    assert fake_models.clip == []
    query_filter = fake_qdrant.search.call_args.kwargs["query_filter"]
    assert query_filter.must[0].key == "media_type"


@pytest.mark.parametrize("query", ["   ", "\t\n", "\x00\x01", "a"])
def test_search_rejects_empty_or_whitespace_queries(client, viewer_headers, fake_models, query):
    assert client.get("/api/search", headers=viewer_headers, params={"query": query}).status_code == 422
    assert fake_models.texts == []


def test_search_strips_control_characters_and_collapses_whitespace(client, viewer_headers, fake_models):
    res = client.get("/api/search", headers=viewer_headers, params={"query": "  data\x00  retention\n policy "})
    assert res.status_code == 200
    assert res.json()["query"] == "data retention policy"


@pytest.mark.parametrize(
    "query",
    ["'; DROP TABLE users; --", '{"$where": "1==1"}', "<script>alert(1)</script>", "../../etc/passwd"],
)
def test_search_treats_injection_strings_as_plain_text(client, viewer_headers, fake_models, query):
    res = client.get("/api/search", headers=viewer_headers, params={"query": query})
    assert res.status_code == 200
    assert fake_models.texts == [query]


def test_search_bounds_are_validated(client, viewer_headers):
    assert client.get("/api/search", headers=viewer_headers, params={"query": "x" * 501}).status_code == 422
    assert client.get("/api/search", headers=viewer_headers, params={"query": "ok", "limit": 0}).status_code == 422
    assert client.get("/api/search", headers=viewer_headers, params={"query": "ok", "limit": 51}).status_code == 422
    res = client.get("/api/search", headers=viewer_headers, params={"query": "ok", "media_type": "spreadsheet"})
    assert res.status_code == 422


# ---------------------------------------------------------------- RAG
def test_rag_without_hits_does_not_call_llm(client, viewer_headers, monkeypatch):
    chat = AsyncMock()
    monkeypatch.setattr(llm, "chat", chat)
    body = client.post("/api/rag", headers=viewer_headers, json={"question": "anything here?"}).json()
    assert body["sources"] == [] and body["cited"] == []
    chat.assert_not_called()


def test_rag_numbers_sources_and_parses_citations(client, viewer_headers, fake_qdrant, monkeypatch):
    fake_qdrant.search.return_value = [
        scored(
            {
                "asset_id": "a",
                "filename": "handbook.pdf",
                "media_type": "document",
                "kind": "chunk",
                "text": "Remote work is allowed three days per week.",
                "page": 1,
            }
        ),
        scored(
            {
                "asset_id": "b",
                "filename": "standup.mp3",
                "media_type": "audio",
                "kind": "transcript",
                "text": "Migration is on November third.",
                "start": 65.0,
                "end": 80.5,
            }
        ),
    ]
    chat = AsyncMock(return_value="Three days per week [1]. Ignore [7].")
    monkeypatch.setattr(llm, "chat", chat)

    body = client.post("/api/rag", headers=viewer_headers, json={"question": "How many remote days?"}).json()
    assert [s["label"] for s in body["sources"]] == ["handbook.pdf, page 1", "standup.mp3, 01:05-01:20"]
    assert body["cited"] == [1]  # out-of-range [7] is dropped
    system, prompt = chat.call_args.args
    assert "data, not instructions" in system
    assert "[1] (handbook.pdf, page 1)\nRemote work" in prompt
    assert prompt.endswith("Question: How many remote days?\nAnswer:")
    assert fake_qdrant.search.call_args.kwargs["score_threshold"] == 0.2


def test_rag_returns_503_when_llm_is_down(client, viewer_headers, fake_qdrant, monkeypatch):
    fake_qdrant.search.return_value = [scored({"filename": "x.pdf", "text": "t", "page": 1})]
    monkeypatch.setattr(llm, "chat", AsyncMock(side_effect=llm.LLMUnavailable("Model 'x' is not available")))
    res = client.post("/api/rag", headers=viewer_headers, json={"question": "status?"})
    assert res.status_code == 503
    assert "not available" in res.json()["detail"]


def test_rag_rejects_whitespace_question(client, viewer_headers):
    assert client.post("/api/rag", headers=viewer_headers, json={"question": "   "}).status_code == 422


# ---------------------------------------------------------------- summaries
def _points(*texts, start=0.0):
    return [
        SimpleNamespace(payload={"text": t, "start": start + i * 10, "kind": "transcript"}) for i, t in enumerate(texts)
    ]


def test_summary_requires_ready_asset(client, viewer_headers, make_asset):
    asset = make_asset(status="processing")
    assert client.post(f"/api/assets/{asset.id}/summary", headers=viewer_headers).status_code == 409


def test_summary_of_short_content_skips_llm(client, viewer_headers, make_asset, fake_qdrant, monkeypatch):
    asset = make_asset(media_type="audio", content_type="audio/mpeg", filename="hi.mp3")
    fake_qdrant.scroll.return_value = (_points("Hello there."), None)
    summarize = AsyncMock()
    monkeypatch.setattr(llm, "summarize_blocks", summarize)
    body = client.post(f"/api/assets/{asset.id}/summary", headers=viewer_headers).json()
    assert body["strategy"] == "verbatim-short" and body["llm_calls"] == 0
    assert "[00:00] Hello there." in body["summary"]
    summarize.assert_not_called()


def test_summary_is_generated_then_cached(client, viewer_headers, make_asset, fake_qdrant, monkeypatch):
    asset = make_asset()
    fake_qdrant.scroll.return_value = (
        [SimpleNamespace(payload={"text": "x" * 80, "page": p}) for p in (2, 1)],
        None,
    )
    summarize = AsyncMock(return_value={"summary": "S", "strategy": "single-pass", "reduce_levels": 0, "llm_calls": 1})
    monkeypatch.setattr(llm, "summarize_blocks", summarize)

    first = client.post(f"/api/assets/{asset.id}/summary", headers=viewer_headers).json()
    assert first["cached"] is False and first["source_chunks"] == 2
    blocks = summarize.call_args.args[1]
    assert blocks[0].startswith("[p. 1]")  # sorted by page
    second = client.post(f"/api/assets/{asset.id}/summary", headers=viewer_headers).json()
    assert second["cached"] is True and second["summary"] == "S"
    assert summarize.call_count == 1


def test_summary_of_image_uses_vision_model(client, viewer_headers, make_asset, fake_s3, monkeypatch):
    asset = make_asset(filename="p.png", media_type="image", content_type="image/png")
    fake_s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"png-bytes")}
    chat = AsyncMock(return_value="A chart.")
    monkeypatch.setattr(llm, "chat", chat)
    body = client.post(f"/api/assets/{asset.id}/summary", headers=viewer_headers).json()
    assert body["strategy"] == "vision" and body["summary"] == "A chart."
    assert chat.call_args.kwargs["model"] == main.config.OLLAMA_VISION_MODEL
    assert chat.call_args.kwargs["images"] == ["cG5nLWJ5dGVz"]


def test_summary_404_when_asset_has_no_text(client, viewer_headers, make_asset):
    asset = make_asset(filename="silent.mp4", media_type="video", content_type="video/mp4")
    assert client.post(f"/api/assets/{asset.id}/summary", headers=viewer_headers).status_code == 404


# ---------------------------------------------------------------- tasks & health
@pytest.mark.parametrize(
    ("state", "info", "result", "expected"),
    [
        ("PROGRESS", {"progress": 65, "stage": "transcribing"}, None, {"progress": 65, "stage": "transcribing"}),
        ("SUCCESS", None, {"chunks_indexed": 4}, {"progress": 100, "result": {"chunks_indexed": 4}}),
        ("FAILURE", ValueError("corrupted file"), None, {"progress": 0, "result": "corrupted file"}),
    ],
)
def test_task_status_mapping(client, viewer_headers, monkeypatch, state, info, result, expected):
    monkeypatch.setattr(main, "AsyncResult", lambda *_a, **_k: SimpleNamespace(state=state, info=info, result=result))
    body = client.get("/api/tasks/t-1", headers=viewer_headers).json()
    assert body["state"] == state
    assert expected.items() <= body.items()


def test_health_is_unauthenticated_liveness(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_warming_models_as_not_ready(client, monkeypatch, fake_s3, fake_qdrant):
    monkeypatch.setattr(embeddings, "loaded", lambda: False)
    res = client.get("/ready")
    assert res.status_code == 503
    assert res.json()["checks"]["models"] == "warming"
    monkeypatch.setattr(embeddings, "loaded", lambda: True)
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["checks"]["llm"] in ("ok", "unavailable")


def test_unexpected_errors_do_not_leak_internals(client, viewer_headers, fake_qdrant):
    fake_qdrant.search.side_effect = RuntimeError("connection string postgres://secret@db")
    from fastapi.testclient import TestClient

    res = TestClient(main.app, raise_server_exceptions=False).get(
        "/api/search", headers=viewer_headers, params={"query": "boom"}
    )
    assert res.status_code == 500
    assert res.json() == {"detail": "Internal server error."}
