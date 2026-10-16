"""Summary generation (runs on the LLM worker): strategies, ordering, vision path, Celery task."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import llm, processor, summaries
from app.database import Asset, SessionLocal


def run(**kw):
    defaults = {"asset_id": "a1", "filename": "doc.pdf", "media_type": "document", "object_key": "a1/doc.pdf"}
    return asyncio.run(summaries.summarize(**(defaults | kw)))


def points(*payloads):
    return ([SimpleNamespace(payload=p) for p in payloads], None)


def test_short_text_is_returned_verbatim_without_llm(fake_qdrant, monkeypatch):
    fake_qdrant.scroll.return_value = points({"text": "Hello there.", "start": 0.0})
    summarize = AsyncMock()
    monkeypatch.setattr(llm, "summarize_blocks", summarize)
    result = run(media_type="audio")
    assert result == {
        "summary": "[00:00] Hello there.",
        "strategy": "verbatim-short",
        "llm_calls": 0,
        "source_chunks": 1,
    }
    summarize.assert_not_called()


def test_blocks_are_ordered_and_located(fake_qdrant, monkeypatch):
    fake_qdrant.scroll.return_value = points({"text": "b" * 80, "page": 2}, {"text": "a" * 80, "page": 1})
    summarize = AsyncMock(return_value={"summary": "S", "strategy": "single-pass", "reduce_levels": 0, "llm_calls": 1})
    monkeypatch.setattr(llm, "summarize_blocks", summarize)
    result = run()
    blocks = summarize.call_args.args[1]
    assert blocks[0].startswith("[p. 1] a") and blocks[1].startswith("[p. 2] b")
    assert result["source_chunks"] == 2 and result["summary"] == "S"


def test_asset_without_text_cannot_be_summarized(fake_qdrant):
    with pytest.raises(summaries.NothingToSummarize):
        run(media_type="video")


def test_image_summary_uses_vision_model_without_system_turn(fake_s3, monkeypatch):
    fake_s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"png-bytes")}
    chat = AsyncMock(return_value="A chart.")
    monkeypatch.setattr(llm, "chat", chat)
    result = run(media_type="image", filename="p.png")
    assert result["strategy"] == "vision" and result["summary"] == "A chart."
    assert chat.call_args.args[0] is None
    assert chat.call_args.kwargs["images"] == ["cG5nLWJ5dGVz"]


def test_map_reduce_reports_progress(monkeypatch):
    async def fake_chat(system, user, **_):
        return "- point"

    monkeypatch.setattr(llm, "chat", fake_chat)
    seen = []
    blocks = [f"[p. {i}] " + "x" * 90 for i in range(30)]
    result = asyncio.run(
        llm.summarize_blocks("big.pdf", blocks, budget=1000, on_progress=lambda d, p: seen.append((d, p)))
    )
    assert seen[-1] == (result["llm_calls"], result["llm_calls"])
    assert [d for d, _ in seen] == list(range(1, result["llm_calls"] + 1))


def test_summary_task_caches_result_on_asset(make_asset, fake_qdrant, monkeypatch):
    asset = make_asset()
    fake_qdrant.scroll.return_value = points({"text": "x" * 200, "page": 1})
    monkeypatch.setattr(
        llm,
        "summarize_blocks",
        AsyncMock(return_value={"summary": "Done.", "strategy": "single-pass", "reduce_levels": 0, "llm_calls": 1}),
    )
    result = processor.summarize_asset.apply(args=[asset.id]).get()
    assert result["summary"] == "Done."
    with SessionLocal() as db:
        assert db.get(Asset, asset.id).summary["summary"] == "Done."


def test_summary_task_fails_cleanly_when_llm_is_down(make_asset, fake_qdrant, monkeypatch):
    asset = make_asset()
    fake_qdrant.scroll.return_value = points({"text": "x" * 200, "page": 1})
    monkeypatch.setattr(
        llm, "summarize_blocks", AsyncMock(side_effect=llm.LLMUnavailable("Model 'q' is not available"))
    )
    outcome = processor.summarize_asset.apply(args=[asset.id])
    assert outcome.failed() and "not available" in str(outcome.result)
