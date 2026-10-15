"""Worker pipelines with real pypdf/OpenCV/Pillow on generated files; models and Qdrant faked."""

import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fpdf import FPDF
from PIL import Image

from app import config, processor, storage
from app.database import Asset, SessionLocal

ASSET_ID = "6f1c1d1e-0000-4000-8000-000000000001"


def make_job(path, media_type, progress=None):
    return processor.Job(ASSET_ID, path.name, media_type, str(path), progress or (lambda p, s: None))


def upserted(fake_qdrant, collection):
    points = []
    for call in fake_qdrant.upsert.call_args_list:
        if call.args[0] == collection:
            points.extend(call.kwargs["points"])
    return points


@pytest.fixture
def pdf_file(tmp_path):
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for page in range(3):
        pdf.add_page()
        pdf.multi_cell(0, 7, " ".join(f"Page {page + 1} sentence {i} about retention." for i in range(25)))
    pdf.add_page()  # blank page: no text layer
    path = tmp_path / "policy.pdf"
    pdf.output(str(path))
    return path


def test_pdf_is_chunked_per_page_and_upserted_in_batches(pdf_file, fake_qdrant, fake_models, monkeypatch):
    monkeypatch.setattr(config, "UPSERT_BATCH_SIZE", 4)
    stats = processor.process_document(make_job(pdf_file, "document"))

    points = upserted(fake_qdrant, "text_chunks")
    assert stats["pages"] == 4 and stats["pages_without_text"] == 1
    assert stats["chunks_indexed"] == len(points) > 3
    assert fake_qdrant.upsert.call_count == -(-len(points) // 4)  # ceil: batched upserts
    assert {p.payload["page"] for p in points} == {1, 2, 3}
    assert all(len(p.payload["text"]) <= config.CHUNK_SIZE_CHARS for p in points)
    assert all(p.payload["asset_id"] == ASSET_ID and p.payload["kind"] == "chunk" for p in points)
    assert len(fake_models.texts) == len(points)  # one embedding per chunk, no extra calls

    # deterministic ids: a second run produces identical point ids (overwrite, not duplicate)
    first_ids = [p.id for p in points]
    fake_qdrant.upsert.reset_mock()
    processor.process_document(make_job(pdf_file, "document"))
    assert [p.id for p in upserted(fake_qdrant, "text_chunks")] == first_ids


def test_video_frames_sampled_every_n_seconds(tmp_path, fake_qdrant, fake_models, monkeypatch):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(100):  # 10 s at 10 fps
        writer.write(np.full((48, 64, 3), i * 2, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(config, "FRAME_SAMPLE_SECONDS", 2.0)
    monkeypatch.setattr(config, "EMBED_BATCH_SIZE", 2)

    seen = []
    count = processor.sample_frames(make_job(path, "video", lambda p, s: seen.append(p)), 10.0, (5, 50))
    points = upserted(fake_qdrant, "image_features")
    assert count == 5 == fake_models.images
    assert [p.payload["start"] for p in points] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert points[0].payload["end"] == 2.0 and points[0].payload["kind"] == "video_frame"
    assert seen == sorted(seen) and max(seen) <= 50


def test_transcription_is_windowed_with_timestamps(tmp_path, fake_qdrant, fake_models, monkeypatch):
    segments = [
        SimpleNamespace(text=f" Sentence {i} of the meeting.", start=i * 3.0, end=i * 3.0 + 3) for i in range(30)
    ]
    fake_whisper = SimpleNamespace(transcribe=lambda path, **kw: (iter(segments), SimpleNamespace(language="en")))
    monkeypatch.setattr(processor, "whisper_model", lambda: fake_whisper)
    monkeypatch.setattr(config, "CHUNK_SIZE_CHARS", 120)

    windows, language = processor.transcribe(make_job(tmp_path / "a.mp3", "audio"), 90.0, (5, 95))
    points = upserted(fake_qdrant, "text_chunks")
    assert language == "en" and windows == len(points) > 1
    assert points[0].payload["start"] == 0.0
    assert all(p.payload["kind"] == "transcript" and p.payload["end"] > p.payload["start"] for p in points)


def test_image_gets_clip_vector_and_optional_caption(tmp_path, fake_qdrant, fake_models, monkeypatch):
    path = tmp_path / "chart.png"
    Image.new("RGB", (300, 200), "blue").save(path)
    monkeypatch.setattr(config, "CAPTION_IMAGES", True)
    monkeypatch.setattr(processor.llm, "chat_sync", lambda *a, **k: "A solid blue rectangle.")

    stats = processor.process_image(make_job(path, "image"))
    assert stats["captioned"] and stats["caption"] == "A solid blue rectangle."
    assert upserted(fake_qdrant, "image_features")[0].payload["kind"] == "image"
    caption_point = upserted(fake_qdrant, "text_chunks")[0]
    assert caption_point.payload == {
        "asset_id": ASSET_ID,
        "filename": "chart.png",
        "media_type": "image",
        "kind": "caption",
        "text": "A solid blue rectangle.",
    }


def test_image_ingestion_survives_caption_failure(tmp_path, fake_qdrant, fake_models, monkeypatch):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (50, 50), "red").save(path)
    monkeypatch.setattr(config, "CAPTION_IMAGES", True)

    def down(*a, **k):
        raise processor.llm.LLMUnavailable("vision model not pulled")

    monkeypatch.setattr(processor.llm, "chat_sync", down)
    stats = processor.process_image(make_job(path, "image"))
    assert stats["captioned"] is False
    assert upserted(fake_qdrant, "text_chunks") == []


# ---------------------------------------------------------------- full task
@pytest.fixture
def stored_asset(make_asset, monkeypatch, tmp_path):
    """An asset whose 'download' copies a local file."""

    def _make(source, **fields):
        asset = make_asset(id=ASSET_ID, status="queued", **fields)
        monkeypatch.setattr(storage, "download", lambda key, dest: shutil.copy(source, dest))
        monkeypatch.setattr(config, "WORK_DIR", str(tmp_path / "work"))
        return asset

    return _make


def test_ingest_task_marks_asset_ready_with_stats(pdf_file, stored_asset, fake_qdrant, fake_models):
    stored_asset(pdf_file, filename="policy.pdf")
    result = processor.ingest_asset.apply(args=[ASSET_ID]).get()
    assert result["chunks_indexed"] > 0 and result["processing_seconds"] >= 0
    assert fake_qdrant.delete.call_count == 2  # old vectors cleared before indexing
    with SessionLocal() as db:
        asset = db.get(Asset, ASSET_ID)
        assert asset.status == "ready" and asset.stats["pages"] == 4


def test_ingest_task_rejects_content_that_does_not_match_extension(tmp_path, stored_asset, fake_qdrant):
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"\x89PNG\r\n\x1a\n not a pdf")
    stored_asset(fake, filename="fake.pdf")
    outcome = processor.ingest_asset.apply(args=[ASSET_ID])
    assert outcome.failed()
    with SessionLocal() as db:
        asset = db.get(Asset, ASSET_ID)
        assert asset.status == "failed" and "does not match" in asset.error
    fake_qdrant.upsert.assert_not_called()


def test_ingest_task_for_deleted_asset_is_skipped(fake_qdrant):
    assert processor.ingest_asset.apply(args=[ASSET_ID]).get() == {"skipped": "asset no longer exists"}


def test_workdir_is_removed_after_task(pdf_file, stored_asset, fake_qdrant, fake_models, tmp_path):
    stored_asset(pdf_file, filename="policy.pdf")
    processor.ingest_asset.apply(args=[ASSET_ID]).get()
    assert list((tmp_path / "work").iterdir()) == []
