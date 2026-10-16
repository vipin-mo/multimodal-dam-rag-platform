"""Celery ingestion worker: object storage -> parse/transcribe/sample -> embed -> Qdrant.

One task per asset, addressed by asset id. Re-running a task is safe: existing vectors for the
asset are deleted first and point ids are deterministic.
"""

import asyncio
import base64
import contextlib
import io
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

from celery import Celery
from celery.signals import worker_process_init
from qdrant_client.http import models as qm

from app import config, embeddings, llm, media, storage, summaries, vectors
from app.chunking import chunk_text, window_segments
from app.database import Asset, SessionLocal, update_asset

logger = logging.getLogger("dam.worker")
logging.getLogger("httpx").setLevel(logging.WARNING)

celery_app = Celery("dam-worker", broker=config.celery_broker_url(), backend=config.CELERY_RESULT_BACKEND)
celery_app.conf.update(
    task_default_queue=config.INGESTION_QUEUE,
    task_track_started=True,
    task_acks_late=True,  # a crashed worker's task is redelivered, not lost
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # long tasks: don't let one child hoard the queue
    worker_max_tasks_per_child=50,
    task_time_limit=3 * 3600,
    result_expires=7 * 24 * 3600,
    task_routes={"dam.summarize_asset": {"queue": config.LLM_QUEUE}},
)


@worker_process_init.connect
def _configure_threads(**_):
    """Splits CPU threads between prefork children instead of each child using every core."""
    threads = int(os.getenv("WORKER_THREADS", "0"))
    if threads > 0:
        import torch

        torch.set_num_threads(threads)


_whisper = None


def whisper_model():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper %s (%s)", config.WHISPER_MODEL_SIZE, config.WHISPER_COMPUTE_TYPE)
        _whisper = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=config.WHISPER_COMPUTE_TYPE,
            cpu_threads=int(os.getenv("WORKER_THREADS", "0")),
        )
    return _whisper


@dataclass
class Job:
    asset_id: str
    filename: str
    media_type: str
    path: str
    report: Callable[[int, str], None]

    def payload(self, **extra) -> dict:
        return {"asset_id": self.asset_id, "filename": self.filename, "media_type": self.media_type, **extra}


def batched(items: Iterable, size: int) -> Iterator[list]:
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


# ====================================================================== task
@celery_app.task(bind=True, name="dam.ingest_asset")
def ingest_asset(self, asset_id: str) -> dict:
    with SessionLocal() as db:
        asset = db.get(Asset, asset_id)
        if asset is None:
            return {"skipped": "asset no longer exists"}
        filename, key, media_type = asset.filename, asset.object_key, asset.media_type

    def report(progress: int, stage: str) -> None:
        self.update_state(state="PROGRESS", meta={"progress": progress, "stage": stage})

    update_asset(asset_id, status="processing", error=None, task_id=self.request.id)
    os.makedirs(config.WORK_DIR, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="ingest-", dir=config.WORK_DIR)
    started = time.perf_counter()
    try:
        report(2, "downloading")
        ext = media.extension(filename)
        # Local path derives from the asset id only; the user-supplied name never touches the filesystem.
        local_path = os.path.join(workdir, f"source{ext}")
        storage.download(key, local_path)
        with open(local_path, "rb") as fh:
            if not media.sniff_ok(ext, fh.read(64)):
                raise media.UnsupportedMediaError("File content does not match its extension.")

        vectors.delete_asset(asset_id)  # idempotent re-indexing
        job = Job(asset_id, filename, media_type, local_path, report)
        stats = PIPELINES[media_type](job)
        stats["processing_seconds"] = round(time.perf_counter() - started, 2)

        with SessionLocal() as db:
            still_exists = db.get(Asset, asset_id) is not None
        if not still_exists:  # deleted while processing: drop what we just wrote
            vectors.delete_asset(asset_id)
            return {"skipped": "asset deleted during processing"}

        caption = stats.pop("caption", None)
        update_asset(
            asset_id,
            status="ready",
            stats=stats,
            caption=caption,
            duration_seconds=stats.get("duration_seconds"),
            summary=None,
        )
        logger.info("Ingested %s (%s) in %.1fs: %s", filename, media_type, stats["processing_seconds"], stats)
        return stats
    except Exception as exc:
        logger.exception("Ingestion failed for %s", asset_id)
        with contextlib.suppress(LookupError):  # asset deleted mid-task
            update_asset(asset_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
        raise
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@celery_app.task(bind=True, name="dam.summarize_asset")
def summarize_asset(self, asset_id: str) -> dict:
    """Generates and caches an asset summary (consumed by the llm-worker service)."""
    with SessionLocal() as db:
        asset = db.get(Asset, asset_id)
        if asset is None:
            raise LookupError("Asset no longer exists.")
        args = (asset.id, asset.filename, asset.media_type, asset.object_key)

    shown = 1

    def report(done: int, planned: int) -> None:
        nonlocal shown  # map-reduce can add a level mid-way; never let the bar move backwards
        shown = max(shown, int(100 * done / max(planned, 1)) - 1)
        self.update_state(state="PROGRESS", meta={"progress": shown, "stage": f"LLM call {done + 1} of {planned}"})

    self.update_state(state="PROGRESS", meta={"progress": 1, "stage": "summarizing"})
    result = asyncio.run(summaries.summarize(*args, on_progress=report))
    with contextlib.suppress(LookupError):
        update_asset(asset_id, summary=result)
    return {"asset_id": asset_id, **result}


# ====================================================================== documents
def process_document(job: Job) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(job.path)
    total_pages = len(reader.pages)
    chunks: list[tuple[int, str]] = []
    empty_pages = 0
    for page_no, page in enumerate(reader.pages, start=1):
        page_chunks = chunk_text(page.extract_text() or "", config.CHUNK_SIZE_CHARS, config.CHUNK_OVERLAP_CHARS)
        empty_pages += not page_chunks
        chunks.extend((page_no, c) for c in page_chunks)
        if page_no % 10 == 0:
            job.report(5 + int(35 * page_no / total_pages), "extracting text")

    indexed = 0
    for batch in batched(enumerate(chunks), config.UPSERT_BATCH_SIZE):
        embedded = embeddings.embed_texts([text for _, (_, text) in batch])
        vectors.qdrant.upsert(
            vectors.TEXT,
            points=[
                qm.PointStruct(
                    id=vectors.point_id(job.asset_id, "chunk", i),
                    vector=vec,
                    payload=job.payload(kind="chunk", text=text, page=page),
                )
                for (i, (page, text)), vec in zip(batch, embedded, strict=True)
            ],
            wait=True,
        )
        indexed += len(batch)
        job.report(40 + int(55 * indexed / max(1, len(chunks))), "embedding")

    # Pages with no text layer (scans) need OCR, which this pipeline does not do.
    return {"pages": total_pages, "chunks_indexed": indexed, "pages_without_text": empty_pages}


# ====================================================================== images
def _jpeg_b64(image, max_side: int = 1024) -> str:
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def caption_image(image) -> str | None:
    """Short VLM description, indexed as text so images are reachable from text search and RAG."""
    try:
        text = llm.chat_sync(
            None, llm.CAPTION_PROMPT, model=config.OLLAMA_VISION_MODEL, images=[_jpeg_b64(image)], max_tokens=160
        )
    except Exception as exc:  # noqa: BLE001 - captions are best-effort
        logger.warning("Captioning skipped: %s", exc)
        return None
    if not text.strip():
        logger.warning("Vision model %s returned an empty caption", config.OLLAMA_VISION_MODEL)
        return None
    return llm.trim_to_sentence(text)


def process_image(job: Job) -> dict:
    from PIL import Image, ImageOps

    with Image.open(job.path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    width, height = image.size
    job.report(20, "embedding image")
    vector = embeddings.embed_images([image])[0]
    vectors.qdrant.upsert(
        vectors.IMAGE,
        points=[
            qm.PointStruct(
                id=vectors.point_id(job.asset_id, "image", 0), vector=vector, payload=job.payload(kind="image")
            )
        ],
        wait=True,
    )

    caption = None
    if config.CAPTION_IMAGES:
        job.report(50, "captioning")
        caption = caption_image(image)
    if caption:
        vectors.qdrant.upsert(
            vectors.TEXT,
            points=[
                qm.PointStruct(
                    id=vectors.point_id(job.asset_id, "caption", 0),
                    vector=embeddings.embed_texts([caption])[0],
                    payload=job.payload(kind="caption", text=caption),
                )
            ],
            wait=True,
        )
    return {"width": width, "height": height, "image_vectors": 1, "captioned": bool(caption), "caption": caption}


# ====================================================================== audio / video
def probe(path: str) -> tuple[float | None, bool]:
    """(duration in seconds, has an audio stream) via PyAV."""
    import av

    with av.open(path) as container:
        duration = container.duration / 1_000_000 if container.duration else None
        return duration, bool(container.streams.audio)


def sample_frames(job: Job, duration: float | None, progress_span: tuple[int, int]) -> int:
    """Embeds one frame every FRAME_SAMPLE_SECONDS. Frames in between are grab()-ed, which skips
    the colour conversion and copy, and sampled frames are downscaled before batching."""
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(job.path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if 0 < fps <= 240 else 25.0
    step = max(1, round(fps * config.FRAME_SAMPLE_SECONDS))
    lo, hi = progress_span
    images: list = []
    stamps: list[float] = []
    indexed = 0

    def flush() -> None:
        nonlocal indexed
        if not images:
            return
        embedded = embeddings.embed_images(images)
        vectors.qdrant.upsert(
            vectors.IMAGE,
            points=[
                qm.PointStruct(
                    id=vectors.point_id(job.asset_id, "frame", indexed + i),
                    vector=vec,
                    payload=job.payload(kind="video_frame", start=ts, end=ts + config.FRAME_SAMPLE_SECONDS),
                )
                for i, (ts, vec) in enumerate(zip(stamps, embedded, strict=True))
            ],
            wait=True,
        )
        indexed += len(images)
        if duration:
            job.report(lo + int((hi - lo) * min(1.0, stamps[-1] / duration)), "sampling frames")
        images.clear()
        stamps.clear()

    try:
        frame_index = 0
        while True:
            if frame_index % step == 0:
                ok, frame = capture.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((448, 448))  # CLIP sees 224x224; keep batches small in memory
                images.append(image)
                stamps.append(frame_index / fps)
                if len(images) >= config.EMBED_BATCH_SIZE:
                    flush()
            elif not capture.grab():
                break
            frame_index += 1
        flush()
    finally:
        capture.release()
    return indexed


def transcribe(job: Job, duration: float | None, progress_span: tuple[int, int]) -> tuple[int, str | None]:
    """faster-whisper decodes the audio track straight from the file (PyAV), with VAD to skip silence."""
    lo, hi = progress_span
    segments_iter, info = whisper_model().transcribe(job.path, beam_size=1, vad_filter=True)
    segments = []
    for segment in segments_iter:
        segments.append(segment)
        if duration and len(segments) % 10 == 0:
            job.report(lo + int((hi - lo) * min(1.0, segment.end / duration)), "transcribing")

    windows = window_segments(segments, config.CHUNK_SIZE_CHARS)
    for offset, batch in enumerate(batched(windows, config.UPSERT_BATCH_SIZE)):
        embedded = embeddings.embed_texts([w.text for w in batch])
        base = offset * config.UPSERT_BATCH_SIZE
        vectors.qdrant.upsert(
            vectors.TEXT,
            points=[
                qm.PointStruct(
                    id=vectors.point_id(job.asset_id, "transcript", base + i),
                    vector=vec,
                    payload=job.payload(kind="transcript", text=w.text, start=w.start, end=w.end),
                )
                for i, (w, vec) in enumerate(zip(batch, embedded, strict=True))
            ],
            wait=True,
        )
    return len(windows), getattr(info, "language", None)


def process_video(job: Job) -> dict:
    duration, has_audio = probe(job.path)
    frames = sample_frames(job, duration, (5, 50 if has_audio else 95))
    windows, language = transcribe(job, duration, (50, 95)) if has_audio else (0, None)
    return {
        "duration_seconds": round(duration, 2) if duration else None,
        "frames_indexed": frames,
        "transcript_windows": windows,
        "language": language,
        "has_audio": has_audio,
    }


def process_audio(job: Job) -> dict:
    duration, _ = probe(job.path)
    windows, language = transcribe(job, duration, (5, 95))
    return {
        "duration_seconds": round(duration, 2) if duration else None,
        "transcript_windows": windows,
        "language": language,
    }


PIPELINES: dict[str, Callable[[Job], dict]] = {
    "document": process_document,
    "image": process_image,
    "video": process_video,
    "audio": process_audio,
}
