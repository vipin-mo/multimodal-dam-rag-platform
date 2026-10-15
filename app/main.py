"""API gateway: auth, uploads, asset catalog, search, RAG and summaries."""

import asyncio
import base64
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from celery import Celery
from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import REGISTRY
from prometheus_client.core import GaugeMetricFamily
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app import config, embeddings, llm, media, storage, vectors
from app.database import Asset, SessionLocal, User, engine, get_db, init_db
from app.security import authenticate, create_access_token, require_admin, require_user

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dam.gateway")

MAX_UPLOAD_BYTES = config.MAX_UPLOAD_MB * 2**20
MAX_DIRECT_UPLOAD_BYTES = config.MAX_DIRECT_UPLOAD_MB * 2**20
MediaType = Literal["document", "image", "video", "audio"]

celery_client = Celery("dam-gateway", broker=config.celery_broker_url(), backend=config.CELERY_RESULT_BACKEND)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(init_db)
    await run_in_threadpool(storage.ensure_bucket)
    await run_in_threadpool(vectors.ensure_collections)
    # Model weights load in the background: /health answers immediately and /ready reports
    # "warming" until both encoders are in memory, so slow cold starts don't fail liveness probes.
    warmup = asyncio.create_task(run_in_threadpool(embeddings.warm_up))
    logger.info("Gateway started; embedding models warming up in the background")
    yield
    warmup.cancel()


app = FastAPI(title="Multimodal DAM Gateway", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,  # bearer tokens travel in headers, not cookies
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(
    app, latency_lowr_buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
).expose(app, include_in_schema=False)


class AssetCatalogCollector:
    """Exports asset counts by status and media type, read from Postgres at scrape time."""

    def collect(self):
        gauge = GaugeMetricFamily("dam_assets", "Assets in the catalog", labels=["status", "media_type"])
        try:
            with SessionLocal() as db:
                rows = db.query(Asset.status, Asset.media_type, func.count()).group_by(Asset.status, Asset.media_type)
                for status_value, media_type, count in rows:
                    gauge.add_metric([status_value, media_type], count)
        except Exception:  # noqa: BLE001 - never fail a scrape because the DB is down
            logger.warning("Could not read asset counts for metrics")
        yield gauge


REGISTRY.register(AssetCatalogCollector())


@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    """Rejects oversized multipart uploads before the body is read."""
    if request.url.path == "/api/upload":
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 2**20:
            return JSONResponse(
                status_code=413,
                content={"detail": f"File exceeds {config.MAX_UPLOAD_MB} MB; use the direct upload API."},
            )
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


# ====================================================================== helpers
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def clean_query(raw: str) -> str:
    """Strips control characters and collapses whitespace; rejects empty queries."""
    query = re.sub(r"\s+", " ", _CONTROL_CHARS.sub(" ", raw)).strip()
    if len(query) < 2:
        raise HTTPException(status_code=422, detail="Query must contain at least 2 visible characters.")
    return query


def fmt_ts(seconds: float | None) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def source_label(payload: dict) -> str:
    name = payload.get("filename", "unknown")
    if payload.get("page") is not None:
        return f"{name}, page {payload['page']}"
    if payload.get("kind") == "caption":
        return f"{name}, image description"
    if payload.get("start") is not None:
        return f"{name}, {fmt_ts(payload['start'])}-{fmt_ts(payload.get('end'))}"
    return name


def format_hit(point) -> dict:
    p = point.payload or {}
    return {
        "asset_id": p.get("asset_id"),
        "filename": p.get("filename"),
        "media_type": p.get("media_type"),
        "kind": p.get("kind"),
        "text": p.get("text"),
        "page": p.get("page"),
        "start": p.get("start"),
        "end": p.get("end"),
        "score": round(float(point.score), 4),
    }


def enqueue_ingestion(asset_id: str) -> str:
    task = celery_client.send_task("dam.ingest_asset", args=[asset_id], queue=config.INGESTION_QUEUE)
    return task.id


def get_asset(asset_id: str, db: Session = Depends(get_db)) -> Asset:
    try:
        uuid.UUID(asset_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Asset not found.")
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    return asset


class _LimitedReader:
    """File wrapper that counts bytes and aborts once the limit is crossed."""

    def __init__(self, fileobj, limit: int):
        self.fileobj, self.limit, self.count = fileobj, limit, 0

    def read(self, size: int = -1) -> bytes:
        data = self.fileobj.read(size)
        self.count += len(data)
        if self.count > self.limit:
            raise OverflowError
        return data


# ====================================================================== health
@app.get("/health", include_in_schema=False)
def health():
    """Liveness: the process is up and serving."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: dependencies reachable and embedding models loaded."""
    checks: dict[str, str] = {}

    def probe(name: str, fn) -> None:
        try:
            fn()
            checks[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - surfaced in the response
            checks[name] = f"error: {type(exc).__name__}"

    def db_ping():
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))

    probe("postgres", db_ping)
    probe("qdrant", vectors.qdrant.get_collections)
    probe("object_store", lambda: storage.s3.head_bucket(Bucket=storage.BUCKET))
    checks["models"] = "ok" if embeddings.loaded() else "warming"
    try:
        httpx.get(f"{config.OLLAMA_HOST}/api/tags", timeout=2).raise_for_status()
        checks["llm"] = "ok"
    except httpx.HTTPError:
        checks["llm"] = "unavailable"  # optional: search and ingestion still work

    required = ("postgres", "qdrant", "object_store", "models")
    is_ready = all(checks[k] == "ok" for k in required)
    return JSONResponse(status_code=200 if is_ready else 503, content={"ready": is_ready, "checks": checks})


# ====================================================================== auth
@app.post("/api/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 password flow (form fields `username`, `password`); also used by Swagger's Authorize."""
    user = authenticate(db, form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=401, detail="Incorrect username or password.", headers={"WWW-Authenticate": "Bearer"}
        )
    return {"access_token": create_access_token(user.username, user.role), "token_type": "bearer", "role": user.role}


@app.get("/api/me")
def me(user: User = Depends(require_user)):
    return {"username": user.username, "role": user.role}


# ====================================================================== uploads
@app.post("/api/upload", status_code=status.HTTP_202_ACCEPTED)
def upload_asset(file: UploadFile = File(...), user: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Multipart upload for files up to MAX_UPLOAD_MB. Streams to object storage in 16 MiB parts
    (the request body is spooled to disk by Starlette, never held in memory) and queues ingestion."""
    safe_name = media.sanitize_filename(file.filename or "")
    try:
        media_type, content_type = media.classify(safe_name)
    except media.UnsupportedMediaError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    head = file.file.read(64)
    file.file.seek(0)
    if not media.sniff_ok(media.extension(safe_name), head):
        raise HTTPException(status_code=415, detail="File content does not match its extension.")

    asset_id = str(uuid.uuid4())
    key = media.object_key(asset_id, safe_name)
    reader = _LimitedReader(file.file, MAX_UPLOAD_BYTES)
    try:
        storage.upload_stream(reader, key, content_type)
    except OverflowError:
        raise HTTPException(status_code=413, detail=f"File exceeds {config.MAX_UPLOAD_MB} MB.")

    asset = Asset(
        id=asset_id,
        filename=safe_name,
        object_key=key,
        media_type=media_type,
        content_type=content_type,
        size_bytes=reader.count,
        status="queued",
        uploaded_by=user.username,
    )
    db.add(asset)
    db.commit()
    asset.task_id = enqueue_ingestion(asset_id)
    db.commit()
    return {"asset_id": asset_id, "task_id": asset.task_id, "filename": safe_name, "status": asset.status}


class DirectUploadInit(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)


class CompletedPart(BaseModel):
    part_number: int = Field(ge=1, le=10_000)
    etag: str = Field(min_length=1, max_length=128)


class DirectUploadComplete(BaseModel):
    upload_id: str = Field(min_length=1, max_length=1024)
    parts: list[CompletedPart] = Field(min_length=1, max_length=10_000)


class DirectUploadAbort(BaseModel):
    upload_id: str = Field(min_length=1, max_length=1024)


@app.post("/api/uploads/init", status_code=201)
def init_direct_upload(req: DirectUploadInit, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Starts a presigned multipart upload. The client PUTs each part straight to object storage,
    so large files never pass through the gateway or the UI."""
    safe_name = media.sanitize_filename(req.filename)
    try:
        media_type, content_type = media.classify(safe_name)
    except media.UnsupportedMediaError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    if req.size_bytes > MAX_DIRECT_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {config.MAX_DIRECT_UPLOAD_MB} MB.")

    asset_id = str(uuid.uuid4())
    key = media.object_key(asset_id, safe_name)
    upload_id = storage.start_multipart(key, content_type)
    part_size, part_count = storage.plan_parts(req.size_bytes)
    db.add(
        Asset(
            id=asset_id,
            filename=safe_name,
            object_key=key,
            media_type=media_type,
            content_type=content_type,
            size_bytes=req.size_bytes,
            status="pending_upload",
            uploaded_by=user.username,
            stats={"upload_id": upload_id},
        )
    )
    db.commit()
    return {
        "asset_id": asset_id,
        "upload_id": upload_id,
        "part_size": part_size,
        "parts": storage.presigned_part_urls(key, upload_id, part_count),
    }


def _pending_upload(asset: Asset, upload_id: str) -> None:
    if asset.status != "pending_upload" or (asset.stats or {}).get("upload_id") != upload_id:
        raise HTTPException(status_code=409, detail="No matching upload in progress for this asset.")


@app.post("/api/uploads/{asset_id}/complete", status_code=202)
def complete_direct_upload(
    req: DirectUploadComplete,
    _: User = Depends(require_admin),
    asset: Asset = Depends(get_asset),
    db: Session = Depends(get_db),
):
    _pending_upload(asset, req.upload_id)
    storage.complete_multipart(asset.object_key, req.upload_id, [p.model_dump() for p in req.parts])

    actual_size = storage.head(asset.object_key)["ContentLength"]
    problem = None
    if actual_size != asset.size_bytes:
        problem = (400, f"Uploaded {actual_size} bytes but {asset.size_bytes} were declared.")
    elif not media.sniff_ok(media.extension(asset.filename), storage.read_head(asset.object_key)):
        problem = (415, "File content does not match its extension.")
    if problem:
        storage.delete_object(asset.object_key)
        db.delete(asset)
        db.commit()
        raise HTTPException(status_code=problem[0], detail=problem[1])

    asset.status, asset.stats = "queued", {}
    db.commit()
    asset.task_id = enqueue_ingestion(asset.id)
    db.commit()
    return {"asset_id": asset.id, "task_id": asset.task_id, "filename": asset.filename, "status": asset.status}


@app.post("/api/uploads/{asset_id}/abort", status_code=204)
def abort_direct_upload(
    req: DirectUploadAbort,
    _: User = Depends(require_admin),
    asset: Asset = Depends(get_asset),
    db: Session = Depends(get_db),
):
    _pending_upload(asset, req.upload_id)
    storage.abort_multipart(asset.object_key, req.upload_id)
    db.delete(asset)
    db.commit()


# ====================================================================== asset catalog
@app.get("/api/assets")
def list_assets(
    media_type: MediaType | None = None,
    status_filter: str | None = Query(None, alias="status", max_length=16),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    query = db.query(Asset)
    if media_type:
        query = query.filter(Asset.media_type == media_type)
    if status_filter:
        query = query.filter(Asset.status == status_filter)
    total = query.count()
    rows = query.order_by(Asset.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "assets": [a.to_dict() for a in rows]}


@app.get("/api/assets/{asset_id}")
def asset_detail(_: User = Depends(require_user), asset: Asset = Depends(get_asset)):
    return asset.to_dict()


@app.get("/api/assets/{asset_id}/url")
def asset_url(_: User = Depends(require_user), asset: Asset = Depends(get_asset)):
    """Short-lived presigned URL so browsers stream media straight from object storage."""
    if asset.status == "pending_upload":
        raise HTTPException(status_code=409, detail="Upload has not completed.")
    url = storage.presigned_get(asset.object_key, asset.content_type, asset.filename)
    return {"asset_id": asset.id, "url": url, "expires_in": config.PRESIGNED_URL_TTL_SECONDS}


@app.delete("/api/assets/{asset_id}", status_code=204)
def delete_asset(_: User = Depends(require_admin), asset: Asset = Depends(get_asset), db: Session = Depends(get_db)):
    """Removes the asset's vectors, stored object and catalog row."""
    if asset.task_id and asset.status in ("queued", "processing"):
        celery_client.control.revoke(asset.task_id)
    vectors.delete_asset(asset.id)
    storage.delete_object(asset.object_key)
    db.delete(asset)
    db.commit()


@app.post("/api/assets/{asset_id}/reindex", status_code=202)
def reindex_asset(_: User = Depends(require_admin), asset: Asset = Depends(get_asset), db: Session = Depends(get_db)):
    if asset.status not in ("ready", "failed"):
        raise HTTPException(status_code=409, detail=f"Asset is {asset.status}; wait for it to finish.")
    asset.status, asset.error, asset.summary = "queued", None, None
    asset.task_id = enqueue_ingestion(asset.id)
    db.commit()
    return {"asset_id": asset.id, "task_id": asset.task_id, "status": asset.status}


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str, _: User = Depends(require_user)):
    """Celery task state from the Redis result backend: PENDING, STARTED, PROGRESS, SUCCESS, FAILURE."""
    result = AsyncResult(task_id, app=celery_client)
    body = {"task_id": task_id, "state": result.state, "progress": 0, "stage": None, "result": None}
    if result.state == "PROGRESS" and isinstance(result.info, dict):
        body["progress"] = result.info.get("progress", 0)
        body["stage"] = result.info.get("stage")
    elif result.state == "SUCCESS":
        body["progress"], body["result"] = 100, result.result
    elif result.state == "FAILURE":
        body["result"] = str(result.info)
    return body


@app.post("/api/admin/reset")
def reset_storage(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Deletes every stored object, vector and catalog entry. Users are kept."""
    deleted = storage.delete_all()
    vectors.reset_collections()
    db.query(Asset).delete()
    db.commit()
    return {"status": "reset", "objects_deleted": deleted}


# ====================================================================== search & RAG
@app.get("/api/search")
async def search(
    query: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(5, ge=1, le=50),
    media_type: MediaType | None = None,
    asset_id: str | None = Query(None, max_length=36),
    _: User = Depends(require_user),
):
    """Text query against both indexes at once: MiniLM over text chunks, CLIP's text tower over
    image/frame vectors. Encoding and Qdrant calls run in the threadpool, so the event loop stays free."""
    q = clean_query(query)
    query_filter = vectors.search_filter(media_type, asset_id)

    async def text_matches():
        vector = (await run_in_threadpool(embeddings.embed_texts, [q]))[0]
        return await run_in_threadpool(vectors.search, vectors.TEXT, vector, limit, query_filter)

    async def visual_matches():
        if media_type in ("document", "audio"):
            return []
        vector = await run_in_threadpool(embeddings.embed_clip_text, q)
        return await run_in_threadpool(vectors.search, vectors.IMAGE, vector, limit, query_filter)

    text_hits, visual_hits = await asyncio.gather(text_matches(), visual_matches())
    return {
        "query": q,
        "text_matches": [format_hit(h) for h in text_hits],
        "visual_matches": [format_hit(h) for h in visual_hits],
    }


class RagRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=10)
    media_type: MediaType | None = None
    asset_id: str | None = Field(None, max_length=36)


@app.post("/api/rag")
async def rag(req: RagRequest, _: User = Depends(require_user)):
    """Retrieve-then-generate: top-k text chunks become numbered sources, the LLM answers with
    inline [n] citations, and each source carries the page/timestamp to jump to."""
    q = clean_query(req.question)
    vector = (await run_in_threadpool(embeddings.embed_texts, [q]))[0]
    hits = await run_in_threadpool(
        vectors.search, vectors.TEXT, vector, req.top_k, vectors.search_filter(req.media_type, req.asset_id), 0.2
    )
    if not hits:
        return {
            "question": q,
            "answer": "I couldn't find anything relevant in the library.",
            "sources": [],
            "cited": [],
        }

    sources = [{"n": i + 1, "label": source_label(h.payload or {}), **format_hit(h)} for i, h in enumerate(hits)]
    try:
        answer = await llm.chat(llm.RAG_SYSTEM, llm.build_rag_prompt(q, sources))
    except llm.LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"question": q, "answer": answer, "sources": sources, "cited": llm.cited_numbers(answer, len(sources))}


@app.post("/api/assets/{asset_id}/summary")
async def summarize_asset(
    refresh: bool = False,
    _: User = Depends(require_user),
    asset: Asset = Depends(get_asset),
    db: Session = Depends(get_db),
):
    """Summary of one asset. Images go to the vision model; everything else is summarized from its
    indexed text, in one prompt if it fits the context budget, otherwise map-reduce. Results are cached."""
    if asset.status != "ready":
        raise HTTPException(status_code=409, detail=f"Asset is {asset.status}; summaries need a ready asset.")
    if asset.summary and not refresh:
        return {"asset_id": asset.id, "filename": asset.filename, "cached": True, **asset.summary}

    try:
        if asset.media_type == "image":
            body = await run_in_threadpool(
                lambda: storage.s3.get_object(Bucket=storage.BUCKET, Key=asset.object_key)["Body"].read()
            )
            report = await llm.chat(
                "You are a careful visual analyst.",
                llm.IMAGE_REPORT_PROMPT,
                model=config.OLLAMA_VISION_MODEL,
                images=[base64.b64encode(body).decode()],
            )
            result = {"summary": report, "strategy": "vision", "llm_calls": 1, "source_chunks": 0}
        else:
            points = await run_in_threadpool(lambda: list(vectors.iter_asset_points(vectors.TEXT, asset.id)))
            if not points:
                raise HTTPException(
                    status_code=404, detail="This asset has no extracted text (no speech or text layer)."
                )
            points.sort(key=lambda p: (p.payload.get("page") or 0, p.payload.get("start") or 0.0))
            blocks = []
            for p in points:
                where = (
                    f"p. {p.payload['page']}" if p.payload.get("page") is not None else fmt_ts(p.payload.get("start"))
                )
                blocks.append(f"[{where}] {p.payload.get('text', '').strip()}")
            text_chars = sum(len(p.payload.get("text", "")) for p in points)
            if text_chars < 100:
                # Too little content for a meaningful summary: return it verbatim without an LLM call.
                result = {
                    "summary": "\n".join(blocks),
                    "strategy": "verbatim-short",
                    "llm_calls": 0,
                    "source_chunks": len(points),
                }
            else:
                result = await llm.summarize_blocks(asset.filename, blocks)
                result["source_chunks"] = len(points)
    except llm.LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    asset.summary = result
    await run_in_threadpool(db.commit)
    return {"asset_id": asset.id, "filename": asset.filename, "cached": False, **result}
