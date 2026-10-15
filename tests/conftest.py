"""Shared fixtures. Unit tests run against in-memory SQLite and faked S3/Qdrant/Celery/LLM
clients; nothing touches the network. Integration tests (tests/integration) use real services."""

import os

# Must be set before any `app.*` import. setdefault lets CI point integration tests at real services.
os.environ.setdefault("JWT_SECRET", "unit-test-secret-key-that-is-long-enough-0123456789")
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")
os.environ.setdefault("MINIO_ACCESS_KEY", "test-access")
os.environ.setdefault("MINIO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("MINIO_ENDPOINT", "http://localhost:9000")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("ADMIN_PASSWORD", "admin-pass-123")
os.environ.setdefault("CAPTION_IMAGES", "false")

from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402

from app.database import Asset, Base, SessionLocal, User, engine, hash_password  # noqa: E402

PDF_BYTES = b"%PDF-1.4\n%fake test document\n"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def scored(payload: dict, score: float = 0.9):
    return SimpleNamespace(payload=payload, score=score)


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if not db.get(User, "test_admin"):
            db.add_all(
                [
                    User(username="test_admin", hashed_password=hash_password("admin-pass-123"), role="admin"),
                    User(username="test_viewer", hashed_password=hash_password("viewer-pass-123"), role="viewer"),
                ]
            )
            db.commit()
    yield


@pytest.fixture(autouse=True)
def _clean_assets():
    yield
    with SessionLocal() as db:
        db.query(Asset).delete()
        db.commit()


@pytest.fixture
def fake_s3(monkeypatch):
    from app import storage

    s3 = MagicMock(name="s3")
    s3.generate_presigned_url.side_effect = lambda op, Params, ExpiresIn: (
        f"http://localhost:9000/{Params['Key']}?op={op}&part={Params.get('PartNumber', '')}"
    )
    s3.create_multipart_upload.return_value = {"UploadId": "upload-123"}

    def upload_fileobj(fileobj, bucket, key, ExtraArgs=None, Config=None):
        while fileobj.read(1024 * 1024):  # drain like boto3 does
            pass

    s3.upload_fileobj.side_effect = upload_fileobj
    monkeypatch.setattr(storage, "s3", s3)
    monkeypatch.setattr(storage, "s3_public", s3)
    return s3


@pytest.fixture
def fake_qdrant(monkeypatch):
    from app import vectors

    qdrant = MagicMock(name="qdrant")
    qdrant.search.return_value = []
    qdrant.scroll.return_value = ([], None)
    monkeypatch.setattr(vectors, "qdrant", qdrant)
    return qdrant


@pytest.fixture
def fake_models(monkeypatch):
    from app import embeddings

    calls = SimpleNamespace(texts=[], clip=[], images=0)

    def embed_texts(texts):
        calls.texts.extend(texts)
        return [[0.0] * 384 for _ in texts]

    def embed_clip_text(text):
        calls.clip.append(text)
        return [0.0] * 512

    def embed_images(images):
        calls.images += len(images)
        return [[0.0] * 512 for _ in images]

    monkeypatch.setattr(embeddings, "embed_texts", embed_texts)
    monkeypatch.setattr(embeddings, "embed_clip_text", embed_clip_text)
    monkeypatch.setattr(embeddings, "embed_images", embed_images)
    return calls


@pytest.fixture
def fake_celery(monkeypatch):
    from app import main

    send = MagicMock(name="send_task", return_value=SimpleNamespace(id="task-abc"))
    monkeypatch.setattr(main.celery_client, "send_task", send)
    return send


@pytest.fixture
def client(fake_s3, fake_qdrant, fake_models, fake_celery):
    from fastapi.testclient import TestClient

    from app.main import app

    # Not used as a context manager: the lifespan (DB/bucket/collection init) is skipped.
    return TestClient(app)


def _token(client, username: str, password: str) -> dict:
    res = client.post("/api/token", data={"username": username, "password": password})
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


@pytest.fixture
def admin_headers(client):
    return _token(client, "test_admin", "admin-pass-123")


@pytest.fixture
def viewer_headers(client):
    return _token(client, "test_viewer", "viewer-pass-123")


@pytest.fixture
def make_asset():
    def _make(**overrides) -> Asset:
        import uuid

        asset_id = overrides.pop("id", str(uuid.uuid4()))
        fields = {
            "filename": "report.pdf",
            "media_type": "document",
            "content_type": "application/pdf",
            "size_bytes": 1234,
            "status": "ready",
            "uploaded_by": "test_admin",
            **overrides,
        }
        fields.setdefault("object_key", f"{asset_id}/{fields['filename']}")
        with SessionLocal() as db:
            asset = Asset(id=asset_id, **fields)
            db.add(asset)
            db.commit()
            return asset

    return _make
