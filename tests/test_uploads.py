import io

import pytest

from app import main
from app.database import Asset, SessionLocal
from tests.conftest import PDF_BYTES, PNG_BYTES


def _upload(client, headers, name: str, data: bytes):
    return client.post("/api/upload", headers=headers, files={"file": (name, io.BytesIO(data))})


def test_upload_stores_object_and_queues_ingestion(client, admin_headers, fake_s3, fake_celery):
    res = _upload(client, admin_headers, "Quarterly Report.pdf", PDF_BYTES)
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["filename"] == "Quarterly_Report.pdf"
    assert body["task_id"] == "task-abc"

    _, bucket, key = fake_s3.upload_fileobj.call_args.args
    assert key == f"{body['asset_id']}/Quarterly_Report.pdf"
    fake_celery.assert_called_once_with("dam.ingest_asset", args=[body["asset_id"]], queue="ingestion")

    with SessionLocal() as db:
        asset = db.get(Asset, body["asset_id"])
        assert (asset.status, asset.media_type, asset.size_bytes) == ("queued", "document", len(PDF_BYTES))
        assert asset.uploaded_by == "test_admin"


def test_upload_filename_cannot_escape_its_prefix(client, admin_headers, fake_s3):
    res = _upload(client, admin_headers, "../../etc/passwd.pdf", PDF_BYTES)
    assert res.status_code == 202
    key = fake_s3.upload_fileobj.call_args.args[2]
    assert ".." not in key and key.endswith("/passwd.pdf")


@pytest.mark.parametrize(
    ("name", "data", "code"),
    [
        ("malware.exe", b"MZ\x90\x00", 415),
        ("no_extension", PDF_BYTES, 415),
        ("fake.png", PDF_BYTES, 415),  # extension says PNG, bytes say PDF
        ("fake.pdf", PNG_BYTES, 415),
    ],
)
def test_upload_rejects_disallowed_or_mismatched_files(client, admin_headers, fake_s3, name, data, code):
    assert _upload(client, admin_headers, name, data).status_code == code
    fake_s3.upload_fileobj.assert_not_called()


def test_upload_rejected_by_content_length_before_body_is_read(client, admin_headers, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    res = _upload(client, admin_headers, "big.pdf", PDF_BYTES + b"x" * 2 * 2**20)
    assert res.status_code == 413


def test_upload_rejected_while_streaming_when_over_limit(client, admin_headers, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 5 * 2**20)  # header check allows +1 MiB of overhead
    res = _upload(client, admin_headers, "big.pdf", PDF_BYTES + b"x" * (5 * 2**20 + 100))
    assert res.status_code == 413
    with SessionLocal() as db:
        assert db.query(Asset).count() == 0


# ---------------------------------------------------------------- direct (presigned) uploads
def test_direct_upload_init_returns_one_presigned_url_per_part(client, admin_headers):
    size = 40 * 2**20  # 40 MiB with 16 MiB parts -> 3 parts
    res = client.post("/api/uploads/init", headers=admin_headers, json={"filename": "video.mp4", "size_bytes": size})
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["part_size"] == 16 * 2**20
    assert [p["part_number"] for p in body["parts"]] == [1, 2, 3]
    with SessionLocal() as db:
        asset = db.get(Asset, body["asset_id"])
        assert asset.status == "pending_upload"
        assert asset.stats == {"upload_id": "upload-123"}


def test_direct_upload_rejects_oversized_and_unknown_types(client, admin_headers):
    too_big = (main.MAX_DIRECT_UPLOAD_BYTES or 0) + 1
    res = client.post("/api/uploads/init", headers=admin_headers, json={"filename": "v.mp4", "size_bytes": too_big})
    assert res.status_code == 413
    res = client.post("/api/uploads/init", headers=admin_headers, json={"filename": "v.iso", "size_bytes": 10})
    assert res.status_code == 415


def _init(client, headers, size=1000):
    return client.post("/api/uploads/init", headers=headers, json={"filename": "clip.mp4", "size_bytes": size}).json()


def test_direct_upload_complete_verifies_size_then_queues(client, admin_headers, fake_s3, fake_celery):
    plan = _init(client, admin_headers)
    fake_s3.head_object.return_value = {"ContentLength": 1000}
    fake_s3.get_object.return_value = {"Body": io.BytesIO(b"\x00\x00\x00\x18ftypmp42")}
    res = client.post(
        f"/api/uploads/{plan['asset_id']}/complete",
        headers=admin_headers,
        json={"upload_id": plan["upload_id"], "parts": [{"part_number": 1, "etag": '"abc"'}]},
    )
    assert res.status_code == 202, res.text
    assert res.json()["status"] == "queued"
    parts = fake_s3.complete_multipart_upload.call_args.kwargs["MultipartUpload"]["Parts"]
    assert parts == [{"PartNumber": 1, "ETag": '"abc"'}]
    fake_celery.assert_called_once()


def test_direct_upload_complete_rejects_size_mismatch(client, admin_headers, fake_s3, fake_celery):
    plan = _init(client, admin_headers, size=1000)
    fake_s3.head_object.return_value = {"ContentLength": 999}
    res = client.post(
        f"/api/uploads/{plan['asset_id']}/complete",
        headers=admin_headers,
        json={"upload_id": plan["upload_id"], "parts": [{"part_number": 1, "etag": "e"}]},
    )
    assert res.status_code == 400
    fake_s3.delete_object.assert_called_once()
    fake_celery.assert_not_called()
    with SessionLocal() as db:
        assert db.get(Asset, plan["asset_id"]) is None


def test_direct_upload_complete_requires_matching_upload_id(client, admin_headers):
    plan = _init(client, admin_headers)
    res = client.post(
        f"/api/uploads/{plan['asset_id']}/complete",
        headers=admin_headers,
        json={"upload_id": "someone-elses", "parts": [{"part_number": 1, "etag": "e"}]},
    )
    assert res.status_code == 409


def test_direct_upload_abort_removes_pending_asset(client, admin_headers, fake_s3):
    plan = _init(client, admin_headers)
    res = client.post(
        f"/api/uploads/{plan['asset_id']}/abort", headers=admin_headers, json={"upload_id": plan["upload_id"]}
    )
    assert res.status_code == 204
    fake_s3.abort_multipart_upload.assert_called_once()
    with SessionLocal() as db:
        assert db.get(Asset, plan["asset_id"]) is None


# ---------------------------------------------------------------- duplicate detection
def test_identical_content_is_rejected_as_duplicate(client, admin_headers, fake_s3, fake_celery):
    first = _upload(client, admin_headers, "report.pdf", PDF_BYTES)
    assert first.status_code == 202
    fake_s3.upload_fileobj.reset_mock()

    dup = _upload(client, admin_headers, "renamed copy.pdf", PDF_BYTES)  # same bytes, different name
    assert dup.status_code == 409
    assert first.json()["asset_id"] in dup.json()["detail"]
    fake_s3.upload_fileobj.assert_not_called()  # rejected before anything is stored

    forced = client.post(
        "/api/upload",
        headers=admin_headers,
        params={"allow_duplicate": "true"},
        files={"file": ("copy.pdf", io.BytesIO(PDF_BYTES))},
    )
    assert forced.status_code == 202


def test_upload_records_sha256(client, admin_headers):
    import hashlib

    asset_id = _upload(client, admin_headers, "report.pdf", PDF_BYTES).json()["asset_id"]
    with SessionLocal() as db:
        assert db.get(Asset, asset_id).sha256 == hashlib.sha256(PDF_BYTES).hexdigest()


def test_failed_assets_do_not_block_reupload(client, admin_headers, make_asset):
    import hashlib

    make_asset(status="failed", sha256=hashlib.sha256(PDF_BYTES).hexdigest())
    assert _upload(client, admin_headers, "report.pdf", PDF_BYTES).status_code == 202


def test_direct_upload_duplicate_rejected_before_any_bytes(client, admin_headers, make_asset, fake_s3):
    make_asset(sha256="a" * 64)
    body = {"filename": "video.mp4", "size_bytes": 10, "sha256": "a" * 64}
    assert client.post("/api/uploads/init", headers=admin_headers, json=body).status_code == 409
    fake_s3.create_multipart_upload.assert_not_called()
    body["allow_duplicate"] = True
    assert client.post("/api/uploads/init", headers=admin_headers, json=body).status_code == 201


def test_direct_upload_rejects_malformed_hash(client, admin_headers):
    body = {"filename": "video.mp4", "size_bytes": 10, "sha256": "not-a-hash"}
    assert client.post("/api/uploads/init", headers=admin_headers, json=body).status_code == 422
