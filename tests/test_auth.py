from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.database import SessionLocal, User, hash_password


def test_login_returns_bearer_token_with_role(client):
    res = client.post("/api/token", data={"username": "test_admin", "password": "admin-pass-123"})
    assert res.status_code == 200
    body = res.json()
    assert body["token_type"] == "bearer"
    assert body["role"] == "admin"
    claims = jwt.decode(body["access_token"], options={"verify_signature": False})
    assert claims["sub"] == "test_admin"
    assert {"exp", "iat"} <= claims.keys()


@pytest.mark.parametrize(
    ("username", "password"),
    [("test_admin", "wrong-password"), ("no_such_user", "whatever-123")],
)
def test_login_failures_are_indistinguishable(client, username, password):
    res = client.post("/api/token", data={"username": username, "password": password})
    assert res.status_code == 401
    assert res.json()["detail"] == "Incorrect username or password."


def test_me_reports_identity(client, viewer_headers):
    assert client.get("/api/me", headers=viewer_headers).json() == {"username": "test_viewer", "role": "viewer"}


def test_missing_token_is_rejected(client):
    assert client.get("/api/assets").status_code == 401


def test_token_in_query_string_is_not_accepted(client, admin_headers):
    """Tokens must travel in the Authorization header; URLs end up in logs and browser history."""
    token = admin_headers["Authorization"].split()[1]
    assert client.get("/api/assets", params={"token": token}).status_code == 401


def _forge(secret: str, **claims) -> dict:
    base = {"sub": "test_admin", "exp": datetime.now(UTC) + timedelta(minutes=5)}
    return {"Authorization": f"Bearer {jwt.encode(base | claims, secret, algorithm='HS256')}"}


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer not-a-jwt"},
        _forge("some-other-secret-that-is-also-long-enough-123"),
        _forge(
            "unit-test-secret-key-that-is-long-enough-0123456789",
            exp=datetime.now(UTC) - timedelta(minutes=1),
        ),
        _forge("unit-test-secret-key-that-is-long-enough-0123456789", sub="ghost_user"),
    ],
    ids=["malformed", "wrong-signature", "expired", "unknown-user"],
)
def test_invalid_tokens_are_rejected(client, headers):
    res = client.get("/api/assets", headers=headers)
    assert res.status_code == 401
    assert res.headers["www-authenticate"] == "Bearer"


def test_alg_none_token_is_rejected(client):
    token = jwt.encode({"sub": "test_admin", "exp": 9999999999}, key=None, algorithm="none")
    assert client.get("/api/assets", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_role_is_read_from_database_not_token(client):
    """A token minted while the user was admin stops working for admin routes after demotion."""
    with SessionLocal() as db:
        db.add(User(username="temp_admin", hashed_password=hash_password("temp-pass-123"), role="admin"))
        db.commit()
    res = client.post("/api/token", data={"username": "temp_admin", "password": "temp-pass-123"})
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    with SessionLocal() as db:
        db.get(User, "temp_admin").role = "viewer"
        db.commit()
    assert client.post("/api/admin/reset", headers=headers).status_code == 403
    with SessionLocal() as db:
        db.delete(db.get(User, "temp_admin"))
        db.commit()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/admin/reset"),
        ("post", "/api/upload"),
        ("post", "/api/uploads/init"),
        ("delete", "/api/assets/00000000-0000-0000-0000-000000000000"),
        ("post", "/api/assets/00000000-0000-0000-0000-000000000000/reindex"),
    ],
)
def test_viewer_cannot_use_admin_routes(client, viewer_headers, method, path):
    assert getattr(client, method)(path, headers=viewer_headers).status_code == 403


def test_admin_can_reset(client, admin_headers, fake_s3, fake_qdrant):
    fake_s3.get_paginator.return_value.paginate.return_value = [{"Contents": [{"Key": "a/x.pdf"}]}]
    fake_s3.list_multipart_uploads.return_value = {}
    fake_qdrant.get_collections.return_value.collections = []
    res = client.post("/api/admin/reset", headers=admin_headers)
    assert res.status_code == 200
    assert res.json() == {"status": "reset", "objects_deleted": 1}
    assert fake_qdrant.delete_collection.call_count == 2


def test_cors_allows_only_configured_origins(client):
    allowed = client.options(
        "/api/search", headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "GET"}
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"
    denied = client.options(
        "/api/search", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
    )
    assert "access-control-allow-origin" not in denied.headers
