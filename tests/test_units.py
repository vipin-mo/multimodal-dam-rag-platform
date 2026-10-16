"""Pure-function tests: chunking, filename/media handling, config, LLM retries and map-reduce."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app import config, llm, media
from app.chunking import chunk_text, window_segments


# ---------------------------------------------------------------- chunking
def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} has a few words in it." for i in range(60))
    chunks = chunk_text(text, size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    for left, right in zip(chunks, chunks[1:], strict=False):
        last_sentence = left.split(". ")[-1]
        assert right.startswith(last_sentence.rstrip(".")), "trailing sentence carried into next chunk"


def test_chunking_rejoins_hyphenated_line_breaks_and_whitespace():
    assert chunk_text("A data-\nbase   migration\nis planned.", 500, 100) == ["A database migration is planned."]


def test_overlong_sentence_is_split_on_word_boundaries():
    chunks = chunk_text(" ".join(["word"] * 300), size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert all(not c.startswith(" ") and "wor d" not in c for c in chunks)


def test_empty_text_and_bad_overlap():
    assert chunk_text("   \n ", 500, 100) == []
    with pytest.raises(ValueError):
        chunk_text("x", 100, 100)


def test_transcript_windows_keep_timestamps_and_overlap():
    segments = [SimpleNamespace(text=f" segment {i} words here.", start=i * 2.0, end=i * 2.0 + 2) for i in range(20)]
    windows = window_segments(segments, max_chars=100)
    assert windows[0].start == 0.0
    assert all(len(w.text) <= 100 for w in windows)
    # each window starts with the previous window's final segment
    assert windows[1].start == windows[0].end - 2
    assert windows[-1].end == 40.0


# ---------------------------------------------------------------- media
@pytest.mark.parametrize(
    ("raw", "safe"),
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\Report Q3.PDF", "Report_Q3.pdf"),
        ("résumé (final).pdf", "resume_final.pdf"),
        ("$(rm -rf ~).mp4", "rm_-rf.mp4"),
        ("...", "file"),
        ("a" * 300 + ".png", "a" * 116 + ".png"),
    ],
)
def test_sanitize_filename(raw, safe):
    assert media.sanitize_filename(raw) == safe


@pytest.mark.parametrize(
    ("ext", "head", "ok"),
    [
        (".pdf", b"%PDF-1.7", True),
        (".png", b"\x89PNG\r\n\x1a\n....", True),
        (".jpg", b"\xff\xd8\xff\xe0", True),
        (".mp4", b"\x00\x00\x00\x18ftypisom", True),
        (".mkv", b"\x1a\x45\xdf\xa3", True),
        (".wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", True),
        (".mp3", b"ID3\x04", True),
        (".mp3", b"\xff\xfb\x90", True),
        (".pdf", b"MZ\x90\x00", False),
        (".wav", b"RIFF\x00\x00\x00\x00AVI ", False),
    ],
)
def test_content_sniffing(ext, head, ok):
    assert media.sniff_ok(ext, head) is ok


def test_classify_unknown_extension():
    with pytest.raises(media.UnsupportedMediaError):
        media.classify("archive.zip")
    assert media.classify("CLIP.MOV") == ("video", "video/quicktime")


# ---------------------------------------------------------------- config
def test_jwt_secret_is_required_and_long(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(config.ConfigError, match="JWT_SECRET"):
        config.jwt_secret()
    monkeypatch.setenv("JWT_SECRET", "short")
    with pytest.raises(config.ConfigError, match="32"):
        config.jwt_secret()


def test_admin_password_required_to_seed_first_admin(monkeypatch):
    from sqlalchemy.orm import sessionmaker

    from app import database

    engine = database.make_engine("sqlite://")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    with pytest.raises(config.ConfigError, match="ADMIN_PASSWORD"):
        database.init_db()
    monkeypatch.setenv("ADMIN_PASSWORD", "long-enough-pass")
    monkeypatch.setenv("VIEWER_PASSWORD", "viewer-pass-1")
    database.init_db()
    with database.SessionLocal() as db:
        assert {u.role for u in db.query(database.User)} == {"admin", "viewer"}


# ---------------------------------------------------------------- LLM client
class FakeAsyncClient:
    """Stands in for httpx.AsyncClient; replays a scripted list of responses/exceptions."""

    script: list = []
    calls = 0

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json):
        FakeAsyncClient.calls += 1
        item = FakeAsyncClient.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def scripted_llm(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(llm.asyncio, "sleep", no_sleep)
    FakeAsyncClient.calls = 0

    def run(*script):
        FakeAsyncClient.script = list(script)
        return asyncio.run(llm.chat("sys", "user"))

    return run


def _ok(text):
    return httpx.Response(200, json={"message": {"content": text}})


def test_llm_retries_transient_failures(scripted_llm):
    assert scripted_llm(httpx.ConnectError("refused"), httpx.Response(503), _ok(" fine ")) == "fine"
    assert FakeAsyncClient.calls == 3


def test_llm_gives_up_after_max_attempts(scripted_llm):
    with pytest.raises(llm.LLMUnavailable, match="4 attempts"):
        scripted_llm(*[httpx.ConnectError("refused")] * 4)
    assert FakeAsyncClient.calls == llm.MAX_ATTEMPTS


def test_llm_read_timeout_is_not_retried(scripted_llm):
    """Ollama keeps generating for an abandoned request; retrying would only queue more work."""
    with pytest.raises(llm.LLMUnavailable, match="did not finish"):
        scripted_llm(httpx.ReadTimeout("slow"), _ok("never reached"))
    assert FakeAsyncClient.calls == 1


def test_llm_missing_model_fails_fast(scripted_llm):
    with pytest.raises(llm.LLMUnavailable, match="not available"):
        scripted_llm(httpx.Response(404, json={"error": "model not found"}))
    assert FakeAsyncClient.calls == 1


def test_llm_payload_sets_context_window():
    payload = llm._payload("m", "s", "u", images=["b64"], max_tokens=10)
    assert payload["options"]["num_ctx"] == config.OLLAMA_NUM_CTX
    assert payload["options"]["num_predict"] == 10
    assert llm._payload("m", "s", "u", None, None)["options"]["num_predict"] == llm.DEFAULT_MAX_TOKENS
    assert payload["messages"][1]["images"] == ["b64"]


def test_vision_payload_has_no_system_turn():
    assert [m["role"] for m in llm._payload("m", None, "u", ["img"], None)["messages"]] == ["user"]


@pytest.mark.parametrize(
    ("raw", "trimmed"),
    [
        ("A red square. It sits on a white", "A red square."),
        ("Complete sentence.", "Complete sentence."),
        ("no punctuation at all", "no punctuation at all"),
        ("  Spaced\n out.  Next one cut", "Spaced out."),
    ],
)
def test_trim_to_sentence(raw, trimmed):
    assert llm.trim_to_sentence(raw) == trimmed


@pytest.mark.parametrize(
    ("raw", "answer", "cited"),
    [
        ("Due December 15 [1].", "Due December 15 [1].", [1]),
        ("[1]", llm.NO_ANSWER, []),
        ("  [2] .", llm.NO_ANSWER, []),
        ("I don't know.", llm.NO_ANSWER, []),
        ("Plain answer without citations.", "Plain answer without citations.", []),
    ],
)
def test_finalize_answer(raw, answer, cited):
    assert llm.finalize_answer(raw, n_sources=3) == (answer, cited)


def test_cited_numbers():
    assert llm.cited_numbers("A [2] and [1][3], not [9] or [x].", n_sources=3) == [1, 2, 3]


# ---------------------------------------------------------------- map-reduce summaries
def test_pack_blocks_respects_budget():
    windows = llm.pack_blocks(["a" * 40] * 10, budget=100)
    assert all(sum(len(b) + 1 for b in w) <= 100 for w in windows)
    assert sum(len(w) for w in windows) == 10


def test_summary_single_pass_when_content_fits(monkeypatch):
    prompts = []

    async def fake_chat(system, user, **_):
        prompts.append(user)
        return "summary"

    monkeypatch.setattr(llm, "chat", fake_chat)
    result = asyncio.run(llm.summarize_blocks("doc.pdf", ["[p. 1] short text"], budget=1000))
    assert result == {"summary": "summary", "strategy": "single-pass", "reduce_levels": 0, "llm_calls": 1}


def test_summary_map_reduce_when_content_exceeds_budget(monkeypatch):
    systems = []

    async def fake_chat(system, user, **_):
        systems.append(system)
        return "- partial point"

    monkeypatch.setattr(llm, "chat", fake_chat)
    blocks = [f"[p. {i}] " + "x" * 90 for i in range(30)]  # ~3000 chars
    result = asyncio.run(llm.summarize_blocks("big.pdf", blocks, budget=1000))
    assert result["strategy"] == "map-reduce" and result["reduce_levels"] == 1
    partial_calls = systems.count(llm.PARTIAL_SYSTEM)
    assert partial_calls == len(llm.pack_blocks(blocks, 1000)) >= 3
    assert systems[-1] == llm.SUMMARY_SYSTEM
    assert result["llm_calls"] == partial_calls + 1
