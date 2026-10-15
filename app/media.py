"""File-type allowlist, filename sanitization and content sniffing."""

import os
import re
import unicodedata

# extension -> (media type, content type served to browsers)
ALLOWED_TYPES: dict[str, tuple[str, str]] = {
    ".pdf": ("document", "application/pdf"),
    ".png": ("image", "image/png"),
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".mp4": ("video", "video/mp4"),
    ".mov": ("video", "video/quicktime"),
    ".mkv": ("video", "video/x-matroska"),
    ".avi": ("video", "video/x-msvideo"),
    ".mp3": ("audio", "audio/mpeg"),
    ".wav": ("audio", "audio/wav"),
}

MEDIA_TYPES = ("document", "image", "video", "audio")


class UnsupportedMediaError(ValueError):
    pass


def extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def classify(filename: str) -> tuple[str, str]:
    """Returns (media_type, content_type) or raises UnsupportedMediaError."""
    ext = extension(filename)
    if ext not in ALLOWED_TYPES:
        allowed = ", ".join(sorted(ALLOWED_TYPES))
        raise UnsupportedMediaError(f"Unsupported file type '{ext or '(none)'}'. Allowed: {allowed}")
    return ALLOWED_TYPES[ext]


def sanitize_filename(filename: str, max_length: int = 120) -> str:
    """Reduces a client-supplied name to a safe basename: no directories, no control or
    shell-special characters, bounded length, extension preserved."""
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    name = re.sub(r"[\s_]+", "_", name)
    stem, ext = os.path.splitext(name)
    stem = stem.strip("_-")[: max_length - len(ext)] or "file"
    return f"{stem}{ext.lower()}"


def object_key(asset_id: str, safe_name: str) -> str:
    """Storage keys are namespaced by asset id, so identical names never collide."""
    return f"{asset_id}/{safe_name}"


def sniff_ok(ext: str, head: bytes) -> bool:
    """Checks the leading bytes against the claimed extension."""
    if ext == ".pdf":
        return head.startswith(b"%PDF-")
    if ext == ".png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in (".jpg", ".jpeg"):
        return head.startswith(b"\xff\xd8\xff")
    if ext in (".mp4", ".mov"):
        return head[4:8] == b"ftyp" or head[4:8] in (b"moov", b"mdat", b"wide", b"free")
    if ext == ".mkv":
        return head.startswith(b"\x1a\x45\xdf\xa3")
    if ext == ".avi":
        return head.startswith(b"RIFF") and head[8:12] == b"AVI "
    if ext == ".wav":
        return head.startswith(b"RIFF") and head[8:12] == b"WAVE"
    if ext == ".mp3":
        return head.startswith(b"ID3") or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)
    return False
