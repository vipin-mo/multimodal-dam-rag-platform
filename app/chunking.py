"""Text chunking for documents and transcript windowing for audio/video.

Chunks are packed from whole sentences up to `size` characters, and consecutive chunks
share up to `overlap` characters of trailing sentences so facts that straddle a boundary
remain retrievable.
"""

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across line breaks
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s]


def _hard_split(sentence: str, size: int, overlap: int) -> list[str]:
    """Word-boundary split for sentences longer than one chunk."""
    words, pieces, current = sentence.split(" "), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            current = tail[tail.find(" ") + 1 :] if " " in tail else ""
        current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = normalize_whitespace(text)
    if not text:
        return []

    sentences: list[str] = []
    for sentence in split_sentences(text):
        sentences.extend(_hard_split(sentence, size, overlap) if len(sentence) > size else [sentence])

    chunks: list[str] = []
    window: list[str] = []
    for sentence in sentences:
        if window and len(" ".join(window + [sentence])) > size:
            chunks.append(" ".join(window))
            carried: list[str] = []
            for previous in reversed(window):  # keep trailing sentences as overlap
                if len(" ".join([previous] + carried)) > overlap:
                    break
                carried.insert(0, previous)
            window = carried
            # Never let the carried overlap push the next sentence past the size limit.
            while window and len(" ".join(window + [sentence])) > size:
                window.pop(0)
        window.append(sentence)
    if window:
        chunks.append(" ".join(window))
    return chunks


@dataclass
class TranscriptWindow:
    text: str
    start: float
    end: float


def window_segments(segments, max_chars: int = 500) -> list[TranscriptWindow]:
    """Merges consecutive Whisper segments (objects with .text/.start/.end) into windows of
    at most `max_chars`. The last segment of each window is repeated at the start of the next."""
    windows: list[TranscriptWindow] = []
    current: list = []

    def flush() -> None:
        windows.append(
            TranscriptWindow(
                text=" ".join(s.text.strip() for s in current),
                start=float(current[0].start),
                end=float(current[-1].end),
            )
        )

    for segment in segments:
        if not segment.text.strip():
            continue
        length = sum(len(s.text.strip()) + 1 for s in current) + len(segment.text.strip())
        if current and length > max_chars:
            flush()
            current = current[-1:] if len(current) > 1 else []
        current.append(segment)
    if current:
        flush()
    return windows
