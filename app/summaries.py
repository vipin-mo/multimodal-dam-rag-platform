"""Asset summaries, run by the LLM worker (Celery queue `llm`).

Images go to the vision model. Everything else is summarized from its indexed text: one prompt
when it fits the context budget, otherwise map-reduce. On CPU a 100-page PDF takes minutes,
which is why this runs as a background task rather than inside an HTTP request.
"""

import base64
from collections.abc import Callable

from app import config, llm, storage, vectors

MIN_TEXT_CHARS = 100  # below this an LLM summary adds nothing; the text is returned verbatim


class NothingToSummarize(ValueError):
    pass


def fmt_ts(seconds: float | None) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def located_blocks(asset_id: str) -> tuple[list[str], int]:
    """The asset's text chunks in reading order, each prefixed with its page or timestamp,
    plus the total length of the underlying text."""
    points = list(vectors.iter_asset_points(vectors.TEXT, asset_id))
    points.sort(key=lambda p: (p.payload.get("page") or 0, p.payload.get("start") or 0.0))
    blocks, text_chars = [], 0
    for p in points:
        text = (p.payload.get("text") or "").strip()
        where = f"p. {p.payload['page']}" if p.payload.get("page") is not None else fmt_ts(p.payload.get("start"))
        blocks.append(f"[{where}] {text}")
        text_chars += len(text)
    return blocks, text_chars


async def summarize(
    asset_id: str,
    filename: str,
    media_type: str,
    object_key: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    if media_type == "image":
        body = storage.s3.get_object(Bucket=storage.BUCKET, Key=object_key)["Body"].read()
        report = await llm.chat(
            None,
            llm.IMAGE_REPORT_PROMPT,
            model=config.OLLAMA_VISION_MODEL,
            images=[base64.b64encode(body).decode()],
            max_tokens=400,
        )
        return {"summary": report, "strategy": "vision", "llm_calls": 1, "source_chunks": 0}

    blocks, text_chars = located_blocks(asset_id)
    if not blocks:
        raise NothingToSummarize("This asset has no extracted text (no speech or text layer).")
    if text_chars < MIN_TEXT_CHARS:
        return {
            "summary": "\n".join(blocks),
            "strategy": "verbatim-short",
            "llm_calls": 0,
            "source_chunks": len(blocks),
        }
    result = await llm.summarize_blocks(filename, blocks, on_progress=on_progress)
    return {**result, "source_chunks": len(blocks)}
