"""Upload large files straight to object storage with presigned multipart URLs.

The file bytes never pass through the API gateway or the Streamlit UI: the gateway only
signs one URL per part, and this client streams one part at a time from disk, so memory
use is bounded by the part size (16 MiB by default) regardless of file size.

Usage:
    DAM_PASSWORD=... python scripts/dam_upload.py path/to/video.mp4 [--api http://localhost:8000]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx


def _put_part(url: str, data: bytes, attempts: int = 3) -> str:
    for attempt in range(1, attempts + 1):
        try:
            res = httpx.put(url, content=data, timeout=300)
            res.raise_for_status()
            return res.headers["ETag"]
        except httpx.HTTPError:
            if attempt == attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def direct_upload(client: httpx.Client, headers: dict, path: Path, progress: bool = False) -> dict:
    """Runs init -> PUT parts -> complete. Returns the gateway's completion response."""
    size = path.stat().st_size
    init = client.post("/api/uploads/init", headers=headers, json={"filename": path.name, "size_bytes": size})
    init.raise_for_status()
    plan = init.json()
    asset_id, upload_id, part_size = plan["asset_id"], plan["upload_id"], plan["part_size"]

    completed = []
    try:
        with path.open("rb") as fh:
            for part in plan["parts"]:
                chunk = fh.read(part_size)
                etag = _put_part(part["url"], chunk)
                completed.append({"part_number": part["part_number"], "etag": etag})
                if progress:
                    done = min(size, part["part_number"] * part_size)
                    print(f"\r  {done / 2**20:8.1f} / {size / 2**20:.1f} MiB", end="", flush=True)
        if progress:
            print()
    except BaseException:
        client.post(f"/api/uploads/{asset_id}/abort", headers=headers, json={"upload_id": upload_id})
        raise

    done = client.post(
        f"/api/uploads/{asset_id}/complete",
        headers=headers,
        json={"upload_id": upload_id, "parts": completed},
    )
    done.raise_for_status()
    return done.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("--api", default=os.getenv("DAM_API_URL", "http://localhost:8000"))
    parser.add_argument("--user", default=os.getenv("DAM_USER", "admin"))
    parser.add_argument("--password", default=os.getenv("DAM_PASSWORD"))
    args = parser.parse_args()
    if not args.password:
        sys.exit("Set DAM_PASSWORD or pass --password.")

    with httpx.Client(base_url=args.api, timeout=60) as client:
        token = client.post("/api/token", data={"username": args.user, "password": args.password})
        token.raise_for_status()
        headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
        t0 = time.perf_counter()
        result = direct_upload(client, headers, args.file, progress=True)
        elapsed = time.perf_counter() - t0
    mib = args.file.stat().st_size / 2**20
    print(f"Uploaded {mib:.1f} MiB in {elapsed:.1f}s ({mib / elapsed:.1f} MiB/s)")
    print(f"asset_id={result['asset_id']} task_id={result['task_id']}")


if __name__ == "__main__":
    main()
