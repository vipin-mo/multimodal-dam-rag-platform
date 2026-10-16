"""Benchmarks for the DAM stack: ingestion time, search latency, retrieval recall, upload memory.

Works against the current API (`--api v2`, default) and the original API (`--api legacy`)
so before/after numbers come from the same harness.

Examples:
    python scripts/benchmark.py ingest --dir samples/corpus --reset
    python scripts/benchmark.py search -n 200 --concurrency 1 8
    python scripts/benchmark.py eval -k 5
    python scripts/benchmark.py upload --file samples/large/large_upload_test.mp4 --watch dam-api-gateway
    python scripts/benchmark.py upload --direct --file samples/large/large_upload_test.mp4 --watch dam-api-gateway

Credentials come from DAM_USER / DAM_PASSWORD (or --user / --password).
Results are printed and appended as JSON lines to --results (default benchmarks/results.jsonl).
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dam_upload import direct_upload  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Api:
    def __init__(self, base: str, flavor: str, user: str, password: str):
        self.base = base.rstrip("/")
        self.flavor = flavor
        self.client = httpx.Client(base_url=self.base, timeout=600)
        self.token = self._login(user, password)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def _login(self, user: str, password: str) -> str:
        if self.flavor == "legacy":
            res = self.client.post("/api/token", json={"username": user, "password": password})
        else:
            res = self.client.post("/api/token", data={"username": user, "password": password})
        res.raise_for_status()
        return res.json()["access_token"]

    def reset(self) -> None:
        path = "/api/init-db" if self.flavor == "legacy" else "/api/admin/reset"
        self.client.post(path, headers=self.headers).raise_for_status()

    def upload(self, path: Path) -> str:
        with path.open("rb") as fh:
            params = {} if self.flavor == "legacy" else {"allow_duplicate": "true"}
            res = self.client.post("/api/upload", headers=self.headers, params=params, files={"file": (path.name, fh)})
        res.raise_for_status()
        return res.json()["task_id"]

    def wait(self, task_id: str, poll: float = 0.05) -> dict:
        while True:
            res = self.client.get(f"/api/tasks/{task_id}", headers=self.headers)
            res.raise_for_status()
            body = res.json()
            if body["state"] in ("SUCCESS", "FAILURE"):
                return body
            time.sleep(poll)

    def search(self, query: str, limit: int, client: httpx.Client | None = None) -> dict:
        res = (client or self.client).get("/api/search", params={"query": query, "limit": limit}, headers=self.headers)
        res.raise_for_status()
        return res.json()


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def normalize(text: str) -> str:
    """Lowercase, hyphens as spaces, drop thousands separators and other punctuation."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower()).replace("-", " ")
    return " ".join(re.sub(r"[^a-z0-9.: ]+", " ", text).split())


def record(args, kind: str, payload: dict) -> None:
    payload = {"kind": kind, "api": args.api, "at": datetime.now(UTC).isoformat(), "label": args.label, **payload}
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------- ingestion
def cmd_ingest(api: Api, args) -> None:
    if args.reset:
        api.reset()
    rows = []
    for path in sorted(Path(args.dir).iterdir()):
        if path.name.startswith("."):
            continue
        t0 = time.perf_counter()
        task_id = api.upload(path)
        t_uploaded = time.perf_counter()
        status = api.wait(task_id)
        t_done = time.perf_counter()
        rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "state": status["state"],
                "upload_s": round(t_uploaded - t0, 3),
                "processing_s": round(t_done - t_uploaded, 3),
                "result": status.get("result"),
            }
        )
        print(
            f"{path.name:<28} {status['state']:<8} upload {t_uploaded - t0:6.2f}s  "
            f"processing {t_done - t_uploaded:7.2f}s"
        )
    record(args, "ingest", {"files": rows, "total_processing_s": round(sum(r["processing_s"] for r in rows), 2)})


# ---------------------------------------------------------------- latency
def cmd_search(api: Api, args) -> None:
    questions = json.loads(Path(args.questions).read_text())
    queries = [q["q"] for q in questions["text"]] + [q["q"] for q in questions["visual"]]
    for _ in range(3):  # warm up
        api.search(queries[0], 5)
    results = {}
    for conc in args.concurrency:
        latencies: list[float] = []
        lock = threading.Lock()

        def worker(i: int, latencies=latencies, lock=lock) -> None:
            with httpx.Client(base_url=api.base, timeout=120) as c:
                t0 = time.perf_counter()
                api.search(queries[i % len(queries)], 5, client=c)
                with lock:
                    latencies.append((time.perf_counter() - t0) * 1000)

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            list(pool.map(worker, range(args.n)))
        wall = time.perf_counter() - t_start
        results[f"c{conc}"] = {
            "n": args.n,
            "p50_ms": round(percentile(latencies, 50), 1),
            "p95_ms": round(percentile(latencies, 95), 1),
            "p99_ms": round(percentile(latencies, 99), 1),
            "mean_ms": round(statistics.mean(latencies), 1),
            "throughput_rps": round(args.n / wall, 1),
        }
        print(f"concurrency={conc}: {results[f'c{conc}']}")
    record(args, "search_latency", results)


# ---------------------------------------------------------------- recall
def cmd_eval(api: Api, args) -> None:
    questions = json.loads(Path(args.questions).read_text())
    k = args.k
    text_hits, rr, misses = 0, [], []
    for item in questions["text"]:
        matches = api.search(item["q"], k)["text_matches"][:k]
        rank = next(
            (
                i + 1
                for i, m in enumerate(matches)
                if m["filename"] == item["file"]
                and any(normalize(a) in normalize(m.get("text") or "") for a in item["answer"])
            ),
            None,
        )
        text_hits += rank is not None
        rr.append(1 / rank if rank else 0.0)
        if rank is None:
            misses.append({"q": item["q"], "top": [(m["filename"], m.get("text", "")[:80]) for m in matches[:3]]})

    visual_hits = 0
    for item in questions["visual"]:
        matches = api.search(item["q"], k)["visual_matches"][:k]
        visual_hits += any(m["filename"] == item["file"] for m in matches)

    n_text, n_vis = len(questions["text"]), len(questions["visual"])
    record(
        args,
        "retrieval_eval",
        {
            "k": k,
            "text_recall_at_k": round(text_hits / n_text, 3),
            "text_hits": f"{text_hits}/{n_text}",
            "text_mrr": round(sum(rr) / n_text, 3),
            "visual_hit_at_k": f"{visual_hits}/{n_vis}",
            "misses": misses,
        },
    )


# ---------------------------------------------------------------- upload memory
class MemorySampler(threading.Thread):
    """Polls `docker stats` and keeps the peak memory per container (MiB)."""

    UNITS = {"B": 1 / 2**20, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "kB": 1 / 1024, "MB": 1, "GB": 1024}

    def __init__(self, containers: list[str]):
        super().__init__(daemon=True)
        self.containers = containers
        self.peak = {c: 0.0 for c in containers}
        self.baseline: dict[str, float] = {}
        self.stop = threading.Event()

    def sample(self) -> dict[str, float]:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}", *self.containers],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        usage = {}
        for line in out.strip().splitlines():
            name, used = line.split()[0], line.split()[1]
            num, unit = re.match(r"([\d.]+)(\w+)", used).groups()
            usage[name] = float(num) * self.UNITS[unit]
        return usage

    def run(self) -> None:
        self.baseline = self.sample()
        while not self.stop.is_set():
            for name, mib in self.sample().items():
                self.peak[name] = max(self.peak[name], mib)


def cmd_upload(api: Api, args) -> None:
    path = Path(args.file)
    sampler = MemorySampler(args.watch) if args.watch else None
    if sampler:
        sampler.start()
        time.sleep(3)
    t0 = time.perf_counter()
    if args.direct:
        asset = direct_upload(api.client, api.headers, path, allow_duplicate=True)
        task_id = asset["task_id"]
    else:
        task_id = api.upload(path)
    t_uploaded = time.perf_counter()
    status = api.wait(task_id, poll=1.0) if args.wait else {"state": "NOT_WAITED"}
    t_done = time.perf_counter()
    payload = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "mode": "direct-presigned" if args.direct else "api-multipart",
        "upload_s": round(t_uploaded - t0, 2),
        "upload_MBps": round(path.stat().st_size / 2**20 / (t_uploaded - t0), 1),
        "state": status["state"],
        "processing_s": round(t_done - t_uploaded, 2) if args.wait else None,
    }
    if sampler:
        sampler.stop.set()
        sampler.join()
        payload["memory_mib"] = {
            c: {"baseline": round(sampler.baseline.get(c, 0), 1), "peak": round(sampler.peak[c], 1)} for c in args.watch
        }
    record(args, "upload", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("DAM_API_URL", "http://localhost:8000"))
    parser.add_argument("--api", choices=["v2", "legacy"], default="v2")
    parser.add_argument("--user", default=os.getenv("DAM_USER", "admin"))
    parser.add_argument("--password", default=os.getenv("DAM_PASSWORD"))
    parser.add_argument("--label", default="", help="free-form tag stored with results")
    parser.add_argument("--results", default=str(ROOT / "benchmarks" / "results.jsonl"))
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("--dir", default=str(ROOT / "samples" / "corpus"))
    p.add_argument("--reset", action="store_true", help="wipe storage first (admin)")

    p = sub.add_parser("search")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])

    p = sub.add_parser("eval")
    p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("upload")
    p.add_argument("--file", required=True)
    p.add_argument("--direct", action="store_true", help="presigned multipart upload straight to MinIO")
    p.add_argument("--watch", nargs="*", default=[], help="container names to sample with docker stats")
    p.add_argument("--wait", action="store_true", help="also wait for ingestion to finish")

    args = parser.parse_args()
    if not args.password:
        raise SystemExit("Set DAM_PASSWORD or pass --password.")
    api = Api(args.base_url, args.api, args.user, args.password)
    {"ingest": cmd_ingest, "search": cmd_search, "eval": cmd_eval, "upload": cmd_upload}[args.cmd](api, args)


if __name__ == "__main__":
    main()
