# Benchmarks

v1 → v2 numbers, measured with the same harness against v1 (tag `v1.0`, commit `868c0c8`) and v2
(tag `v2.0`). The harness talks to v1 with `--api legacy`, so v1 rows carry the label `legacy`
in the raw results. Raw results are in
[`benchmarks/results.jsonl`](../benchmarks/results.jsonl).

**Setup.** Apple M5 (10 cores), Docker Desktop 29.1 with 12.5 GB RAM, everything on CPU.
Embeddings: all-MiniLM-L6-v2 (text, 384-d) and CLIP ViT-B/32 (images, 512-d). Whisper `tiny`
int8. LLM: qwen2.5:1.5b. Vision model during these runs: moondream (the default is llava:7b,
which didn't fit in memory next to other containers on this machine).

**Data.** `scripts/make_sample_corpus.py` generates every file, so results are reproducible:

- `samples/corpus/`: 4 PDFs, 2 images, 2 speech recordings and a narrated video (fictional content)
- `samples/bench/` (`--bench`): a 100-page PDF, 10 minutes of speech and a 5-minute 720p video with speech
- `samples/large/` (`--large`): a 1.02 GB, 3-minute 1080p video

```bash
python scripts/make_sample_corpus.py --bench --large
export DAM_PASSWORD=...                      # admin password from .env
python scripts/benchmark.py ingest --reset --dir samples/bench
python scripts/benchmark.py eval -k 5
python scripts/benchmark.py search -n 200 --concurrency 1 8
python scripts/benchmark.py upload --direct --file samples/large/large_upload_test.mp4 --watch <gateway-container> --wait
```

## Ingestion time

The time from the upload response to the task reaching `SUCCESS`. Each number is the mean of 2
runs after a warm-up run, which loads the models.

The v1 worker ran with `OMP_NUM_THREADS=1`. The **"v2, same threads"** column holds that
budget fixed, so its difference comes from the code changes alone. **"v2, default"** uses the
new default of 4 threads per worker process.

| File | v1 | v2, same threads | v2, default | What changed |
|---|---:|---:|---:|---|
| 100-page PDF (600 chunks) | 33.9 s | 3.6 s (**9.4×**) | 3.1 s (**11×**) | One HTTP embedding call per line was replaced by batched local `encode()` |
| 5-min 720p video with speech | 30.5 s | 17.7 s (1.7×) | 12.8 s (**2.4×**) | `grab()` for frames that aren't sampled, batched CLIP, no pydub re-encode |
| 10-min speech (MP3) | 19.9 s | 20.5 s (1.0×) | 13.7 s (1.45×) | Whisper dominates the time; the gain comes only from threads |
| 1.02 GB 3-min 1080p video | 26.3 s | – | 10.2 s (2.6×) | As for the video row |

Image ingestion is slower in v2 (about 5–9 s versus 0.3 s) because each image now gets a vision-model
caption. Set `CAPTION_IMAGES=false` to turn captions off.

## Retrieval quality

There are 22 text questions and 3 visual queries ([`eval/questions.json`](../eval/questions.json)).

- A text question counts as a hit when a top-5 text match comes from the expected file and
  contains the answer phrase. Alternates allow for Whisper writing "eleven" as "11".
- "With distractors" adds the `samples/bench` files to the index: about 600 extra chunks of
  similar business prose, which makes the set harder.

| Index | v1 recall@5 | v1 MRR | v2 recall@5 | v2 MRR |
|---|---:|---:|---:|---:|
| Corpus only | 21/22 (0.955) | 0.788 | **22/22 (1.0)** | **0.955** |
| Corpus + distractors | 19/22 (0.864) | 0.723 | **21/22 (0.955)** | **0.932** |
| Visual (text → image/frame) | 3/3 | – | 3/3 | – |

The gain comes from chunking. v1 indexed each PDF line separately, so a fact that wrapped
onto a second line could not be retrieved. v2 packs whole sentences into chunks of about 500
characters with 100 characters of overlap, and merges transcript segments into timestamped windows.

**Caveat.** I wrote both the corpus and the questions, and both are small. Treat these results as
evidence that the pipeline works and that chunking helped, not as a general retrieval benchmark.

## Search latency

`GET /api/search` (text and visual index, k=5), 200 requests on the corpus index.

| Concurrency | v1 p50 / p95 | v1 throughput | v2 p50 / p95 | v2 throughput |
|---:|---:|---:|---:|---:|
| 1 | 81 / 121 ms | 10.6 req/s | 67 / 93 ms | 13.0 req/s |
| 8 | 556 / 600 ms | 15.1 req/s | 264 / 339 ms | **29.9 req/s (2×)** |

v1 made blocking Qdrant calls inside the `async` handler and ran the two searches one after
the other. v2 encodes and queries both indexes concurrently in the threadpool, which keeps the
event loop free.

## Large uploads and memory

The test file is 1.02 GB. Peak memory was sampled with `docker stats` during the upload.

| Path | Upload time | Gateway memory (before → peak) |
|---|---:|---:|
| v1, multipart through the API | 7.4 s | 1,745 → 1,801 MiB (+56) |
| v2, presigned multipart direct to MinIO | 8.5 s (114 MiB/s) | 825 → 825 MiB (**+0**) |

- **The v1 API path already streamed.** Starlette spools the request to disk, and boto3
  sends it in parts. The real limit was the Streamlit UI, which keeps the whole file in memory
  and caps uploads at 200 MB by default.
- **In v2, large files skip both the UI and the gateway.** The CLI (`scripts/dam_upload.py`) PUTs
  16 MiB parts to presigned URLs. The gateway only signs the URLs, then verifies the final size
  and the file's magic bytes.
- **Idle gateway memory fell from 1,745 to 825 MiB**, with both embedding models loaded.

## Images

Recorded with `python scripts/benchmark.py images <tags>` (`image_sizes` in `results.jsonl`).
The v1 images are the ones built from the `v1.0` code. "Unpacked" is the size on disk that
`docker images` shows. "Compressed" is the image content size from `docker image inspect` (Docker
Desktop's containerd store), which is roughly what a pull downloads.

| Image | v1 unpacked | v2 unpacked | v1 compressed | v2 compressed |
|---|---:|---:|---:|---:|
| API gateway | 11.7 GB | 2.04 GB | 3.93 GB | 0.43 GB |
| Ingestion worker | 11.7 GB (same image) | 2.44 GB | 3.93 GB | 0.54 GB |
| UI | 11.7 GB (same image) | 0.78 GB | 3.93 GB | 0.17 GB |

The v1 image contained 3.3 GB of NVIDIA CUDA libraries, because PyPI's default torch wheel
includes them, plus build tools and ffmpeg dev headers. v2 installs torch from the CPU-only
index and builds three targets from one multi-stage Dockerfile.

## Summaries (LLM worker)

These ran on qwen2.5:1.5b on CPU, as background Celery tasks. The recorded run used
`python scripts/benchmark.py summary --file <name>`, which forces a fresh summary and times it
until the task finishes (`summary` in `results.jsonl`). The earlier runs were timed with an
ad-hoc script during development.

| Asset | Strategy | LLM calls | Recorded run | Earlier runs |
|---|---|---:|---:|---:|
| 10-min transcript (27 windows) | single pass | 1 | 12.2 s | 5.7 s |
| 100-page PDF (600 chunks) | map-reduce, 2 levels | 20 | 333.5 s | 176 s, 374 s |

CPU-bound LLM time varies a lot with machine load, so treat these as a range (about 3–6 minutes
for the 100-page PDF on this laptop), not a single number.

The first version ran map-reduce partials two at a time inside the HTTP request and took 404 s.
The two concurrent calls only queued behind each other in Ollama, which serves one request at a
time, and they hit timeouts. Summaries now run sequentially on a dedicated `llm` queue.
