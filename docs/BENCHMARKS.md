# Benchmarks

v1 → v2 numbers, measured with the same harness against v1 (tag `v1.0`, commit `868c0c8`) and v2.
The harness talks to v1 with `--api legacy`, so v1 records carry `"api": "legacy"`. Raw results are
in [`benchmarks/results.jsonl`](../benchmarks/results.jsonl); the [run index](#run-index) at the end
says which records feed each table.

**Setup.** Apple M5 (10 cores), Docker Desktop 29.1 with 12.5 GB RAM, everything on CPU.
Embeddings: all-MiniLM-L6-v2 (text, 384-d) and CLIP ViT-B/32 (images, 512-d). Whisper `tiny`
int8. LLM: qwen2.5:1.5b. Vision model during these runs: moondream (the default is llava:7b,
which didn't fit in memory next to other containers on this machine, so it was never tested).

**Data.** `scripts/make_sample_corpus.py` generates every file, so results are reproducible:

- `samples/corpus/`: 4 PDFs, 2 images, 2 speech recordings and a narrated video (fictional content)
- `samples/bench/` (`--bench`): a 100-page PDF, 10 minutes of speech and a 5-minute 720p video with speech
- `samples/large/` (`--large`): a 1.02 GB, 3-minute 1080p video

```bash
python scripts/make_sample_corpus.py --bench --large
export DAM_PASSWORD=...                      # admin password from .env
python scripts/benchmark.py ingest --reset --dir samples/bench
python scripts/benchmark.py eval -k 5
python scripts/benchmark.py answers
python scripts/benchmark.py search -n 200 --concurrency 1 8
python scripts/benchmark.py upload --direct --file samples/large/large_upload_test.mp4 --watch <gateway-container> --wait
```

**Caveats that apply to everything below.**
- One laptop, and the load generator runs on the same machine as the stack.
- Five runs per condition for the headline tables; the upload and memory test ran once.
- The corpus and questions are mine and small (see [Retrieval quality](#retrieval-quality)).
- The v1 rerun in November was rebuilt from `v1.0`. v1 doesn't pin its dependencies, so pip
  resolved torch 2.15.0 and ctranslate2 4.9.0, where the October baseline had 2.14.0 and 4.8.2.

## Ingestion time

The time from the upload response to the task reaching `SUCCESS`, after a warm-up run that loads
the models. **Median of 5 runs, with the range across runs in brackets.**

The v1 worker ran with `OMP_NUM_THREADS=1`. The **"v2, same threads"** column holds that budget
fixed, so its difference comes from the code changes alone. **"v2, default"** uses the new default
of 4 threads per worker process, so part of that gain is just more threads.

| File | v1 | v2, same threads | v2, default | What changed |
|---|---:|---:|---:|---|
| 100-page PDF (600 chunks) | 31.1 s (30.8–32.2) | 3.65 s (3.46–4.88), **8.5×** | 3.27 s (3.02–3.34), **9.5×** | One HTTP embedding call per line was replaced by batched local `encode()` |
| 5-min 720p video with speech | 31.2 s (30.8–31.9) | 15.6 s (15.3–16.6), 2.0× | 12.6 s (11.8–15.4), **2.5×** | `grab()` for frames that aren't sampled, batched CLIP, no pydub re-encode |
| 10-min speech (MP3) | 20.5 s (19.7–21.2) | 17.6 s (17.2–22.9), 1.2× | 13.6 s (13.2–14.6), 1.5× | Whisper dominates the time; most of the gain is threads |

Earlier two-run measurements (October, original v1 environment) gave the same picture: PDF
33.9 s → 3.6 s (same threads) / 3.1 s (default), video 30.5 s → 17.7 s / 12.8 s, and a 1.02 GB
3-minute 1080p video 26.3 s → 10.2 s.

Image ingestion is slower in v2 (about 5–9 s versus 0.3 s) because each image now gets a vision-model
caption. Set `CAPTION_IMAGES=false` to turn captions off.

## Retrieval quality

There are 22 text questions and 3 visual queries ([`eval/questions.json`](../eval/questions.json)).
Retrieval is deterministic (same models, same index), so each condition was run once.

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

**Caveat.** The gain is 2 questions out of 22, on a corpus and question set I wrote. Treat it as
evidence that the pipeline works and that chunking helped, not as a general retrieval benchmark.

## Answer quality (RAG)

`python scripts/benchmark.py answers` asks the same 22 questions through `/api/rag` (v2 only; v1's
RAG endpoint was not scored). An answer is **correct** when it contains one of the expected answer
phrases. A **citation is correct** when one of the sources the answer cites is the expected file.
LLM output varies between runs, so this was run 3 times.

| Metric (corpus only, top-5 sources, qwen2.5:1.5b) | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Correct answers | 21/22 | 21/22 | 21/22 |
| Answers citing the right file | 19/22 | 20/22 | 17/22 |
| "I don't know" replies | 0 | 0 | 0 |
| Median time per answer | 0.75 s | 0.83 s | 1.13 s |

- **What it catches and what it doesn't.** Scoring is by key phrase, not by a human or a judge
  model, so a correct fact inside an otherwise muddled answer still counts.
- **The answer key was widened once, after the first run.** That run scored 20/22. One miss was a
  correct answer ("$75 per day") that the key, "75 dollars per day", didn't match, so "75 per day"
  was added as an alternate. The three runs in the table use the widened key.
- **The steady miss is "Who is on call next week?"** Whisper `tiny` transcribed that sentence as
  "Pre-alone the on-call rotation", and the model answered from a different sentence.
- **Citations are the weaker part.** The 1.5B model sometimes cites the wrong source number even
  when the answer is right.
- **The first run of this eval found a bug.** The model wrote `[Source 2]`, which the citation
  parser ignored, so correct citations were dropped. The parser now accepts `[Source 2]`,
  `[Sources 1 and 3]` and `[1, 3]`. That first run (20/22 correct) is kept in the raw results and
  isn't used in the table.

### Citation prompt A/B

`scripts/citation_ab.py` checks one prompt decision directly. For each of the 22 questions it
retrieves the same top-5 sources once, then asks the model twice:
- **A, v2.0's original prompt:** the citation rules are only in the system prompt.
- **B, the current prompt:** a short citation instruction placed right before "Answer:".

Same model, system prompt and citation parser. Run twice:

| | A: original prompt | B: current prompt |
|---|---:|---:|
| Answers with any citation | 13/22, 15/22 | **22/22, 22/22** |
| Answers citing the expected file | 13/22, 15/22 | **19/22, 19/22** |

An earlier informal check (5 tries of one question, 0/5 → 5/5) led to the change. This A/B is the
recorded version of it.

## Search latency

`GET /api/search` (text and visual index, k=5), 200 requests per run on the corpus index.
**Median of 5 runs, with the range in brackets.**

| Concurrency | v1 p50 / p95 | v1 throughput | v2 p50 / p95 | v2 throughput |
|---:|---:|---:|---:|---:|
| 1 | 83 / 106 ms | 11.0 req/s (10.7–11.1) | 65 / 81 ms | 13.5 req/s (12.8–13.6) |
| 8 | 556 / 609 ms | 15.4 req/s (14.1–15.6) | 275 / 350 ms | **28.2 req/s (25.6–30.3), 1.8×** |

v1 made blocking Qdrant calls and model encoding inside the `async` handler and ran the two
searches one after the other. v2 runs encoding and both Qdrant searches concurrently in the
threadpool, which keeps the event loop free. The October single runs (15.1 → 29.9 req/s) sit
inside these ranges.

## Large uploads and memory

The test file is 1.02 GB. Peak memory was sampled with `docker stats` during the upload. This
ran once per version.

| Path | Upload time | Gateway memory (before → peak) |
|---|---:|---:|
| v1, multipart through the API | 7.4 s | 1,745 → 1,801 MiB (+56) |
| v2, presigned multipart direct to MinIO | 8.5 s (114 MiB/s) | 825 → 825 MiB (**+0**) |

- **The v1 API path already streamed.** Starlette spools the request to disk, and boto3
  sends it in parts. The real limit was the Streamlit UI, which keeps the whole file in memory
  and caps uploads at 200 MB by default.
- **In v2, large files skip both the UI and the gateway**, so +0 MiB is by design rather than a
  tuning result. The CLI (`scripts/dam_upload.py`) PUTs 16 MiB parts to presigned URLs. The
  gateway only signs the URLs, then verifies the final size and the file's magic bytes.
- **Idle gateway memory fell from 1,745 to 825 MiB**, with both embedding models loaded.

## Images

Recorded with `python scripts/benchmark.py images <tags>` (`image_sizes` in `results.jsonl`).
"Unpacked" is the size on disk that `docker images` shows. "Compressed" is the image content size
from `docker image inspect` (Docker Desktop's containerd store), which is roughly what a pull
downloads. v1 builds one image for all three services.

| Image | v1 unpacked | v2 unpacked | v1 compressed | v2 compressed |
|---|---:|---:|---:|---:|
| API gateway | 11.7 GB | 2.04 GB | 3.93 GB | 0.43 GB |
| Ingestion worker | 11.7 GB (same image) | 2.44 GB | 3.93 GB | 0.54 GB |
| UI | 11.7 GB (same image) | 0.78 GB | 3.93 GB | 0.17 GB |

The v1 image contained 3.3 GB of NVIDIA CUDA libraries, because PyPI's default torch wheel
includes them, plus build tools and ffmpeg dev headers. v2 installs torch from the CPU-only
index and builds three targets from one multi-stage Dockerfile. The November rebuild of v1 came
to 12.0 GB unpacked / 3.98 GB compressed, because of the newer torch.

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
time, and they hit timeouts. Because read timeouts were retried, one request in that version ran
for about 12 minutes across four timed-out attempts before failing. This was seen in the gateway
logs on Oct 15 and isn't in `results.jsonl`. Summaries now run sequentially on a dedicated `llm` queue.

## Run index

Every record in `results.jsonl`, grouped by label. Line numbers are 1-based.

| Label(s) | Lines | What it is | Used in |
|---|---|---|---|
| `baseline-run1/2`, `baseline` (ingest) | 1–2, 5 | v1, small corpus, October | Not used (files too small to time) |
| `baseline-bench-run1/2` | 3–4 | v1 ingestion, October | "Earlier two-run measurements" |
| `baseline` (eval, search, upload) | 6–8 | v1 retrieval before the answer-key fix; v1 search; v1 1 GB upload | Upload table (line 8) |
| `v2-threads1-bench-run1/2`, `v2-default-bench-run1/2` | 9–12 | v2 ingestion, October | "Earlier two-run measurements" |
| `v2` (ingest, eval, search, upload) | 13–16 | v2 first full pass; eval before the answer-key fix | Upload table (line 16) |
| `v2-fixed-key`, `legacy-fixed-key` | 17, 21 | Retrieval after the answer-key fix (Whisper "eleven" → "11", hyphens), corpus only | Retrieval table |
| `v2-with-distractors`, `legacy-with-distractors` | 20, 23 | Retrieval with the bench files added | Retrieval table |
| `v2-corpus`, `v2-distractors`, `legacy-distractors`, `v2-final-distractors`, `v2.1-distractors`, `v2.0.2-corpus` | 18–19, 22, 24, 28, 34, 42 | Setup ingests before an eval | Not used |
| `v2-final`, `v2-final-with-distractors` | 25–27, 29 | Re-check on the final October build | Not used (same as lines 17/20) |
| *(no label)* | 30–31 | Run during my manual UI testing on Oct 16. `retention_policy.pdf` wasn't in the index at the time (most likely removed in the delete scenario): all three misses are questions about that file, and none of its chunks appear in their top results. Its 31.6 req/s is within the normal range. | Excluded |
| `v2.1-dedup`, `v2.1-with-distractors`, `v2.1-dedup-c1-rerun1/2` | 32–33, 35–37 | Re-check after the dedup change | Not used (same retrieval; latency inside the ranges above) |
| `v1-images`, `v2-images` | 38–39 | Image sizes | Images table |
| `v2-summary` | 40–41 | Summary timings | Summaries table |
| `v2.0.2-answers` | 43 | First answer eval, before the citation-parser fix | Excluded (see Answer quality) |
| `v2.0.2-answers-run1/2/3` | 44–46 | Answer eval | Answer quality table |
| `v2.0.2-bench-default-run1–5`, `v2.0.2-bench-threads1-run1–5` | 47–51, 57–61 | v2 ingestion, 5 runs per thread setting | Ingestion table |
| `v2.0.2-search-run1–5` | 52–56 | v2 search, 5 runs | Search table |
| `v1-rebuilt-bench-run1–5`, `v1-rebuilt-search-run1–5`, `v1-rebuilt-images` | 62–72 | v1 rebuilt from `v1.0`: ingestion, search, image size | Ingestion and search tables; Images note |
| `v2.0.2-citation-ab-run1/2` | 73–74 | Citation prompt A/B, 22 questions × 2 prompts | Citation prompt A/B |
