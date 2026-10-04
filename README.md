# govec-bench

Benchmarks for [GoVec](https://github.com/Pradyothsp/govec), a compact vector search engine,
against Chroma and Qdrant: insert and query latency, recall, memory, disk and cold start. Every
database runs a pinned, published Docker image under identical resource limits, so anyone can
reproduce the numbers.

## Results

October 2026, on a MacBook (Apple Silicon) with Docker Desktop. Each database ran with 2 CPUs and
2 GB. Latencies are medians across 3–4 repeated runs; ranges are in [Run-to-run variation](#run-to-run-variation).

| | GoVec | GoVec (int8) | Chroma | Qdrant |
|---|---:|---:|---:|---:|
| Query latency, k=10 (mean) | 1.54 ms | **1.35 ms** | 2.54 ms | 1.89 ms |
| Single insert (mean) | 2.41 ms | **2.25 ms** | 11.76 ms | 2.42 ms |
| Batch insert, per vector | 0.57 ms | 0.45 ms | **0.35 ms** | 1.21 ms |
| Recall@1 | **99.0%** | 97.0% | **99.0%** | **99.0%** |
| Recall@10 | 99.5% | 93.8% | **99.8%** | 99.7% |
| Recall@50 | 98.5% | 95.6% | 99.7% | **99.7%** |
| Cold start (median) | **12.0 ms** | 14.6 ms | 129.9 ms | 22.6 ms |
| RAM, 100k vectors | 252 MB | 173 MB | **109 MB** | 115 MB |
| Disk, 100k vectors | 119 MB | **47 MB** | 87 MB | 231 MB |

What the numbers say:

- **Queries:** GoVec was about 1.6x faster than Chroma in every run, and on par with Qdrant
  (faster at the median, slower in one run of four).
- **Single inserts:** GoVec and Qdrant tie; Chroma is about 5x slower.
- **Recall:** all three find 99% of true nearest neighbours at k=1. GoVec trails slightly at
  k=10 and by about 1.2 points at k=50, because it searches with `ef_search=50`, leaving no
  margin at k=50. A higher `ef_search` closes the gap at some latency cost.
- **Cold start:** GoVec, a single static binary, is ready in about 12 ms; Chroma takes over 100.
- **Where GoVec loses:** batch-insert throughput (Chroma) and RAM, for which the RSS figure is an
  upper bound for GoVec (see [Caveats](#caveats)).
- **int8 quantization** cuts GoVec's disk by 61% and RAM by 31%, at a recall cost of a few
  points.

## What's measured

| Benchmark | Dataset | What it measures |
|---|---|---|
| Insert latency | SIFT10K | Time per vector for 10,000 single inserts, and per vector for batches of 100 |
| Query latency | SIFT10K | Time for each of 100 queries at k = 1, 5, 10, 50 |
| Recall | SIFT10K | Share of each query's true top-k neighbours returned, against exact cosine neighbours computed by brute force |
| Memory and disk | SIFT100K | Container RSS and on-disk size after loading 100,000 vectors, relative to an empty baseline |
| Cold start | — | Time from container start until the database answers a query, over 5 restarts |

SIFT10K and SIFT100K are 128-dimensional image descriptors from the standard
[ANN_SIFT](http://corpus-texmex.irisa.fr/) corpus; SIFT100K is the first 100,000 vectors of
SIFT1M.

## Setup

| Database | Image | Index |
|---|---|---|
| GoVec | `ghcr.io/pradyothsp/govec:0.1.0` | HNSW, `M=16`, `ef_construction=200`, `ef_search=50`, cosine |
| GoVec (int8) | the same, with `quantization: scalar` | as above, vectors stored as int8 |
| Chroma | Chroma 1.4.4, pinned by digest | HNSW, its defaults, cosine |
| Qdrant | `qdrant/qdrant:v1.19.0` | HNSW, its defaults, cosine |

- **Same limits:** 2 CPUs and 2 GB per container (`docker-compose.yml`).
- **Stock settings** apart from matching HNSW parameters and the metric. Nothing tuned for one
  database only.
- **Same client behaviour:** every adapter reuses one connection; GoVec is driven by its
  published Python SDK over REST.

## Reproduce

Requirements: Docker, [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev/installation/).

```bash
git clone https://github.com/Pradyothsp/govec-bench.git
cd govec-bench
uv sync --all-groups

task data:download     # SIFT10K and SIFT1M into data/raw/
task docker:up         # GoVec, GoVec int8, Chroma and Qdrant
task bench:all         # all five benchmarks, about 15 minutes
task docker:down
```

Each benchmark writes `results/<benchmark>_<timestamp>.json`, with one section per database.
Run single benchmarks with `task bench:insert`, `bench:query`, `bench:recall`, `bench:memory` or
`bench:coldstart`.

## Run-to-run variation

Latency on a laptop moves between runs, so the table reports medians. The ranges behind them:

| | GoVec | GoVec (int8) | Chroma | Qdrant |
|---|---|---|---|---|
| Query latency, k=10 | 1.45–1.95 ms | 1.14–2.24 ms | 2.34–3.25 ms | 1.40–2.85 ms |
| Single insert | 2.40–2.55 ms | 2.01–2.48 ms | 11.26–12.30 ms | 1.98–2.50 ms |
| Cold start | 11.2–13.5 ms | 13.9–15.6 ms | 117–166 ms | 16.7–23.2 ms |

Whole runs shift together, so compare databases within a run. GoVec's query latency was 1.60–1.68x
better than Chroma's in every run; against Qdrant it ranged from 0.96x to 1.46x.

## Caveats

- **A laptop, through Docker Desktop.** Docker Desktop on macOS makes `fsync` unusually cheap,
  which flatters every database's write path. The numbers compare the databases with each other,
  not with production hardware.
- **GoVec's RAM is an upper bound.** RSS reflects when Go's garbage collector returns memory, not
  live data; the same load has measured anywhere from 115 MB (live heap) to 593 MB (RSS) depending
  on timing. Setting `GOMEMLIMIT` cuts the measured figure by about 46%, but it's left off so
  every database runs on stock settings.
- **SIFT is not text embeddings.** It's a standard ANN dataset with uniform vector lengths, not
  the output of an embedding model; relative results should carry over, absolute recall may not.
- **Recall uses one index build per database.** HNSW graphs differ slightly between builds, so
  recall moves by about one query in a hundred between runs.

## Contributing

[AGENTS.md](AGENTS.md) describes how the harness is organized, the rules that keep the comparison
fair, and how to add a database.

## License

[MIT](LICENSE)
