# govec-bench

Benchmarks for [GoVec](https://github.com/Pradyothsp/govec), a compact vector search engine,
against Chroma and Qdrant: insert and query latency, recall, memory, disk and cold start. Every
database runs a pinned, published Docker image under identical resource limits, so anyone can
reproduce the numbers.

## Results

> **Preliminary.** Single runs on October 6, 2026, on a laptop that wasn't kept idle. Directions
> are clear; exact figures will move. These will be replaced by medians of repeated runs against
> a published GoVec release. GoVec ran as the 0.2.0 image or a build of its `main` branch after
> it (since released as 0.2.1), both with the index settings this repo's `govec-config.yaml`
> sets. The earlier SIFT-only results (GoVec 0.1.0, `ef_search=50`) are at harness tag
> [`v0.1.0`](https://github.com/Pradyothsp/govec-bench/tree/v0.1.0).

On a MacBook (Apple Silicon) with Docker Desktop, one database at a time, 2 CPUs and 2 GB each.
Every database builds HNSW with `M=16`, `ef_construction=100` and searches with ef 100, cosine.

### Text embeddings: DBpedia, 1536 dimensions, 100k vectors

From `task bench:pipeline` (`SIZE=large`): every number for a database comes from one index.

| | GoVec | GoVec (int8) | Chroma | Qdrant |
|---|---:|---:|---:|---:|
| Query latency, k=10 (mean) | 6.22 ms | 5.96 ms | 5.21 ms | **3.98 ms** |
| *the same at 10k vectors* | *4.72 ms* | *4.66 ms* | *5.35 ms* | ***4.38 ms*** |
| Single insert (mean, 10k set) | **4.57 ms** | 4.65 ms | 11.76 ms | 5.82 ms |
| Batch insert, per vector | **1.90 ms** | 1.98 ms | 2.07 ms | 8.23 ms |
| Recall@1 | **99.0%** | 86.0% | **99.0%** | 98.0% |
| Recall@10 | 98.4% | 88.4% | **98.5%** | 97.5%† |
| Recall@50 | 96.4% | 89.1% | 96.6% | **97.2%** |
| Restart, empty (median) | **7.9 ms** | 8.3 ms | 55.8 ms | 10.3 ms |
| Restart with the data loaded (median) | 981 ms | 485 ms | 181 ms | **78 ms**§ |
| RAM | 1,329 MB | 463 MB | 707 MB | 161 MB* |
| Disk | 632 MB | **171 MB** | 659 MB | 867 MB‖ |

### Image descriptors: SIFT, 128 dimensions, 10k vectors (RAM and disk at 100k)

| | GoVec | GoVec (int8) | Chroma | Qdrant |
|---|---:|---:|---:|---:|
| Query latency, k=10 (mean) | **1.37 ms** | 1.41 ms | 2.63 ms | 2.45 ms |
| Single insert (mean) | **1.80 ms** | 1.88 ms | 7.53 ms | 2.00 ms |
| Batch insert, per vector | **0.17 ms** | 0.18 ms | 0.21 ms | 0.88 ms |
| Recall@1 | **99.0%** | 96.0% | **99.0%** | 98.0%† |
| Recall@10 | **99.8%** | 93.9% | **99.8%** | 98.8%† |
| Recall@50 | **99.7%** | 96.2% | **99.7%** | 98.8%† |
| Restart, empty (median) | **8.5 ms** | 8.7 ms | 55.7 ms | 11.1 ms |
| RAM | 247 MB | 165 MB | **109 MB** | 136 MB* |
| Disk | 67 MB | **29 MB** | 87 MB | 15,086 MB‡ |

\* Qdrant keeps its vectors on disk by default and lets the OS cache only part of them (at
DBpedia 100k, about 200 MB of its 614 MB, measured before any queries), so its RAM isn't
comparable with GoVec's and Chroma's, which hold every vector in memory. Compare it with GoVec's
own disk-backed mode [below](#govec-in-memory-and-memory-mapped).
† One query found almost none of its true neighbours (none at all on SIFT; without it Qdrant scores
99.0%, 99.8% and 99.8% there). In about 1 SIFT build in 11 (4 of 46), Qdrant's graph search at its
default ef misses one query's neighbourhood entirely, the same way on every repeat; it happened
at DBpedia 100k too. The vectors are stored, and an exact search or ef 400 finds them.
‡ Apparent size, which counts space Qdrant had reserved but not written, likely mid-optimization.
The harness now records allocated size too.
§ GoVec reads its whole snapshot into memory before it answers; Qdrant and Chroma map their files
and answer at once, reading pages in as queries touch them.
‖ Allocated size, from a separate run (apparent: 1,219 MB); the main run predates that measurement.
The other disk figures are apparent size from the main run.

### GoVec in memory and memory-mapped

GoVec can keep its vectors in memory-mapped files instead of its heap (`enable_mmap`). Same data,
DBpedia 100k, `--db govec --db govec-mmap`:

| | In memory (default) | Memory-mapped |
|---|---:|---:|
| RAM in use (docker stats) | 1,343 MB | **816 MB** |
| of which the GoVec process | 1,319 MB | **149 MB** |
| Disk (allocated) | 632 MB | 632 MB |
| Recall@10 | 98.0% | 97.7% |
| Query latency, k=10 (mean) | 6.28 ms | 6.53 ms |
| Batch insert, per vector | 1.89 ms | 1.96 ms |
| Restart with the data loaded (median) | 866 ms | **418 ms** |

Memory-mapped, the vectors live in the OS file cache instead of the Go heap: 40% less memory in
use, half the restart time, and the same recall, query latency and disk while the data fits in
memory (it does here, under the 2 GB cap).

### Recall against latency: the ef sweep (DBpedia, 10k vectors)

Each database at five search ef values, median query latency at k=10. Recall@10 across the three
databases is within 3 points at ef 25 and within 0.1 point from ef 100 up.

| ef | Recall@10 (GoVec) | GoVec | Chroma | Qdrant |
|---:|---:|---:|---:|---:|
| 25 | 92.4% | 4.02 ms | **3.71 ms** | 3.96 ms |
| 100 | 99.3% | 4.96 ms | 5.28 ms | **4.86 ms** |
| 200 | 100% | 5.93 ms | 6.89 ms | **4.67 ms** |
| 400 | 100% | 7.83 ms | 9.29 ms | **5.09 ms** |

What the numbers say:

- **Recall:** at the same ef, all three find the same neighbours: 97–100% at k=10. At 100k a single
  index build moves recall by about a point, so the order between them in one run means little.
- **Queries on text embeddings:** at 10k vectors GoVec is level with Qdrant at ef 100. At 100k it
  is the slowest: 1.2x Chroma and 1.6x Qdrant. Its latency grows 32% from 10k to 100k while
  Chroma's and Qdrant's stay flat, and higher ef widens the gap (at 10k, Qdrant is 1.3x faster at
  ef 200 and 1.5x at ef 400). GoVec's distance computation is plain Go without SIMD, and its graph
  stores neighbour lists as pointers, which costs more as the graph outgrows the CPU caches.
- **Queries on SIFT:** GoVec measures fastest, but at 128 dimensions search is cheap, and
  Chroma's and Qdrant's latency barely changes from ef 25 to 400. What's measured there is mostly
  each client library and HTTP, not search.
- **Inserts:** GoVec has the fastest single and batch inserts at equal build effort on both
  datasets; Chroma's single inserts are 2.5–4x slower.
- **Restarts:** empty, GoVec is ready in about 8 ms and Chroma in about 55. With 100k vectors
  loaded, GoVec is the slowest to answer (about 1 s), since it reads everything back first.
- **Where GoVec loses:** query latency at 100k, and RAM: in memory it holds about 1.9x Chroma's,
  roughly twice the raw vectors (an upper bound for Go; see [Caveats](#caveats)). Its
  memory-mapped mode cuts memory in use by 40% at no cost to recall or latency here.
- **int8 quantization** cuts GoVec's disk by 57–73% and RAM by 33–65%, but costs 6–10 points of
  recall, most on text embeddings: its value range doesn't fit embedding vectors well yet.

## What's measured

| Benchmark | Dataset | What it measures |
|---|---|---|
| Insert latency | 10k vectors | Time per vector for 10,000 single inserts, and per vector for batches of 100 |
| Query latency | 10k or 100k vectors | Time for each of 100 queries at k = 1, 5, 10, 50 |
| Recall | 10k or 100k vectors | Share of each query's true top-k neighbours returned, against exact cosine neighbours computed by brute force |
| Memory and disk | 100k vectors | Memory in use (`docker stats`, plus the process/file-cache split) and allocated disk after loading, relative to an empty baseline |
| Restart | — or loaded | Time from container start until the database answers a query, over 5 restarts, empty or with the data loaded |
| ef sweep | 10k vectors | Recall@10 and @50 and query latency at search ef 25, 50, 100, 200, 400 (`task bench:sweep`) |

`task bench:pipeline` runs all of these from one load per database at either size (`SIZE=large`
for 100k), so every number describes the same index; the per-benchmark tasks run one at a time.

Two datasets, chosen with `DATASET=sift` (the default) or `DATASET=dbpedia`:

- **SIFT:** 128-dimensional image descriptors from the standard
  [ANN_SIFT](http://corpus-texmex.irisa.fr/) corpus. SIFT10K, and the first 100,000 vectors of
  SIFT1M.
- **DBpedia:** 1536-dimensional OpenAI `text-embedding-ada-002` embeddings of DBpedia entity
  abstracts ([KShivendu/dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)):
  what a RAG system stores. 10k and 100k base vectors plus 100 held-out queries.

## Setup

| Database | Image | Index |
|---|---|---|
| GoVec | `ghcr.io/pradyothsp/govec:0.2.1` | HNSW, `M=16`, `ef_construction=100`, `ef_search=100`, cosine (set in `govec-config.yaml`) |
| GoVec (int8) | the same, with `quantization: scalar` | as above, vectors stored as int8 |
| Chroma | Chroma 1.4.4, pinned by digest | HNSW, its defaults (`M=16`, `ef_construction=100`, `ef_search=100`), cosine |
| Qdrant | `qdrant/qdrant:v1.19.0` | HNSW, its defaults (`m=16`, `ef_construct=100`; search ef defaults to it), cosine |

- **Same limits:** 2 CPUs and 2 GB per container (`docker-compose.yml`).
- **One database at a time:** each benchmark starts only the container it measures, on fresh
  volumes.
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

task data:download     # SIFT10K and SIFT1M into data/raw/ (DATASET=dbpedia for DBpedia)
task bench:all         # all five benchmarks; each starts one database at a time
task bench:sweep       # recall against latency across search ef
DATASET=dbpedia SIZE=large task bench:pipeline   # everything at 100k text embeddings
```

Each benchmark writes `results/<benchmark>_<timestamp>.json`, with one section per database.
Run single benchmarks with `task bench:insert`, `bench:query`, `bench:recall`, `bench:memory` or
`bench:coldstart`.
Add `-- --db govec --db chroma` to run only some databases.

## Run-to-run variation

Latency on a laptop moves between runs by tens of percent, and whole runs shift together, so
compare databases within a run. The preliminary results above are single runs; the final ones
will report medians and ranges across repeated runs.

## Caveats

- **A laptop, through Docker Desktop.** Docker Desktop on macOS makes `fsync` unusually cheap,
  which flatters every database's write path. The numbers compare the databases with each other,
  not with production hardware.
- **GoVec's RAM is an upper bound.** RSS reflects when Go's garbage collector returns memory, not
  live data; the same load has measured anywhere from 115 MB (live heap) to 593 MB (RSS) depending
  on timing. Setting `GOMEMLIMIT` cuts the measured figure by about 46%, but it's left off so
  every database runs on stock settings.
- **Size changes the picture.** GoVec's query latency on text embeddings grows 32% from 10k to
  100k vectors while the others' stays flat, and recall moves too. Results are shown at 100k where
  it matters; at millions of vectors they would shift again.
- **Recall uses one index build per database.** HNSW graphs differ slightly between builds, so
  recall moves by about one query in a hundred between runs.

## Contributing

[AGENTS.md](AGENTS.md) describes how the harness is organized, the rules that keep the comparison
fair, and how to add a database.

## License

[MIT](LICENSE)
