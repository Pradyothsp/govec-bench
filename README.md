# govec-bench

Benchmarks for [GoVec](https://github.com/Pradyothsp/govec), a compact vector search engine,
against Chroma and Qdrant: insert and query latency, recall, memory, disk and cold start. Every
database runs a pinned, published Docker image under identical resource limits, so anyone can
reproduce the numbers.

## Results

> Measured on October 8, 2026, against GoVec 0.2.1. The DBpedia 100k and SIFT 10k tables are
> medians of three runs, each with the databases in a fresh random order; the gRPC rows and the
> memory-mapped comparison are medians of three further runs each; the DBpedia 10k row and SIFT's
> RAM and disk at 100k are single runs. The ef sweep is a single run from October 6, with
> GoVec as the 0.2.0 image. The earlier SIFT-only results (GoVec 0.1.0, `ef_search=50`) are at
> harness tag [`v0.1.0`](https://github.com/Pradyothsp/govec-bench/tree/v0.1.0). The raw runs
> behind every figure are in [`results/published/`](results/published/).

On a MacBook (Apple Silicon) with Docker Desktop, one database at a time, 2 CPUs and 2 GB each.
Every database builds HNSW with `M=16`, `ef_construction=100` and searches with ef 100, cosine.

### Text embeddings: DBpedia, 1536 dimensions, 100k vectors

From `task bench:pipeline` (`SIZE=large`): every number for a database comes from one index.

| | GoVec | GoVec (int8) | Chroma | Qdrant |
|---|---:|---:|---:|---:|
| Query latency, k=10 (mean) | 6.08 ms | 6.25 ms | 5.53 ms | **4.87 ms** |
| *the same at 10k vectors* | *4.85 ms* | *5.15 ms* | *5.26 ms* | ***4.75 ms*** |
| Query latency over gRPC, k=10 (mean)‖ | 3.80 ms | | | 4.49 ms |
| Single insert, per vector (10k set) | **4.64 ms** | 4.92 ms | 12.15 ms | 4.69 ms |
| Batch insert, per vector | 1.96 ms | 2.03 ms | 2.14 ms | **1.10 ms**† |
| Recall@1 | **100%** | 86.0% | 99.0% | **100%** |
| Recall@10 | 98.3% | 88.4% | 98.5% | **99.3%** |
| Recall@50 | 96.6% | 89.0% | 96.8% | **98.1%** |
| Restart with the data loaded (median)§ | 873 ms | 612 ms | 309 ms | **211 ms** |
| RAM | 1,338 MB | 458 MB | 712 MB | 354 MB* |
| Disk (allocated) | 632 MB | **171 MB** | 659 MB | 776 MB |

### Image descriptors: SIFT, 128 dimensions, 10k vectors (RAM and disk at 100k)

| | GoVec | GoVec (int8) | Chroma | Qdrant |
|---|---:|---:|---:|---:|
| Query latency, k=10 (mean) | 2.07 ms | **1.91 ms** | 2.93 ms | 2.33 ms |
| Query latency over gRPC, k=10 (mean)‖ | 0.81 ms | | | 1.95 ms |
| Single insert, per vector | 1.94 ms | 1.82 ms | 6.69 ms | **1.70 ms** |
| Batch insert, per vector | 0.18 ms | 0.19 ms | 0.22 ms | **0.11 ms**† |
| Recall@1 | 99.0% | 96.0% | 99.0% | **100%** |
| Recall@10 | 99.8% | 93.9% | 99.8% | **100%** |
| Recall@50 | 99.7% | 96.2% | 99.7% | **99.8%** |
| RAM | 242 MB | 168 MB | **108 MB** | 315 MB* |
| Disk (allocated) | 67 MB | **29 MB** | 87 MB | 193 MB |

\* Qdrant keeps its vectors on disk by default and lets the OS cache only part of them, so its
figure is mostly file cache and depends on how much of it is cached when sampled: 296–468 MB
across the three DBpedia 100k runs. It isn't comparable with GoVec's and Chroma's, which hold every
vector in memory. Compare it with GoVec's own disk-backed mode
[below](#govec-in-memory-and-memory-mapped).
† Qdrant's upserts return before its index is built and it indexes in the background on both
CPUs; the figure includes waiting for that to finish. Qdrant's indexing threshold is lowered so
it indexes at all (see [Setup](#setup)).
§ Timed from `docker compose start`, so every figure includes about 130 ms of Docker starting the
container. GoVec reads its whole snapshot into memory before it answers; Qdrant and Chroma map
their files and answer at once, reading pages in as queries touch them. Empty, every database
answers within 130–180 ms, almost all of it Docker's, so that's not in the tables.
‖ The same index queried through each client's gRPC mode (Chroma's client has none), from three
separate runs of GoVec and Qdrant side by side; recall matched REST exactly in every run. In
those runs, REST at k=10 was 6.10 ms for GoVec and 3.88 ms for Qdrant on DBpedia, and 2.13 ms and
2.36 ms on SIFT. gRPC cut GoVec's latency by 38–57% but didn't help Qdrant's: 13% slower on
DBpedia, 4% faster on SIFT, and 39% slower on SIFT at k=50. Qdrant's Python client likely spends
what gRPC saves on turning each response into its own model objects. These rows compare client
libraries as much as databases, so no figure in them is marked best.

### GoVec in memory and memory-mapped

GoVec can keep its vectors in memory-mapped files instead of its heap (`enable_mmap`). Same data,
DBpedia 100k, medians of three runs of the two side by side (`--db govec --db govec-mmap`):

| | In memory (default) | Memory-mapped |
|---|---:|---:|
| RAM in use (docker stats) | 1,336 MB | **836 MB** |
| of which the GoVec process | 1,307 MB | **152 MB** |
| Disk (allocated) | 632 MB | 632 MB |
| Recall@10 | 98.4% | 98.3% |
| Query latency, k=10 (mean) | 6.18 ms | 6.16 ms |
| Query latency, k=10 (p99) | 8.32 ms | 7.93 ms |
| Batch insert, per vector | 1.97 ms | 1.96 ms |
| Restart with the data loaded (median) | 949 ms | **467 ms** |

Memory-mapped, the vectors live in the OS file cache instead of the Go heap: 37% less memory in
use, half the restart time, and the same recall, disk and query latency, with the data fitting
under the 2 GB cap. Query speed is the least settled figure: in an earlier set of three runs,
memory-mapped queries were 27% slower in every run, and here they're level. The cause isn't
pinned down; the likeliest is how much of the files the OS kept cached, which a memory-mapped
search depends on and the in-memory one doesn't.

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
  is the slowest: 1.1x Chroma and 1.25x Qdrant. Its latency grows 25% from 10k to 100k while
  Chroma's and Qdrant's grow 3–5%, and higher ef widens the gap (at 10k, Qdrant is 1.3x faster at
  ef 200 and 1.5x at ef 400). GoVec's distance computation is plain Go without SIMD, and its graph
  stores neighbour lists as pointers, which costs more as the graph outgrows the CPU caches.
  These are end-to-end figures, each through its database's Python client, and the client
  matters as much as the database: through each one's fastest client, GoVec over gRPC (3.80 ms)
  and Qdrant over REST (3.88 ms in the same runs) are level at 100k. The growth from 10k to 100k
  is the better measure of search itself, and there GoVec falls behind.
- **Queries on SIFT:** at 128 dimensions search is cheap: GoVec answers in under 1 ms over gRPC,
  and Chroma's and Qdrant's latency barely changes from ef 25 to 400. What the table measures there
  is mostly each client library and HTTP, so it says little about the databases.
- **Inserts:** Qdrant loads fastest in batches, 1.6–1.8x faster than GoVec, by indexing in
  the background on both CPUs. One at a time, GoVec and Qdrant are level on text embeddings and
  Qdrant is 12% faster on SIFT. Chroma's single inserts are 2.6–3.9x slower than the fastest.
- **Restarts:** with 100k vectors loaded, GoVec is the slowest to answer (about 0.9 s), since it
  reads everything back first; Qdrant takes 0.2 s and Chroma 0.3 s.
- **Where GoVec loses:** query latency at 100k, batch inserts, restarts with data, and RAM: in
  memory it holds about 1.9x Chroma's, roughly twice the raw vectors (an upper bound for Go; see
  [Caveats](#caveats)). Its memory-mapped mode cuts memory in use by 37% and halves restart
  time; see below.
- **int8 quantization** cuts GoVec's disk by 57–73% and RAM by 31–66%, but costs 6–10 points of
  recall, most on text embeddings: its value range doesn't fit embedding vectors well yet.

## What's measured

| Benchmark | Dataset | What it measures |
|---|---|---|
| Insert latency | 10k vectors | Wall time per vector to load 10,000 vectors, one at a time or in batches of 100, until all are indexed (Qdrant indexes in the background, so its wait for that is included) |
| Query latency | 10k or 100k vectors | Time for each of 100 queries at k = 1, 5, 10, 50 |
| Recall | 10k or 100k vectors | Share of each query's true top-k neighbours returned, against exact cosine neighbours computed by brute force |
| Memory and disk | 100k vectors | Memory in use (`docker stats`, plus the process/file-cache split) and allocated disk after loading, relative to an empty baseline |
| Restart | — or loaded | Time from `docker compose start` until the database answers a query, over 5 restarts, empty or with the data loaded; includes Docker's own start-up, the same for every database |
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
| Qdrant | `qdrant/qdrant:v1.19.0` | HNSW, its defaults (`m=16`, `ef_construct=100`, search ef 100), cosine; `indexing_threshold` lowered (below) |

- **Same limits:** 2 CPUs and 2 GB per container (`docker-compose.yml`).
- **One database at a time:** each benchmark starts only the container it measures, on fresh
  volumes.
- **Search ef set explicitly** to 100 for every database, though it is each one's default.
- **Stock settings** apart from matching HNSW parameters and the metric, with one exception:
  Qdrant's indexing threshold is lowered so it builds an HNSW index at all. On stock settings it
  leaves segments under 10 MB unindexed and searches them by brute force: all of SIFT 10k, and
  15% of DBpedia 10k.
- **Same client behaviour:** every adapter reuses one connection; GoVec is driven by its
  published Python SDK over REST, as are Chroma and Qdrant through their official clients
  (GoVec and Qdrant also over gRPC, in rows marked ‖).

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

Latency on a laptop moves between runs by tens of percent, and whole nights shift together: in
an earlier set of three runs every database's single inserts were 2–9% slower than in the
published set, and Qdrant's 30% slower, with no change to any image or code path. So the tables
are medians of three runs, each run starts the databases in a fresh random order (recorded in
the results as `run_position`), and they compare databases within the same set. In one DBpedia
100k run Qdrant's k=10 p99 was 82 ms against about 7 ms in the other two; the medians aren't
moved by it.

## Caveats

- **A laptop, through Docker Desktop.** Docker Desktop on macOS makes `fsync` unusually cheap,
  which flatters every database's write path. The numbers compare the databases with each other,
  not with production hardware.
- **GoVec's RAM is an upper bound.** RSS reflects when Go's garbage collector returns memory, not
  live data; the same load has measured anywhere from 115 MB (live heap) to 593 MB (RSS) depending
  on timing. Setting `GOMEMLIMIT` cuts the measured figure by about 46%, but it's left off so
  every database runs on stock settings.
- **Size changes the picture.** GoVec's query latency on text embeddings grows 25% from 10k to
  100k vectors while the others' grow 3–5%, and recall moves too. Results are shown at 100k where
  it matters; at millions of vectors they would shift again.
- **Recall moves between builds.** HNSW graphs differ slightly between builds, so recall moves
  by about one query in a hundred between runs. In about 1 SIFT build in 11 (4 of 46, in earlier
  runs), Qdrant's graph search at ef 100 misses one query's neighbourhood entirely, the same way
  on every repeat; the vectors are stored, and an exact search or ef 400 finds them. None of the
  published runs had it.
- **One client, one query at a time.** Latency is end to end through each database's Python
  client, and nothing measures throughput under concurrent load. With 100 queries per run, p99
  is close to the single slowest query.

## Contributing

[AGENTS.md](AGENTS.md) describes how the harness is organized, the rules that keep the comparison
fair, and how to add a database.

## License

[MIT](LICENSE)
