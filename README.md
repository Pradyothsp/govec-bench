# govec-bench

Benchmarks for [GoVec](https://github.com/Pradyothsp/govec), a compact vector search engine,
against Chroma and Qdrant: insert and query latency, recall, memory, disk and cold start. Every
database runs a pinned, published Docker image under identical resource limits, so anyone can
reproduce the numbers.

## Results

> Measured on November 3, 2026, against GoVec 0.3.0, on a GitHub-hosted runner (`ubuntu-latest`:
> AMD EPYC 9V45, 4 vCPUs, 16 GB, Docker on Linux) with
> [`.github/workflows/bench.yml`](.github/workflows/bench.yml). The DBpedia 100k figures are
> medians of three rounds of one session, `experiments/release-0.3.0.yaml`, every database run
> once per round in an order that reverses each round; GoVec 0.2.1 ran in the same session for
> the comparison below. Concurrent queries come from a session of their own, on another processor,
> and say so. The SIFT table and the ef sweep weren't re-run: they are from October
> runs on a Mac, with older GoVec, and say so where they appear. The raw runs behind every figure
> are in [`results/published/`](results/published/) (this session in `sessions/release-0.3.0/`).
>
> `ubuntu-latest` doesn't fix the processor: each session gets whatever VM Azure hands out, and
> `session.json` records which. A later session on an AMD EPYC 7763 (Zen 3) measured every
> database 1.5–1.6x slower than on this EPYC 9V45 (Zen 5). So the milliseconds below belong to
> this machine. Ratios between databases in the same session carry over far better (GoVec's
> one-client latency over Chroma's: 1.46 here, 1.45 there; over Qdrant's: 1.49 and 1.37), so
> compare those across sessions, and read a ratio that moves as a change worth explaining.

One database at a time, 2 CPUs and 2 GB each. Every database builds HNSW with `M=16`,
`ef_construction=100` and searches with ef 100, cosine. What the harness changes from each
database's defaults, and why, is listed under [Setup](#what-the-harness-changes).

### Text embeddings: DBpedia, 1536 dimensions, 100k vectors

From the pipeline at `SIZE=large`: every number for a database comes from one index.

| | GoVec | GoVec (int8) | Chroma | Qdrant |
|---|---:|---:|---:|---:|
| Query latency, k=10 (mean) | 3.17 ms | 4.72 ms | 2.17 ms | **2.13 ms** |
| Query latency over gRPC, k=10 (mean)‖ | 2.72 ms | 4.34 ms | | 1.74 ms |
| Single insert, per vector (10k set) | **1.54 ms** | 2.42 ms | 8.35 ms | 4.25 ms |
| Batch insert, per vector | 0.75 ms | 1.44 ms | **0.65 ms** | 1.19 ms† |
| Recall@1 | 99.0% | 86.0% | 99.0% | **100%** |
| Recall@10 | 98.3% | 88.6% | 98.6% | **99.0%** |
| Recall@50 | 96.5% | 89.1% | 96.8% | **98.0%** |
| Restart with the data loaded (median)§ | 928 ms | 713 ms | 489 ms | **290 ms** |
| RAM, the database's process | 1,299 MB | 451 MB | 682 MB | 14 MB* |
| RAM, docker stats | 1,320 MB | 457 MB | 703 MB | 83 MB* |
| Disk (allocated) | 632 MB | **171 MB** | 659 MB | 680 MB |

\* Qdrant keeps its vectors in memory-mapped files by default, so they sit in the OS file cache
(666 MB here), not in its process, and docker stats counts only the part of that cache in active
use. It isn't comparable with GoVec's and Chroma's figures, which hold every vector in their
process. Compare it with GoVec's own memory-mapped mode [below](#govec-in-memory-and-memory-mapped).
† Qdrant's upserts return before its index is built, and it indexes in the background on both
CPUs; the figure includes waiting for that to finish.
§ Timed from `docker compose start`, so every figure includes Docker starting the container.
GoVec reads its whole snapshot into memory before it answers; Qdrant and Chroma map their files
and answer at once, reading pages in as queries touch them. Empty, every database answers within
155–240 ms, almost all of it Docker's, so that's not in the table.
‖ The same index, queried through each client's gRPC mode (Chroma's client has none); recall
matched REST. These rows compare client libraries as much as databases.

### GoVec 0.3.0 against 0.2.1

Both releases in the same session, in alternating order, DBpedia 100k:

| | 0.2.1 | 0.3.0 | Change |
|---|---:|---:|---:|
| Query latency, k=10 (mean) | 5.58 ms | 3.17 ms | −43% |
| Query latency over gRPC, k=10 (mean) | 5.26 ms | 2.72 ms | −48% |
| Batch insert, per vector | 1.43 ms | 0.75 ms | −47% |
| Single insert, per vector (10k set) | 2.40 ms | 1.54 ms | −36% |
| Restart with the data loaded (median) | 1,320 ms | 928 ms | −30% |
| Recall@10 | 98.2% | 98.3% | level |
| RAM, process | 1,305 MB | 1,299 MB | level |

Every run of 0.3.0 beat every run of 0.2.1 on each of the first five rows. 0.3.0 computes the
float32 dot product with SIMD (AVX2 on this runner), allocates far less per search, and loads its
snapshot without copying each vector three times.

### GoVec in memory and memory-mapped

GoVec can keep its vectors in memory-mapped files instead of its heap (`enable_mmap`). Same
session, DBpedia 100k:

| | In memory (default) | Memory-mapped |
|---|---:|---:|
| RAM, GoVec's process | 1,299 MB | **153 MB** |
| File cache | 632 MB | 624 MB |
| RAM, docker stats | 1,320 MB | 174 MB |
| Disk (allocated) | 632 MB | 632 MB |
| Recall@10 | 98.3% | 98.6% |
| Query latency, k=10 (mean) | 3.17 ms | 3.01 ms |
| Query latency, k=10 (p99) | 4.09 ms | 3.90 ms |
| Batch insert, per vector | 0.75 ms | 0.80 ms |
| Restart with the data loaded (median) | 928 ms | **561 ms** |

Memory-mapped, the vectors live in the OS file cache instead of the Go heap. In memory, the file
cache holds the snapshot file, which the server doesn't need once it's loaded and the OS can
reclaim; memory-mapped, it holds the vectors searches read. So what memory-mapped GoVec needs is
its process plus the file cache, about 777 MB against 1,299 MB in memory, with the same recall,
query latency and disk, and a 40% faster restart. docker stats shows 174 MB because GoVec tells
the OS it reads the files at random, and docker stats leaves out file pages that aren't in
active use: read the process and file cache rows.

### Concurrent queries (DBpedia, 100k vectors)

> A separate session, `experiments/concurrency-0.3.0.yaml`, on November 3–4, 2026, on a
> different runner: an **AMD EPYC 7763 (Zen 3)**, 4 vCPUs, where every database answers one
> query 1.5–1.6x slower than on the EPYC 9V45 above. Compare the databases with each other in
> this table, not its milliseconds with the tables above. Medians of three rounds.

1 to 32 client processes query one loaded index at once, each with its own connection, back to
back for 20 s per level after a 3 s warm-up. Processes, not threads: a Python client spends most
of a query encoding and decoding under the interpreter lock. Every answer is graded, so a
database can't go faster by answering worse: each one's recall@10 stayed the same at every
level, and no query failed. The database's CPU comes from its cgroup, the clients' from their
own process time.

| | GoVec | GoVec, gRPC | Chroma | Qdrant | Qdrant, gRPC |
|---|---:|---:|---:|---:|---:|
| 1 client, queries/s | 209 | 263 | **303** | 287 | 374 |
| Most queries/s (at clients) | 482 (4) | 592 (32) | 480 (2) | **539** (32) | 723 (16) |
| Database CPU at that point (cap 2) | 1.97 | 1.97 | 1.46 | 1.71 | 1.66 |
| Queries/s per database CPU | 244 | 301 | 329 | 315 | 435 |
| p99 at 32 clients | 129 ms | 97 ms | **90 ms** | 101 ms | 71 ms |
| Recall@10 | 98.6% | 98.6% | 98.7% | 98.9% | 98.9% |

The REST columns compare the databases; the gRPC ones, like the gRPC rows above, compare client
libraries as much as databases. Chroma's client has no gRPC mode.

- **GoVec runs out of CPU first.** It reaches its 2-CPU cap at 4 clients; past that more clients
  only queue, and p99 grows with them. Its peak is 0.89x Qdrant's over REST and 0.82x over gRPC,
  and it does about 23% fewer queries per CPU second than Qdrant over REST (31% over gRPC): about
  the gap in its single-query latency, now as a ceiling.
- **Chroma peaks at 2 clients and then loses throughput:** 480 queries/s at 2, about 410 from 4
  clients on, using only 1.3 of its 2 CPUs, so something in it serializes queries. On a Mac it
  stopped at about 1.3 CPUs too, though without losing throughput. Why isn't known.
- **Qdrant's ceiling here is a lower bound.** It levels off at about 1.7 CPUs, under its cap,
  while its client is the heaviest: 1.6–1.9 cores of client CPU at the peak, against 0.4–0.6 for
  GoVec's and Chroma's. Database and clients together use about 3.3–3.6 of the runner's 4 vCPUs,
  so the runner may be what stops it. In one of three rounds its throughput at 8 clients halved
  (252 queries/s, p99 162 ms); the median stands, and the cause isn't known.

### Recall against latency: the ef sweep (DBpedia, 10k vectors)

> From October 6, on a Mac with Docker Desktop, with GoVec 0.2.0: not re-run on 0.3.0, and not
> comparable with the figures above.

Each database at five search ef values, median query latency at k=10. Recall@10 across the three
databases is within 3 points at ef 25 and within 0.1 point from ef 100 up.

| ef | Recall@10 (GoVec) | GoVec | Chroma | Qdrant |
|---:|---:|---:|---:|---:|
| 25 | 92.4% | 4.02 ms | **3.71 ms** | 3.96 ms |
| 100 | 99.3% | 4.96 ms | 5.28 ms | **4.86 ms** |
| 200 | 100% | 5.93 ms | 6.89 ms | **4.67 ms** |
| 400 | 100% | 7.83 ms | 9.29 ms | **5.09 ms** |

### Image descriptors: SIFT, 128 dimensions, 10k vectors (RAM and disk at 100k)

> From October 8, on a Mac with Docker Desktop, with GoVec 0.2.1: medians of three runs, gRPC
> from three further runs, RAM and disk at 100k a single run. Not re-run on 0.3.0, and not
> comparable with the DBpedia figures above. Qdrant's REST figures there predate the
> uncompressed-response change, which makes no difference on a Mac (checked: k=10 2.19 ms
> before it, between k=5's 2.08 and k=50's 2.77).

| | GoVec | GoVec (int8) | Chroma | Qdrant |
|---|---:|---:|---:|---:|
| Query latency, k=10 (mean) | 2.07 ms | **1.91 ms** | 2.93 ms | 2.33 ms |
| Query latency over gRPC, k=10 (mean)‖ | 0.81 ms | | | 1.95 ms |
| Single insert, per vector | 1.94 ms | 1.82 ms | 6.69 ms | **1.70 ms** |
| Batch insert, per vector | 0.18 ms | 0.19 ms | 0.22 ms | **0.11 ms**† |
| Recall@1 | 99.0% | 96.0% | 99.0% | **100%** |
| Recall@10 | 99.8% | 93.9% | 99.8% | **100%** |
| Recall@50 | 99.7% | 96.2% | 99.7% | **99.8%** |
| RAM (docker stats) | 242 MB | 168 MB | **108 MB** | 315 MB* |
| Disk (allocated) | 67 MB | **29 MB** | 87 MB | 193 MB |

### What the numbers say

- **Recall:** at the same ef, all three find the same neighbours on text embeddings: recall@10
  98.3–99.0%. Qdrant's recall@50 is about a point higher.
- **Queries on text embeddings:** at 100k GoVec is the slowest, 1.5x Chroma and Qdrant over REST
  (3.17 ms against 2.17 and 2.13) and 1.6x Qdrant over gRPC, though 0.3.0 nearly halved it. Its
  graph stores neighbour lists as pointers, which costs more as the graph outgrows the CPU caches.
- **Queries on SIFT** (October, Mac): at 128 dimensions search is cheap, so the table measures
  mostly each client library and HTTP, and says little about the databases.
- **Inserts:** one at a time, GoVec is fastest: 2.8x faster than Qdrant and 5.4x faster than
  Chroma. In batches, Chroma is 13% faster than GoVec and Qdrant 1.6x slower, its figure including
  the background indexing it does after its upserts return.
- **Restarts:** with 100k vectors loaded, GoVec is the slowest to answer (0.93 s, or 0.56 s
  memory-mapped), since it reads its whole snapshot first; Chroma takes 0.49 s and Qdrant 0.29 s.
- **Where GoVec loses:** query latency at 100k, batch inserts against Chroma, restarts with data,
  and RAM: in memory its process holds 1.9x Chroma's, about twice the raw vectors (an upper bound
  for Go; see [Caveats](#caveats)). Memory-mapped, it needs about 40% less and restarts 40%
  faster.
- **int8 quantization** cuts GoVec's disk by 73% and its process memory by 65%, but costs about
  10 points of recall on text embeddings, whose value range it doesn't fit well yet. In 0.3.0 it
  is also slower to search than float32 (4.72 ms against 3.17), since only the float32 dot product
  uses SIMD so far, and its batch inserts take twice as long.

## What's measured

| Benchmark | Dataset | What it measures |
|---|---|---|
| Insert latency | 10k vectors | Wall time per vector to load 10,000 vectors, one at a time or in batches of 100, until all are indexed (Qdrant indexes in the background, so its wait for that is included) |
| Query latency | 10k or 100k vectors | Time for each of 100 queries at k = 1, 5, 10, 50 |
| Recall | 10k or 100k vectors | Share of each query's true top-k neighbours returned, against exact cosine neighbours computed by brute force |
| Memory and disk | 100k vectors | Memory in use (`docker stats`, plus the process/file-cache split) and allocated disk after loading, relative to an empty baseline |
| Restart | — or loaded | Time from `docker compose start` until the database answers a query, over 5 restarts, empty or with the data loaded; includes Docker's own start-up, the same for every database |
| Concurrent queries | 10k or 100k vectors | Queries/s, latency, recall@10 and the database's and clients' CPU with 1, 2, 4, 8, 16 and 32 client processes querying at once, 20 s per level (`--steps concurrent-queries,concurrent-queries-grpc`) |
| ef sweep | 10k vectors | Recall@10 and @50 and query latency at search ef 25, 50, 100, 200, 400 (`task bench:sweep`) |

`task bench:pipeline` runs all of these from one load per database at either size (`SIZE=large`
for 100k), so every number describes the same index; `--steps` runs only some of them.

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
| GoVec | `ghcr.io/pradyothsp/govec:0.3.0` | HNSW, `M=16`, `ef_construction=100`, `ef_search=100`, cosine (set in `govec-config.yaml`) |
| GoVec (int8) | the same, with `quantization: scalar` | as above, vectors stored as int8 |
| Chroma | Chroma 1.4.4, pinned by digest | HNSW, its defaults (`M=16`, `ef_construction=100`, `ef_search=100`), cosine |
| Qdrant | `qdrant/qdrant:v1.19.0` | HNSW, its defaults (`m=16`, `ef_construct=100`, search ef 100), cosine; `indexing_threshold` and `full_scan_threshold` lowered (below) |

- **Same limits:** 2 CPUs and 2 GB per container (`docker-compose.yml`).
- **One database at a time:** each benchmark starts only the container it measures, on fresh
  volumes.
- **Search ef set explicitly** to 100 for every database, though it is each one's default.
- **Stock settings** apart from matching HNSW parameters and the metric, and the changes listed
  below, each made so every database does the same work under the same client behaviour.
- **Official clients:** GoVec is driven by its published Python SDK over REST, as are Chroma and
  Qdrant through theirs (GoVec and Qdrant also over gRPC, in rows marked ‖).

### What the harness changes

Every change from a database's or client's defaults, and why. Each was found by a measurement
that didn't add up; the code comments in `govec_bench/adapters/` say more.

| Database | Change | Why |
|---|---|---|
| All | One connection reused for every request | A new connection per request puts a TCP handshake inside every measured latency |
| All | Search ef 100, set explicitly | It's each one's default; set so no result depends on a default changing |
| GoVec | HNSW `M=16`, `ef_construction=100`, `ef_search=100` in `govec-config.yaml`; memory-mapped GoVec is told the dimension at start (`GOVEC_DIMENSIONS`) | Matching parameters; the mmap store sizes its slots from the dimension |
| GoVec | SDK 0.1.3 or later | GoVec's SDK encoded REST requests with `json.dumps` up to 0.1.2, ~2 ms per call at 1536 dimensions, where Qdrant's client encodes in Rust. 0.1.3 uses orjson. Results before November 2026 used an earlier SDK and overstate GoVec's REST latency by that much. It's GoVec's own client, fixed in the SDK, not the harness |
| Chroma | Cosine space and search ef set when the collection is created (`hnsw:space`, `hnsw:search_ef`) | Chroma 1.4.4 ignores a changed search ef on an existing collection |
| Qdrant | Keep-alive turned back on (`limits=httpx.Limits()`) | qdrant-client turns it off for `localhost`, opening a connection per request, which also exhausted Docker Desktop's port forwarding during 10k single inserts |
| Qdrant | REST responses uncompressed (`Accept-Encoding: identity`) | httpx asks for gzip and Qdrant streams it in chunks; on Linux a response of about 1–5 KB (k=8 to 39 here) waited ~45 ms for its last chunk, Nagle against delayed ACK. k=10 measured 44 ms against 2.3 ms at k=5. Docker Desktop on a Mac hid it, so the Mac results weren't affected. GoVec and Chroma don't compress |
| Qdrant | `indexing_threshold` 1 KB, `full_scan_threshold` 10 KB; `m=16`, `ef_construct=100` written out | On stock settings Qdrant leaves segments under 10 MB unindexed and searches them by brute force: all of SIFT 10k, 15% of DBpedia 10k. 0 would turn indexing off; 10 KB is the API's minimum |
| Qdrant | Inserts timed until indexed (`wait_until_indexed()`, polled once at the end of a load) | Qdrant indexes after its upserts return, while GoVec and Chroma index before; polling after every batch put the poll interval inside each batch instead |
| Qdrant | String IDs stored as UUIDv5 points, the original ID in the payload, returned with each result | Qdrant accepts only integer or UUID IDs |

## Reproduce

Requirements: Docker, [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev/installation/).

```bash
git clone https://github.com/Pradyothsp/govec-bench.git
cd govec-bench
uv sync --all-groups

task data:download     # SIFT10K and SIFT1M into data/raw/ (DATASET=dbpedia for DBpedia)
task bench:pipeline    # every measurement on SIFT 10k; starts one database at a time
task bench:sweep       # recall against latency across search ef
DATASET=dbpedia SIZE=large task bench:pipeline   # everything at 100k text embeddings
```

The published DBpedia figures come from an experiment run on a GitHub-hosted runner, all
databases in one job; from a fork with Actions enabled:

```bash
gh workflow run bench.yml -f experiment=release-0.3.0   # about 1.5 hours; the session comes back as an artifact
```

The pipeline writes `results/pipeline_<timestamp>.json`, with one section per database and
what each ran as (image, image ID, settings) under `setup`, and indexes it in `results/bench.db`
(SQLite: one row per metric per database per run). `task results:import` rebuilds that index from
the files, and `task results:runs` lists what's in it.
`task bench:insert`, `bench:query`, `bench:memory` and `bench:coldstart` run only those steps of the
pipeline (`-- --steps queries,memory` picks any), with what they need.
Add `-- --db govec --db chroma` to run only some databases. The databases are listed in
`databases.yaml`: GoVec's variants (int8, memory-mapped, brute force) are entries there, each a
few settings over `govec-config.yaml`. `--set KEY=VALUE` changes a GoVec setting for one run.

A comparison run over several rounds in one session (a GoVec release against a newer commit, say)
is a file in `experiments/`, run with `task bench:run EXPERIMENT=<name>`: each round runs every
arm through the pipeline, in an order that alternates between rounds, and the session's files go in
`results/sessions/`. `task results:compare SESSION=<name>` then gives each arm's median and range
per metric, the change, and whether every run was better or the runs overlap. In a terminal, `bench:run`
shows the session live, and `task bench:ui` browses every session with a plot of each metric's runs.

## Run-to-run variation

Latency moves between runs by tens of percent, and whole sessions shift together: on a laptop,
in an earlier set of three runs every database's single inserts were 2–9% slower than in the
published set, and Qdrant's 30% slower, with no change to any image or code path. So the tables
are medians of three rounds, every database run once per round in an order that reverses each
round, and they compare databases within the same session, on the same machine. Between
sessions, compare ratios (GoVec's latency over Qdrant's, say), not milliseconds: the runner's
processor changes from session to session, and with it every database's speed. A shared cloud
runner varies within a session too: in the published session the widest ranges are p99
latencies (GoVec's k=10 p99 3.58–8.51 ms across its three rounds); most means moved by a few
percent, the most about 20% (GoVec int8's k=10, 4.65–5.63 ms).
`task results:compare SESSION=release-0.3.0` prints each figure's range.

## Caveats

- **A shared cloud VM, its processor not fixed.** The DBpedia figures come from a GitHub-hosted
  runner, here an AMD EPYC 9V45: 4 vCPUs shared with other tenants, the benchmark client on the
  same VM as the database (each database capped at 2 CPUs). They compare the databases with each other, not with dedicated production hardware.
  Docker runs natively on Linux there. The older sections ran on a Mac through Docker Desktop,
  which makes `fsync` unusually cheap and flatters every database's write path.
- **GoVec's RAM is an upper bound.** RSS reflects when Go's garbage collector returns memory, not
  live data; the same load has measured anywhere from 115 MB (live heap) to 593 MB (RSS) depending
  on timing. Setting `GOMEMLIMIT` cuts the measured figure by about 46%, but it's left off so
  every database runs on stock settings.
- **Size changes the picture.** In October, on a Mac with GoVec 0.2.1, GoVec's query latency on
  text embeddings grew 25% from 10k to 100k vectors while the others' grew 3–5%, and recall moves
  too. Results are shown at 100k where it matters; at millions of vectors they would shift again.
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
