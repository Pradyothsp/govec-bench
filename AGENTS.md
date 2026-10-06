# AGENTS.md

Guidance for AI agents (Claude, Gemini) and contributors working in this repo: how the harness is
organized, the rules that keep its numbers honest, and the measurement traps already found.
Results and how to reproduce them are in `README.md`.

## What this is

A benchmark harness comparing [GoVec](https://github.com/Pradyothsp/govec) against Chroma and
Qdrant (and GoVec's own int8 mode) on insert latency, query latency, recall, memory and cold
start. Every database runs in Docker under identical resource caps, driven from Python through a
common adapter interface.

**The numbers are the product.** A change that makes GoVec look better without making the
comparison fairer is a bug. When a result looks surprising, find out why before publishing it;
several "GoVec wins" here have turned out to be measurement errors.

## Commands

| Command | Use |
|---|---|
| `uv sync --all-groups` | Set up the environment |
| `task data:download` | Fetch SIFT10K and SIFT1M (SIFT100K is its first 100k vectors); `DATASET=dbpedia` for the text embeddings |
| `task docker:up` / `task docker:down` | Start / stop all databases, for manual poking; benchmarks start their own |
| `task bench:all` | Run all five benchmarks; each writes `results/<benchmark>_<timestamp>.json`. `DATASET=dbpedia` switches dataset |
| `task bench:insert` (`query`, `recall`, `memory`, `coldstart`) | Run one benchmark; `-- --db govec --db qdrant` runs only those databases |
| `task bench:pipeline` | Every benchmark from one load per database: batch insert, disk and RAM, recall and query latency, restarts with data and empty, single inserts on 10k. `SIZE=large` for 100k |
| `task bench:sweep` | Recall@k against query latency per search ef (`-- --ef 50 --ef 100 --k 10` to narrow); GoVec, Chroma and Qdrant unless `--db` says otherwise (`--db govec-scalar` adds int8); not part of `bench:all` |
| `COMPOSE_FILE=docker-compose.yml:compose.govec-efc200.yaml task bench:insert -- --db govec` | Any benchmark with GoVec built at `ef_construction=200` (its default up to 0.2.0) instead of 100 |
| `task test` | Unit tests |
| `task fmt` / `task fmt:check` | Format and autofix / ruff + ty + format check |
| `task docker:clean` | Remove this project's containers, images and volumes |

## Code map

```text
govec_bench/
  adapters/
    base.py            VectorDBAdapter: insert, batch_insert, query, stats, reset
    govec_adapter.py   via the govec SDK from PyPI, REST transport
    chroma_adapter.py  via chromadb's HTTP client, cosine space
    qdrant_adapter.py  via qdrant-client, cosine distance, keep-alive forced on
    registry.py        DATABASES: the one list of databases, by compose service: adapter, disk paths,
                       whether it needs the dimension at startup, whether it's opt-in (govec-mmap)
  benchmarks/          one script per benchmark; common.py has arguments, dataset loading, docker helpers
                       pipeline.py runs them all from one load: PIPELINE is a list of stages of steps
                       and running_alone(), which runs one database's container at a time
  datasets/
    base.py            VectorDataset; ArrayItems keeps vectors in one array, builds items on read
    sift.py            fvecs/ivecs readers; SIFT10K and SIFT100K (128-dim image descriptors)
    dbpedia.py         DBpedia 10k/100k: OpenAI ada-002 text embeddings, 1536-dim, from Parquet
    registry.py        --dataset name -> (10k loader, 100k loader, dimensions); Size and load_sized()
    groundtruth.py     exact cosine neighbours by brute force: the recall answer key
  results.py           latency/recall statistics and the JSON results writer
docker-compose.yml     the databases, pinned images, 2 CPU / 2 GB each
govec-config*.yaml     GoVec's config for the float32 and int8 services
```

## Rules for a fair comparison

- **Pinned, published images only.** `docker-compose.yml` pins every database to a released
  image (Chroma by digest, since it publishes no tag for 1.4.4). Never benchmark a local build
  or `:latest`: results must be reproducible by anyone, and comparable between runs. Bump a pin
  deliberately and re-run everything.
- **Identical resource caps.** Every service gets 2 CPUs and 2 GB.
- **One database at a time.** Every benchmark starts only the container it measures, on fresh
  volumes, and removes it afterwards (`running_alone()`). Idle neighbours still compact,
  optimize and collect garbage, and four 2-CPU containers claim every core of an 8-core laptop.
- **Stock settings unless the comparison needs otherwise.** HNSW parameters are matched where
  they're comparable (`M=16`, cosine). Anything tuned for one database only (e.g. `GOMEMLIMIT`
  for GoVec) stays off, with a comment saying why.
- **Same client behaviour for every database.** Each adapter reuses one connection. Per-request
  connection setup puts a TCP handshake inside every measured latency.
- **Medians across repeated runs for latency.** Single runs on a laptop swing by tens of
  percent. Compare databases within a run, where they share the same machine conditions.

## Adding a database

1. An adapter in `adapters/`, implementing `VectorDBAdapter`.
2. A service in `docker-compose.yml`: pinned image, the same caps, a health check if possible.
3. An entry in `adapters/registry.py` (`DATABASES`), keyed by that service's name: its adapter
   and the data paths the memory benchmark measures. Every benchmark picks it up from there.
4. Run the full suite, and check its numbers are plausible before trusting them.

## Measurement traps (found the hard way)

- **qdrant-client disables keep-alive for `localhost`.** Every request opened a new connection:
  latencies included a handshake, and 10k rapid single inserts exhausted Docker Desktop's port
  forwarding. The adapter passes `limits=httpx.Limits()` to restore keep-alive.
- **RSS is not memory use for Go.** `docker stats` reflects when Go's GC returns memory, not live
  data; the same GoVec load measured 593 MB, 259 MB and 115 MB (pprof live heap) depending on
  timing. Treat GoVec RAM numbers as an upper bound.
- **Cold start means are fragile.** Five restarts per run; one slow restart moves the mean by
  80%. Report the median.
- **Docker Desktop makes `fsync` misleadingly fast** on macOS, which flatters every database's
  write path. Numbers are comparable to each other, not to bare metal.
- **Chroma caps batch sizes**, and Qdrant accepts only `u64` or UUID point IDs (the adapter
  derives a stable UUID from each string ID).
- **Grade recall against the metric the databases search with.** Every database here runs
  cosine, but SIFT's shipped ground truth is Euclidean. The two disagree on near-ties, so the
  shipped file marked correct cosine answers wrong, capped even a perfect search at 98.0%
  recall@1, and understated every database by up to 2 points. `recall.py` now computes exact
  cosine neighbours by brute force (`datasets/groundtruth.py`). If you change a metric, change
  the key with it.
- **A GoVec snapshot overrides the configured `ef_search` (up to 0.2.0; fixed after).** The snapshot stores the graph's
  ef_search and loading it overwrites `GOVEC_HNSW_EF_SEARCH`/`hnsw_ef_search`, so restarting a
  loaded container with a new ef searches with the old one. `ef_sweep.py` starts a fresh
  container per ef instead, and checks the env reached it.
- **Chroma ignores a changed `ef_search` on an existing collection.** On 1.4.4,
  `collection.modify(configuration={"hnsw": {"ef_search": …}})` succeeds and the server reports
  the new value, but queries keep the ef the collection was created with. Set it at creation
  (`hnsw:search_ef`); `ef_sweep.py` uses a fresh collection per ef.
- **Qdrant sometimes misses one query entirely.** In about 1 SIFT build in 11, Qdrant's graph
  search at its default ef returns none of one query's true neighbours, the same on every repeat;
  the vectors are stored, and exact search or ef 400 finds them. Seen at DBpedia 100k too. Real
  Qdrant behaviour, not the harness: report worst-query recall and medians across builds, and
  don't "fix" it by retrying.
- **Apparent disk size overstates sparse files.** GoVec's mmap chunks are created at 1 GiB and
  filled as vectors arrive, so `du -b` reports 1 GiB at 10k vectors. The memory benchmark records
  allocated blocks (`du -k`) too; use those.
- **`docker stats` counts file cache only while it's active.** Memory-mapped data (GoVec mmap,
  likely Qdrant) lives in the file cache, not the process. The benchmarks record the cgroup's
  process/file-cache breakdown next to the `docker stats` figure, so a "smaller" number can be
  told apart from memory that moved.
- **Keep the SDK and the server image in step.** The govec adapter uses the published SDK. When
  you bump the `ghcr.io/pradyothsp/govec` pin, bump `govec>=…` in `pyproject.toml` with it: a
  new server behind an old pinned SDK once crashed the query benchmark mid-run.

## Before you finish

1. `task fmt && task fmt:check`.
2. `task test`.
3. If you changed an adapter or a benchmark, run it and sanity-check the numbers against the
   previous results.
