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
| `task data:download` | Fetch SIFT10K and SIFT1M (SIFT100K is its first 100k vectors) |
| `task docker:up` / `task docker:down` | Start / stop all databases |
| `task bench:all` | Run all five benchmarks; each writes `results/<benchmark>_<timestamp>.json` |
| `task bench:insert` (`query`, `recall`, `memory`, `coldstart`) | Run one benchmark |
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
    registry.py        build_adapters(): the databases every benchmark iterates
  benchmarks/          one script per benchmark; common.py has dataset loading and docker helpers
  datasets/sift.py     fvecs/ivecs readers; SIFT10K and SIFT100K
  datasets/groundtruth.py  exact cosine neighbours by brute force: the recall answer key
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
- **Stock settings unless the comparison needs otherwise.** HNSW parameters are matched where
  they're comparable (`M=16`, cosine). Anything tuned for one database only (e.g. `GOMEMLIMIT`
  for GoVec) stays off, with a comment saying why.
- **Same client behaviour for every database.** Each adapter reuses one connection. Per-request
  connection setup puts a TCP handshake inside every measured latency.
- **Medians across repeated runs for latency.** Single runs on a laptop swing by tens of
  percent. Compare databases within a run, where they share the same machine conditions.

## Adding a database

1. An adapter in `adapters/`, implementing `VectorDBAdapter`, registered in `registry.py`.
2. A service in `docker-compose.yml`: pinned image, the same caps, a health check if possible.
3. Its container name and data paths in `benchmarks/memory.py` (`DBS`) and its service in
   `benchmarks/cold_start.py`.
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
- **Keep the SDK and the server image in step.** The govec adapter uses the published SDK. When
  you bump the `ghcr.io/pradyothsp/govec` pin, bump `govec>=…` in `pyproject.toml` with it: a
  new server behind an old pinned SDK once crashed the query benchmark mid-run.

## Before you finish

1. `task fmt && task fmt:check`.
2. `task test`.
3. If you changed an adapter or a benchmark, run it against `task docker:up` and sanity-check
   the numbers against the previous results.
