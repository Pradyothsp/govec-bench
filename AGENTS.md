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
| `task data:export` | Write DBpedia 10k (`SIZE=large`: 100k) as fvecs/ivecs with its exact top-100 to `data/export/<name>/`, for GoVec's in-repo Go benchmarks (`GOVEC_BENCH_DATA=<that dir>`); `DATASET=sift` for SIFT |
| `task docker:up` / `task docker:down` | Start / stop all databases, for manual poking; benchmarks start their own |
| `task bench:pipeline` (also `bench`, `bench:all`) | Every measurement from one load per database: batch insert, disk and RAM, recall and query latency (GoVec and Qdrant also over gRPC, same index; Chroma has no gRPC client), restarts with data and empty, single inserts on 10k. Writes `results/pipeline_<timestamp>.json`. `SIZE=large` for 100k, `DATASET=dbpedia` for text embeddings, `-- --db govec --db qdrant` for only those databases |
| `task bench:pipeline -- --steps memory,restart-loaded` | Only those steps, and what they need (the load); a stage with none doesn't start its container. The file lists the steps that ran under `steps`. Steps: `batch-insert`, `memory`, `queries`, `queries-grpc`, `restart-loaded`, `restart-empty`, `single-insert`. An experiment takes `steps:` the same way |
| `task bench:insert` (`query`/`recall`, `memory`, `coldstart`) | The pipeline with the steps each standalone benchmark used to be: inserts; recall and latency from the same searches; disk and RAM at 100k; restarts |
| `task bench:sweep` | Recall@k against query latency per search ef (`-- --ef 50 --ef 100 --k 10` to narrow); GoVec, Chroma and Qdrant unless `--db` says otherwise (`--db govec-scalar` adds int8); not part of the pipeline: it changes Qdrant's ef on one loaded graph, which a step can't |
| `task bench:insert -- --db govec --set hnsw_ef_construction=200` | Any benchmark with a GoVec setting changed for one run: a key of `govec-config.yaml` as its `GOVEC_` variable without the prefix; repeat `--set` for several. Here `ef_construction=200`, GoVec's default up to 0.2.0 |
| `task bench:pipeline -- --db govec-mmap --govec HEAD` | Run GoVec from a commit of the GoVec repo (`../govec`, or `GOVEC_REPO`) instead of the pin: built once from a clean copy, tagged `govec-bench/govec:<commit>`. Any image works too (`--govec govec:local`, `--govec ghcr.io/pradyothsp/govec:0.2.1`). For development comparisons; published numbers use the pin |
| `task bench:run EXPERIMENT=mmap-0.2.1-vs-head` | Run an experiment from `experiments/`: arms (a database, and for GoVec an image or commit and settings) run through the whole pipeline in rounds, the order alternating or shuffled between rounds. It starts Docker, builds images and the workload before round 1, keeps the Mac awake and waits for Docker's disk between 100k runs. One session's results files, `session.json`, `events.jsonl` and `run.log` go in `results/sessions/<experiment>/<started>/`. In a terminal it opens the live view and runs the session in a process of its own: `q` leaves the run going, `s` twice stops it as Ctrl-C would. `-- --plain` prints instead; `-- --baseline published` adds a baseline to the live table; `-- --dry-run` shows the plan and each arm's commit; `-- --rounds 1` overrides the file |
| `task bench:ui` | The newest session still running, live; with none running, browse sessions: the comparison `results:compare` prints, a baseline picker, a dot per run for the highlighted metric (`h`: its median in every session of the experiment), Enter for the metric's runs, `a` for every metric. `-- --live <experiment>` opens that experiment's newest session live instead |
| `task results:compare SESSION=mmap-0.2.1-vs-head` | What a session found: per metric, each arm's median and range, the change against the control (round 1's first arm, or `--control`) and a verdict that says better only when every run beat every control run (`level` under 1%). Headline metrics by default; `--metric 'queries_grpc.*'` or `--all` for more. `--baseline <session>` or `--baseline published` adds each arm against that, marking arms that ran the same image and settings in both, since their change is the sessions' drift. SESSION is an experiment (its latest session), `experiment/session` or a session folder |
| `task results:import` | Index results files into `results/bench.db` (each benchmark also indexes its own file as it writes it); `-- ../govec-notes/bench` adds older sessions. Safe to repeat: a file already indexed, from anywhere, is skipped |
| `task results:runs` | List indexed runs, newest first, with what each database ran as; `-- --benchmark pipeline --database govec-mmap` to narrow |
| `task test` | Unit tests |
| `task fmt` / `task fmt:check` | Format and autofix / ruff + ty + format check |
| `task docker:clean` | Remove this project's containers, images and volumes |

## Code map

```text
govec_bench/
  adapters/
    base.py            VectorDBAdapter: insert, batch_insert, query, stats, reset
    govec_adapter.py   via the govec SDK from PyPI, REST transport (gRPC for the pipeline's gRPC step)
    chroma_adapter.py  via chromadb's HTTP client, cosine space
    qdrant_adapter.py  via qdrant-client, cosine distance, keep-alive forced on (gRPC for the pipeline's gRPC step)
    registry.py        ENGINES: one per compose service (adapter, disk paths, gRPC client); DATABASES,
                       read from databases.yaml: each database's engine and settings, whether it needs
                       the dimension at startup, whether it's opt-in (govec-mmap, govec-brute)
  benchmarks/          common.py has arguments (--db, --govec, --set), dataset loading, docker helpers
                       and running_alone(), which runs one database's container at a time with its
                       settings and image layered on, and checks they reached it
                       pipeline.py is every measurement from one load: PIPELINE is a list of stages of
                       named steps, --steps picks some; insert.py, memory.py and search.py measure
                       for its steps (search.py also for ef_sweep.py, the one benchmark of its own)
                       images.py turns --govec into an image, building a GoVec commit when given one
                       experiment.py runs an experiment file's arms in rounds (bench run)
  datasets/
    base.py            VectorDataset; ArrayItems keeps vectors in one array, builds items on read
    sift.py            fvecs/ivecs readers; SIFT10K and SIFT100K (128-dim image descriptors)
    dbpedia.py         DBpedia 10k/100k: OpenAI ada-002 text embeddings, 1536-dim, from Parquet
    registry.py        --dataset name -> (10k loader, 100k loader, dimensions); Size and load_sized()
    groundtruth.py     exact cosine neighbours by brute force: the recall answer key
  results.py           latency/recall statistics and the JSON results writer, which also indexes each file
  results_file.py      a results file's shape, as TypedDicts: what each database ran as, the run's tag
  compare.py           bench compare: a session's arms per metric, from bench.db; HEADLINE is its rows
  events.py            events.jsonl: what a running session is doing (round, arm, step, warnings)
  tui/                 the terminal app (Textual): browse.py over bench.db, live.py over a session's
                       folder (events, session.json, run.log), so it never talks to the runner
  store.py             results/bench.db: every results file as rows (runs, what each database ran as,
                       one row per metric under a dotted name); an index over the JSON, rebuildable
  cli.py               the bench commands (python -m govec_bench.cli): import, runs, run, compare, ui
docker-compose.yml     one service per engine, pinned images, 2 CPU / 2 GB each
databases.yaml         every database by name: its engine and the settings it changes (GOVEC_ variables)
experiments/           comparisons as files, for bench run: arms, rounds, order, dataset and size
govec-config.yaml      GoVec's config; each GoVec database changes it through environment variables
```

## Rules for a fair comparison

- **Pinned, published images for published numbers.** `docker-compose.yml` pins every database
  to a released image (Chroma by digest, since it publishes no tag for 1.4.4). Published results
  must be reproducible by anyone, and comparable between runs: never publish a local build or
  `:latest`. Bump a pin deliberately and re-run everything. `--govec` runs another GoVec image or
  a build of a commit, for development comparisons (before and after a change); every results
  file records each database's image, image ID, GoVec commit and settings under `setup`, so such a
  result can't pass for the pin.
- **Identical resource caps.** Every service gets 2 CPUs and 2 GB.
- **One database at a time.** Every benchmark starts only the container it measures, on fresh
  volumes, and removes it afterwards (`running_alone()`). Idle neighbours still compact,
  optimize and collect garbage, and four 2-CPU containers claim every core of an 8-core laptop.
- **Watch a session the same way in every session you compare.** The live view runs on the Mac
  beside the benchmark client: a redraw a second, and `docker stats` every 10 s. Its effect on
  latency hasn't been measured, so compare live-view sessions with live-view sessions (or run
  both `--plain`), and measure it before mixing them.
- **Stock settings unless the comparison needs otherwise.** HNSW parameters are matched where
  they're comparable (`M=16`, cosine). Anything tuned for one database only (e.g. `GOMEMLIMIT`
  for GoVec) stays off, with a comment saying why.
- **Same client behaviour for every database.** Each adapter reuses one connection. Per-request
  connection setup puts a TCP handshake inside every measured latency.
- **Medians across repeated runs for latency.** Single runs on a laptop swing by tens of
  percent. Compare databases within a run, where they share the same machine conditions.

## Adding a database

**Another setup of an engine** (GoVec with a setting changed): an entry in `databases.yaml`, with
the settings as environment variables. Nothing else: it runs as the engine's service, and
`running_alone()` checks the settings reached the container. For one run only, use `--set`.

**A new engine:**

1. An adapter in `adapters/`, implementing `VectorDBAdapter`.
2. A service in `docker-compose.yml`: pinned image, the same caps, a health check if possible.
3. An entry in `ENGINES` in `adapters/registry.py`, keyed by that service's name: its adapter and
   the data paths the memory benchmark measures.
4. An entry in `databases.yaml` naming the engine. Every benchmark picks it up from there.
5. Run the full suite, and check its numbers are plausible before trusting them.

## Measurement traps (found the hard way)

- **qdrant-client disables keep-alive for `localhost`.** Every request opened a new connection:
  latencies included a handshake, and 10k rapid single inserts exhausted Docker Desktop's port
  forwarding. The adapter passes `limits=httpx.Limits()` to restore keep-alive.
- **RSS is not memory use for Go.** `docker stats` reflects when Go's GC returns memory, not live
  data; the same GoVec load measured 593 MB, 259 MB and 115 MB (pprof live heap) depending on
  timing. Treat GoVec RAM numbers as an upper bound.
- **Cold start means are fragile.** Five restarts per run; one slow restart moves the mean by
  80%. Report the median.
- **Time a restart from before `compose start`.** The server boots while that command runs;
  timing from after it measured little more than the probe (9 ms "restarts"; Chroma's probe
  makes several calls).
- **Qdrant indexes after its upserts return.** Insert timings wait for indexing once, at the end
  of the load (`wait_until_indexed()`). Polling after every batch put the 50 ms poll interval
  inside each batch and made Qdrant look 5x slower than it is; skipping the wait on single
  inserts flattered it instead. Its lowered `indexing_threshold` stays: on stock settings it
  never indexes segments under 10 MB (all of SIFT 10k, 15% of DBpedia 10k).
- **Docker Desktop makes `fsync` misleadingly fast** on macOS, which flatters every database's
  write path. Numbers are comparable to each other, not to bare metal.
- **Chroma caps batch sizes**, and Qdrant accepts only `u64` or UUID point IDs (the adapter
  derives a stable UUID from each string ID).
- **Grade recall against the metric the databases search with.** Every database here runs
  cosine, but SIFT's shipped ground truth is Euclidean. The two disagree on near-ties, so the
  shipped file marked correct cosine answers wrong, capped even a perfect search at 98.0%
  recall@1, and understated every database by up to 2 points. `search.py` grades against exact
  cosine neighbours by brute force (`datasets/groundtruth.py`). If you change a metric, change
  the key with it.
- **A GoVec snapshot overrides the configured `ef_search` (up to 0.2.0; fixed after).** The snapshot stores the graph's
  ef_search and loading it overwrites `GOVEC_HNSW_EF_SEARCH`/`hnsw_ef_search`, so restarting a
  loaded container with a new ef searches with the old one. `ef_sweep.py` starts a fresh
  container per ef instead (`running_alone()` checks the env reached it).
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
- **Qdrant needs tens of GB of temporary disk at 100k.** While it optimizes a 100k load of
  1536-d vectors it has written about 30 GB, though it settles under 1 GB. 100k runs (memory
  benchmark, `--size large`) check first for 35 GB free, inside Docker's disk and, on Docker
  Desktop, on the Mac's disk it grows into, and stop with the reason otherwise.
- **The SDK's gRPC stubs need `grpcio>=1.84`,** but the SDK only declares `>=1.60`, so an older
  `grpcio` in `uv.lock` passes install and crashes the first gRPC call on import. Keep the lock at
  1.84 or later.
- **Keep the SDK and the server image in step.** The govec adapter uses the published SDK. When
  you bump the `ghcr.io/pradyothsp/govec` pin, bump `govec>=…` in `pyproject.toml` with it: a
  new server behind an old pinned SDK once crashed the query benchmark mid-run.

## Code style

- **No bare dicts for records.** Anything with a known set of fields is a type: a
  `@dataclass(slots=True)` for state the code builds up (frozen if nothing changes it after,
  as in `experiment.Machine`), a `NamedTuple` for small immutable values (`registry.Database`),
  a `TypedDict` for JSON written or read as it is (`results_file.ResultsFile`, `StartedAs`). A
  dict is for a real mapping, keyed by data: databases by name, settings by `GOVEC_` variable,
  metrics by dotted name. A benchmark's measurements stay open JSON (`dict[str, object]`) only
  where the steps that ran decide the keys, as in the pipeline's per-database results.

## Before you finish

1. `task fmt && task fmt:check`.
2. `task test`.
3. If you changed an adapter or a benchmark, run it and sanity-check the numbers against the
   previous results.
