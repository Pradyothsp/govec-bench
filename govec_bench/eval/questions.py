"""The RAG eval question set.

Mined from STATUS.md's 25 numbered findings (`§1`-`§25`) and the 12
single-topic docs under `govec/docs/architecture/` (+ `QUANTIZATION_METRICS.md`),
per RAG_V2_PLAN.md §3. `expected_source`/`expected_heading` match the values
`MarkdownDirLoader`/`MarkdownHeadingChunker` actually produce for each chunk,
so recall@k can be scored by exact (source, heading-prefix) containment.

Two negative controls (carried over from `benchmarks/rag.py`'s original
question set) are appended at the end -- they re-test the two known §25
regressions: the `METADATA_INDEX.md` miss (via `arch-metadata-index` above)
and the multi-tenancy negation blind spot (`neg-multi-tenancy` below).
"""

from govec_bench.eval.store import EvalQuestion

_STATUS_SOURCE = "govec-bench_STATUS"

_WHAT_WE_LEARNED = "What We Learned"
_CURRENT_RESULTS = "Current Results: Tuned HNSW govec vs Chroma (SIFT10K, 2026-08-16, fully dockerized)"
_LAST_FINDING_UNDER_WHAT_WE_LEARNED = 11


def _status_heading(number: int, title: str) -> str:
    # §1-11 sit under "## What We Learned"; §12-25 sit under the later
    # "## Current Results: ..." section instead -- both are "###" within their
    # own "##" parent, per MarkdownHeadingChunker's h2-then-h3 recursion.
    parent = _WHAT_WE_LEARNED if number <= _LAST_FINDING_UNDER_WHAT_WE_LEARNED else _CURRENT_RESULTS
    return f"{parent}: {number}. {title}"


QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        question_id="status-1",
        question="What three bugs were found in govec's HNSW search implementation?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(1, "govec's HNSW implementation had three real bugs"),
        gold_answer=(
            "heap.Max()/PopLast() assumed the wrong array slot held the maximum; the 'improved' check "
            "compared against the best result instead of the worst kept; and the termination condition "
            "diverged from the canonical SEARCH-LAYER algorithm."
        ),
    ),
    EvalQuestion(
        question_id="status-2",
        question="Why did govec add Euclidean distance support, and what recall did it achieve on SIFT10K?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(2, "Added Euclidean distance to govec"),
        gold_answer=(
            "SIFT10K's official ground truth is computed with Euclidean distance, but govec only supported "
            "cosine; after adding euclidean end-to-end, brute-force + euclidean hit exactly 100% recall at "
            "every k."
        ),
    ),
    EvalQuestion(
        question_id="status-3",
        question="Why was auto_save_enabled disabled during govec's benchmark runs?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(3, "Benchmark fairness is a config problem as much as a code problem"),
        gold_answer=(
            "govec's periodic snapshot (SaveToFile) holds the full write lock for the entire GOB-encode + "
            "disk write; a save landing mid-benchmark blocked in-flight queries and caused p99 spikes, so it "
            "was disabled as pure infrastructure noise."
        ),
    ),
    EvalQuestion(
        question_id="status-4",
        question="What batch size limit does Chroma enforce, and how does govec-bench work around it?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(4, "Chroma enforces a max batch size"),
        gold_answer=(
            "Chroma's server-side max_batch_size is 5461, and govec has no such limit; "
            "common.py::load_dataset() now chunks all benchmark inserts (default 100/batch) so both adapters "
            "load the same way."
        ),
    ),
    EvalQuestion(
        question_id="status-5",
        question="What causes Chroma's ~25-40ms p99 latency outlier during benchmarks?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(5, "Open question: Chroma's own p99 jitter"),
        gold_answer=(
            "Not fully root-caused -- it's independent of govec's state (still occurs with govec's auto-save "
            "disabled). Leading theories are Python-client GC pauses or Chroma's own internal background "
            "persistence; not investigated further."
        ),
    ),
    EvalQuestion(
        question_id="status-6",
        question="Why is govec's batch insert ~11x slower per-vector than Chroma's, and why wasn't it fixed immediately?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(6, "govec's batch-insert gap is a durability trade-off, not a bug"),
        gold_answer=(
            "govec calls fsync via WAL.WriteEntry() once per vector even inside a batch, making every insert "
            "individually crash-durable; fixing it would trade that per-vector durability guarantee for "
            "whole-batch durability, a real contract change left as a deliberate trade-off rather than fixed "
            "right away."
        ),
    ),
    EvalQuestion(
        question_id="status-7",
        question="Why did govec's official Docker image crash-loop with a WAL 'permission denied' error?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(7, "govec's shipped Dockerfile couldn't actually persist data"),
        gold_answer=(
            "The runtime stage switches to USER nobody:nobody, but /app was left root:root-owned from the "
            "build stage, so the image could never write its WAL; fixed with `chown -R nobody:nobody /app` "
            "before the USER switch."
        ),
    ),
    EvalQuestion(
        question_id="status-8",
        question="Why did govec's insert latency improve ~5x after dockerizing on macOS, and what does that imply?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            8,
            "Docker Desktop's virtual disk makes `fsync` misleadingly fast — a real gotcha for anyone "
            "benchmarking via Docker on macOS",
        ),
        gold_answer=(
            "Docker Desktop on Apple Silicon runs containers in a VM via Apple's Virtualization.framework, "
            "whose virtual-disk sync modes can let a guest's fsync succeed without the data reaching the "
            "host's physical disk -- so insert-latency numbers measured through Docker Desktop on macOS are "
            "not representative of real bare-metal durability/performance."
        ),
    ),
    EvalQuestion(
        question_id="status-9",
        question="What does it cost govec in insert latency to match Chroma's HNSW recall on SIFT10K?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(9, "Matching Chroma's recall costs govec far more on inserts than expected"),
        gold_answer=(
            "Matching Chroma's ~99% recall@10 required raising hnsw_m to 96, six times Chroma's implicit "
            "default; that erased govec's single-insert advantage entirely and widened the batch-insert gap "
            "to ~29x."
        ),
    ),
    EvalQuestion(
        question_id="status-10",
        question="Why did govec need such a high M value to match Chroma's recall, before the ef_construction fix?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            10, "Root cause: govec's HNSW has no distinct `ef_construction` — it reuses `ef_search` for graph building too"
        ),
        gold_answer=(
            "govec reused a single EfSearch field for both query-time search breadth and graph-construction "
            "breadth, so a cheap ef_search (needed for fast queries) also meant a low-quality, cheaply-built "
            "graph; adding a distinct ef_construction let it build a good graph while keeping ef_search cheap."
        ),
    ),
    EvalQuestion(
        question_id="status-11",
        question="Can raising govec's HNSW M parameter alone close the recall@1 gap to Chroma?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            11,
            "`M` sweep: raising `M` alone can't close the gap to Chroma — evidence *for* heuristic selection, "
            "not an alternative to it",
        ),
        gold_answer=(
            "No -- sweeping M up to 128 got recall@10 to 99.0%, but recall@1 still trailed Chroma's 97.0% at "
            "91.0%, while single-insert cost nearly quadrupled; this is the empirical case for implementing "
            "SELECT-NEIGHBORS-HEURISTIC instead of just raising M."
        ),
    ),
    EvalQuestion(
        question_id="status-12",
        question="How is 'queryable' defined for govec-bench's cold-start benchmark, and how much faster is govec than Chroma?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(12, "Cold start benchmark: built and measured"),
        gold_answer=(
            "'Queryable' means the adapter's own reset() call succeeds; govec's static Go binary came up ~5x "
            "faster at the mean (22.89ms vs 114.51ms) and ~8x faster at p50 than Chroma's Python-based "
            "service."
        ),
    ),
    EvalQuestion(
        question_id="status-13",
        question="How are RAM and disk usage measured identically across govec and Chroma in the memory benchmark?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            13, "Memory benchmark: govec uses ~12x more RAM than Chroma for the same 100k vectors — disk is fine, RAM is not"
        ),
        gold_answer=(
            "RAM via `docker stats`, disk via `docker exec ... du -sb`, identically for both databases with "
            "no per-database code path; govec's on-disk snapshot needs an explicit flush() call first since "
            "auto_save_enabled is off."
        ),
    ),
    EvalQuestion(
        question_id="status-14",
        question="What specific line of code accounts for 74% of govec's live heap, per the pprof heap profile?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            14, "Confirmed with a real heap profile: 74% of govec's live heap is map-bucket overhead in `addNeighbor`, one line"
        ),
        gold_answer=(
            "hnsw.(*layerNode).addNeighbor's map assignment (n.neighbors[newNode.Key] = newNode) -- storing "
            "up to M pointers in a Go map carries far more overhead than hnswlib's flat fixed-capacity array."
        ),
    ),
    EvalQuestion(
        question_id="status-15",
        question="Why is a single RSS snapshot unreliable for comparing govec's memory usage to Chroma's?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            15,
            "Corrected RAM comparison: multi-interval sampling shows govec is *still* descending at 5 minutes "
            "— the trustworthy number is §14's pprof figure, not any RSS snapshot",
        ),
        gold_answer=(
            "govec's RSS is still descending at the 5-minute mark post-load due to Go's GC/scavenger lag, so "
            "a single-instant measurement gives a materially different ratio depending on when you sample; "
            "pprof's inuse_space (365MB) is the trustworthy, non-moving-target figure."
        ),
    ),
    EvalQuestion(
        question_id="status-16",
        question="Does Chroma get an unfair docker-specific advantage in govec-bench's memory comparison?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            16, "Checked whether Chroma has a docker-specific advantage in the memory comparison — no evidence of one, if anything the reverse"
        ),
        gold_answer=(
            "No -- resource caps are identical, docker stats' page-cache exclusion applies fairly to both, "
            "and if anything the asymmetry runs the other way since govec's image is hand-optimized (107MB) "
            "while Chroma runs its stock 822MB image."
        ),
    ),
    EvalQuestion(
        question_id="status-17",
        question="What was the real-world impact of replacing govec's map-based neighbor storage with a slice?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            17, "Implemented the `map[K]*layerNode[K,V]` → `[]*layerNode[K,V]` fix (§14) — real, verified, but a smaller win than predicted"
        ),
        gold_answer=(
            "Live heap dropped 47.7% (365MB to 191MB), single-insert got 31% faster and batch-insert 43% "
            "faster, with query and recall unaffected -- a clean win with zero regressions."
        ),
    ),
    EvalQuestion(
        question_id="status-18",
        question="Why was the improvement from batching govec's WAL fsync smaller than the bare-host math predicted?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            18, "Batched the WAL fsync for `BatchInsert` — a real win, but much smaller than the bare-host math predicted"
        ),
        gold_answer=(
            "The bare-host ~87% fsync-cost estimate predated containerizing; Docker Desktop's virtual disk "
            "already made fsync artificially cheap, so by the time this fix ran in Docker, fsync was a "
            "smaller share of the per-vector cost, leaving real HNSW graph-construction cost as the dominant "
            "remainder."
        ),
    ),
    EvalQuestion(
        question_id="status-19",
        question="How much of govec's batch-insert gap to Chroma is explained by CPU parallelism?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            19, "Measured how much of the batch-insert gap is Chroma using more CPU — real, but modest (~21%, not most of it)"
        ),
        gold_answer=(
            "Only about 1.04x of the 6.33x ratio -- capping both containers to 1 physical CPU showed Chroma "
            "got only ~21% slower, not the ~2x a naive doubling-of-threads assumption would predict; the "
            "remaining ~5.29x is a real architecture/language gap."
        ),
    ),
    EvalQuestion(
        question_id="status-20",
        question="What three bugs had to be fixed before SELECT-NEIGHBORS-HEURISTIC actually improved govec's recall?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            20, "Implemented `SELECT-NEIGHBORS-HEURISTIC` — closed the recall gap, then recovered every real cost it introduced"
        ),
        gold_answer=(
            "replenish() picking replacements by 'first found' instead of 'best found'; the base layer "
            "needing 2xM capacity instead of M (matching hnswlib's maxM0_); and query-time search() exploring "
            "at breadth k instead of max(k, ef_search), the dominant bug."
        ),
    ),
    EvalQuestion(
        question_id="status-21",
        question="What did the GOMEMLIMIT experiment reveal about govec's apparent 5.5x RAM gap to Chroma?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(21, "RAM wasn't the gap it looked like — and recall@50's \"reversal\" isn't one either"),
        gold_answer=(
            "Setting GOMEMLIMIT=250MiB cut measured RAM by 46% (592.8MB to 319.2MB) with no latency cost, "
            "showing most of the gap was Go's GC/allocator timing rather than real storage cost -- though it "
            "wasn't adopted as a default since the value is dataset-size-specific."
        ),
    ),
    EvalQuestion(
        question_id="status-22",
        question="Does govec-bench use Pinecone for its vector database comparisons?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(22, "Added a Qdrant adapter, replacing the planned Pinecone one"),
        gold_answer=(
            "No -- Pinecone was replaced by Qdrant, since Pinecone is cloud-only with no Docker/self-hosted "
            "option and can't be fairly compared against govec/Chroma's fully-local, resource-capped "
            "containers."
        ),
    ),
    EvalQuestion(
        question_id="status-23",
        question="Why did scalar quantization recall collapse to near-random on SIFT10K?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            23, "Scalar quantization had a silent [-1,1] clamp bug — recall collapsed to near-random, now a real ~43%/~58% RAM/disk win"
        ),
        gold_answer=(
            "QuantizeVector assumed input vectors were normalized to [-1,1] but nothing enforced it; "
            "SIFT10K's vectors range up to 119, so almost every dimension clamped to 127, destroying the "
            "vector's structure before it reached the graph."
        ),
    ),
    EvalQuestion(
        question_id="status-24",
        question="What bug surfaced the first time Qdrant was run at 100k-vector scale in govec-bench?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            24, "First same-run, four-way, 2-CPU comparison — all five benchmarks, govec/govec-scalar/Chroma/Qdrant together"
        ),
        gold_answer=(
            "memory.py's 100k-vector load failed with 'Qdrant indexing did not catch up within 30.0s' -- "
            "indexing catch-up cost grows with collection size, so _INDEXING_POLL_TIMEOUT_S was raised from "
            "30s to 120s to match memory.py's existing DISK_STABILIZE_TIMEOUT_S."
        ),
    ),
    EvalQuestion(
        question_id="status-25",
        question="What real limitation of all-MiniLM-L6-v2 did growing the RAG question set surface?",
        expected_source=_STATUS_SOURCE,
        expected_heading=_status_heading(
            25, "Built the RAG retrieval track — real doc corpus, real DBs, and a known embedding-model limitation found by growing the question set"
        ),
        gold_answer=(
            "A vocabulary-mismatch ceiling and a negation blind spot -- METADATA_INDEX.md's own answer chunk "
            "scores below a generic, irrelevant chunk on the metadata-filtering question, and the one correct "
            "'no, this isn't supported' chunk for the negative-control questions scores far below an "
            "unrelated chunk that happens to share surface vocabulary."
        ),
    ),
    EvalQuestion(
        question_id="arch-batch-insert",
        question="What is the root cause of govec's per-vector batch-insert overhead compared to Chroma?",
        expected_source="govec_docs_architecture_BATCH_INSERT_PERFORMANCE",
        expected_heading="2. Root Cause Analysis: A. Per-Vector WAL Disk Write Overhead",
        gold_answer=(
            "govec calls fsync (file.Sync()) once per vector via WAL.WriteEntry(), even inside a batch "
            "request, making each insert individually crash-durable at a steep per-vector I/O cost, unlike "
            "Chroma's single batch commit."
        ),
    ),
    EvalQuestion(
        question_id="arch-distance-metrics",
        question="Why does govec support both cosine and Euclidean distance metrics?",
        expected_source="govec_docs_architecture_DISTANCE_METRICS",
        expected_heading="Why This Matters: SIFT Benchmark Evidence",
        gold_answer=(
            "SIFT10K's official ground truth is computed with Euclidean distance, so a fair recall benchmark "
            "against it requires euclidean support; cosine remains the default for typical normalized-"
            "embedding use cases."
        ),
    ),
    EvalQuestion(
        question_id="arch-concurrent-insert",
        question="Why is govec's HNSW insert path fully serial, unlike hnswlib's?",
        expected_source="govec_docs_architecture_HNSW_CONCURRENT_INSERT",
        expected_heading="The Gap",
        gold_answer=(
            "govec's internal/hnsw has no internal concurrency at all -- inserts run strictly one item at a "
            "time -- while hnswlib (used by Chroma) parallelizes insertion across threads."
        ),
    ),
    EvalQuestion(
        question_id="arch-ef-construction",
        question="What problem did separating ef_construction from ef_search solve in govec's HNSW?",
        expected_source="govec_docs_architecture_HNSW_EF_CONSTRUCTION",
        expected_heading="The Problem",
        gold_answer=(
            "govec reused a single EfSearch field for both query-time and graph-construction candidate-list "
            "breadth, so a cheap ef_search (for fast queries) also meant a low-quality, cheaply-built graph; "
            "splitting them lets construction be thorough while search stays cheap."
        ),
    ),
    EvalQuestion(
        question_id="arch-memory-layout",
        question="What memory-layout change remains unaddressed after fixing govec's neighbor storage from a map to a slice?",
        expected_source="govec_docs_architecture_HNSW_MEMORY_LAYOUT",
        expected_heading="Remaining gap: `layer.nodes` and `metadata` are still hash maps, even though the IDs are already dense",
        gold_answer=(
            "layer.nodes and the metadata store are still Go hash maps even though node IDs are already "
            "dense integers, meaning they could be flattened into dense ID-indexed arrays for a further "
            "memory win, the same class of fix already shipped for addNeighbor."
        ),
    ),
    EvalQuestion(
        question_id="arch-neighbor-selection",
        question="What neighbor-selection algorithm does govec's HNSW actually use, and how does it differ from hnswlib's?",
        expected_source="govec_docs_architecture_HNSW_NEIGHBOR_SELECTION",
        expected_heading="The Gap",
        gold_answer=(
            "govec implements SELECT-NEIGHBORS-SIMPLE, not SELECT-NEIGHBORS-HEURISTIC -- a simpler selection "
            "rule that can leave neighbors landing in already-well-covered regions instead of new, useful "
            "parts of the graph, unlike the heuristic hnswlib (and thus Chroma) uses."
        ),
    ),
    EvalQuestion(
        question_id="arch-search-recall-bugs",
        question="What was Bug 2 in govec's HNSW search, involving the 'improved' check?",
        expected_source="govec_docs_architecture_HNSW_SEARCH_RECALL_BUGS",
        expected_heading='Bug 2: Search "Improved" Check Compared Against the Wrong End of the Result Set',
        gold_answer=(
            "The search loop's 'did this help' check compared a candidate against the best result found so "
            "far instead of the worst currently kept, causing the search to give up almost immediately."
        ),
    ),
    EvalQuestion(
        question_id="arch-id-mapping",
        question="What is the IDMapper and why does govec need it?",
        expected_source="govec_docs_architecture_ID_MAPPING",
        expected_heading="Why This Change?",
        gold_answer=(
            "The IDMapper translates arbitrary user-supplied string IDs to internal uint32 keys, since "
            "govec's HNSW graph and storage layers are keyed on compact integers rather than arbitrary "
            "strings."
        ),
    ),
    EvalQuestion(
        question_id="arch-memory-tuning",
        question="What is GOMEMLIMIT used for in this project, and why isn't it a default?",
        expected_source="govec_docs_architecture_MEMORY_TUNING",
        expected_heading="The `GOMEMLIMIT` Experiment",
        gold_answer=(
            "GOMEMLIMIT is a Go 1.19+ soft memory cap that cut govec's measured RSS by 46% with no latency "
            "cost in testing, but it's a fixed cap tuned to one dataset size, so it isn't defaulted since it "
            "would be wrong (too tight or too loose) for other data sizes."
        ),
    ),
    EvalQuestion(
        question_id="arch-metadata-index",
        question="How do I enable metadata filtering in govec, and what's the default?",
        expected_source="govec_docs_architecture_METADATA_INDEX",
        expected_heading="How to enable",
        gold_answer="Set enable_metadata_index: true in config; it defaults to false.",
    ),
    EvalQuestion(
        question_id="arch-mmap",
        question="How does govec avoid holding the full vector dataset in RAM?",
        expected_source="govec_docs_architecture_MMAP_VECTOR_STORAGE",
        expected_heading="How It Works",
        gold_answer=(
            "govec supports mmap-backed vector storage, memory-mapping vectors from disk instead of loading "
            "them all into the process's own heap, enabled via a config flag."
        ),
    ),
    EvalQuestion(
        question_id="arch-quantization-metrics",
        question="What performance metrics does govec-bench's quantization benchmark suite measure?",
        expected_source="govec_docs_QUANTIZATION_METRICS",
        expected_heading="Available Metrics: 1. **Performance Metrics**",
        gold_answer="Throughput and latency percentiles, among other performance metrics, for quantized vs. unquantized vector operations.",
    ),
    # Negative controls, carried over from benchmarks/rag.py -- re-test §25's two known failures.
    EvalQuestion(
        question_id="neg-multi-tenancy",
        question="How do I configure multi-tenant namespaces in govec?",
        expected_source="govec_AGENTS",
        expected_heading="Scope & Non-Goals",
        gold_answer=(
            "Multi-tenancy is explicitly out of scope per govec's AGENTS.md -- there is no multi-tenant "
            "namespace configuration."
        ),
    ),
    EvalQuestion(
        question_id="neg-kubernetes",
        question="What Helm chart does govec-bench use to deploy to Kubernetes?",
        expected_source="NONE",
        expected_heading=None,
        gold_answer=(
            "None -- govec-bench doesn't use Kubernetes or Helm anywhere; this is a true negative control "
            "with no correct source in the corpus."
        ),
    ),
]
