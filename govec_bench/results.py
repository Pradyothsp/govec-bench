import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path("results")


@dataclass(frozen=True, slots=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p99_ms: float


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    return LatencyStats(
        mean_ms=statistics.mean(latencies_ms),
        p50_ms=statistics.median(latencies_ms),
        p99_ms=statistics.quantiles(latencies_ms, n=100)[98],
    )


def write_results(benchmark: str, dataset: str, results: dict[str, object]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    payload = {
        "benchmark": benchmark,
        "timestamp": timestamp.isoformat(),
        "dataset": dataset,
        "results": results,
    }

    path = RESULTS_DIR / f"{benchmark}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, indent=2))

    return path
