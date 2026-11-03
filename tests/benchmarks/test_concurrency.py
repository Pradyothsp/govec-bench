from collections.abc import Callable

import pytest

from govec_bench.benchmarks.concurrency import (
    P999_MIN_SAMPLES,
    WINDOW_S,
    ClientResult,
    build_point,
    p999,
    parse_cpu_usage_us,
)


@pytest.fixture
def client_result() -> Callable[..., ClientResult]:
    def _build(latencies_ms: list[float], recall: float = 1.0, cpu_s: float = 0.0, errors: int = 0) -> ClientResult:
        return ClientResult(latencies_ms, [recall] * len(latencies_ms), cpu_s, errors)

    return _build


def test_parse_cpu_usage_us__cgroup_v2_cpu_stat__usage_in_microseconds() -> None:
    # Arrange
    cpu_stat = "usage_usec 81234567\nuser_usec 60000000\nsystem_usec 21234567\nnr_periods 10\n"

    # Act
    usage = parse_cpu_usage_us(cpu_stat)

    # Assert
    assert usage == 81_234_567


def test_parse_cpu_usage_us__no_usage_line__raises_value_error() -> None:
    # Arrange
    cpu_stat = "user_usec 1\nsystem_usec 2\n"

    # Act / Assert
    with pytest.raises(ValueError, match="usage_usec"):
        parse_cpu_usage_us(cpu_stat)


def test_p999__fewer_samples_than_a_thousand__the_slowest() -> None:
    # Arrange
    latencies = [1.0] * (P999_MIN_SAMPLES - 2) + [9.0]

    # Act
    value = p999(latencies)

    # Assert
    assert value == 9.0


def test_p999__enough_samples__ignores_the_single_slowest() -> None:
    # Arrange
    latencies = [1.0] * 9_999 + [500.0]

    # Act
    value = p999(latencies)

    # Assert
    assert value == 1.0


def test_build_point__two_clients__throughput_recall_and_cpu_over_the_window(
    client_result: Callable[..., ClientResult],
) -> None:
    # Arrange
    clients = [
        client_result([2.0] * 300, recall=1.0, cpu_s=6.0, errors=1),
        client_result([4.0] * 100, recall=0.5, cpu_s=4.0),
    ]

    # Act
    point = build_point(2, clients, server_cpu_s=30.0)

    # Assert
    assert point.concurrency == 2
    assert point.queries == 400
    assert point.throughput_qps == pytest.approx(400 / WINDOW_S)
    assert point.recall_mean == pytest.approx((300 * 1.0 + 100 * 0.5) / 400)
    assert point.server_cpu_cores == pytest.approx(30.0 / WINDOW_S)
    assert point.client_cpu_cores == pytest.approx(10.0 / WINDOW_S)
    assert point.errors == 1


def test_build_point__no_query_finished__raises_runtime_error(client_result: Callable[..., ClientResult]) -> None:
    # Arrange
    clients = [client_result([]), client_result([])]

    # Act / Assert
    with pytest.raises(RuntimeError, match="no query finished"):
        build_point(2, clients, server_cpu_s=0.0)
