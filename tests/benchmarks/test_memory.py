import pytest

from govec_bench.benchmarks.memory import RamBreakdown, parse_memory_stat


def test_parse_memory_stat__cgroup_v2__process_is_anon_and_cache_is_file() -> None:
    # Arrange
    text = "anon 1000\nfile 400\nfile_mapped 300\nactive_file 100\n"

    # Act
    breakdown = parse_memory_stat(text)

    # Assert
    assert breakdown == RamBreakdown(process_bytes=1000, file_cache_bytes=400)


def test_parse_memory_stat__cgroup_v1__process_is_rss_and_cache_is_cache() -> None:
    # Arrange
    text = "cache 400\nrss 1000\nmapped_file 300\n"

    # Act
    breakdown = parse_memory_stat(text)

    # Assert
    assert breakdown == RamBreakdown(process_bytes=1000, file_cache_bytes=400)


def test_parse_memory_stat__neither_layout__raises() -> None:
    # Arrange
    text = "pgfault 5\n"

    # Act
    with pytest.raises(ValueError, match=r"memory\.stat has none of") as exc_info:
        parse_memory_stat(text)

    # Assert
    assert "anon" in str(exc_info.value)
