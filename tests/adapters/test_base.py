import re
from pathlib import Path

import pytest

from govec_bench.adapters.base import DEFAULT_SEARCH_EF

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("config", sorted(REPO_ROOT.glob("govec-config*.yaml")), ids=lambda path: path.name)
def test_govec_config__hnsw_ef_search__matches_the_other_databases(config: Path) -> None:
    # Arrange
    text = config.read_text()

    # Act
    match = re.search(r"^\s*hnsw_ef_search:\s*(\d+)", text, re.MULTILINE)

    # Assert
    assert match is not None
    assert int(match.group(1)) == DEFAULT_SEARCH_EF
