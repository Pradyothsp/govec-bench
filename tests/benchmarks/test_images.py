import pytest

from govec_bench.benchmarks.images import built_tag, is_image, resolve_govec_image


@pytest.mark.parametrize("ref", ["ghcr.io/pradyothsp/govec:0.2.1", "govec:local", "govec-bench/govec:abc123"])
def test_is_image__ref_with_a_colon__is_an_image(ref: str) -> None:
    # Arrange

    # Act
    image = is_image(ref)

    # Assert
    assert image


@pytest.mark.parametrize("ref", ["HEAD", "v0.2.1", "fix/mmap-hint", "26481bc"])
def test_is_image__git_ref__is_not(ref: str) -> None:
    # Arrange

    # Act
    image = is_image(ref)

    # Assert
    assert not image


def test_resolve_govec_image__image__used_as_it_is_without_a_commit() -> None:
    # Arrange
    ref = "ghcr.io/pradyothsp/govec:0.2.1"

    # Act
    resolved = resolve_govec_image(ref)

    # Assert
    assert resolved.image == ref
    assert resolved.commit is None


def test_built_tag__commit__names_the_commit() -> None:
    # Arrange
    commit = "450295f0a1b2"

    # Act
    tag = built_tag(commit)

    # Assert
    assert tag == "govec-bench/govec:450295f0a1b2"
