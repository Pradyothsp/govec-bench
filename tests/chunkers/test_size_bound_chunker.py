from typing import override

from govec_bench.chunkers.base import Chunker, Section
from govec_bench.chunkers.size_bound_chunker import SizeBoundChunker


class _FakeChunker(Chunker, skip=True):
    @override
    def chunk(self, text: str) -> list[Section]:
        return [Section("Only", text)]


def test_chunk__section_within_limit__returned_unchanged() -> None:
    # Arrange
    chunker = SizeBoundChunker(_FakeChunker(), max_chars=100)

    # Act
    result = chunker.chunk("short body")

    # Assert
    assert result == [Section("Only", "short body")]


def test_chunk__section_over_limit__splits_by_paragraph_and_numbers_headings() -> None:
    # Arrange
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunker = SizeBoundChunker(_FakeChunker(), max_chars=20)

    # Act
    result = chunker.chunk(text)

    # Assert
    assert result == [
        Section("Only (1/3)", "First paragraph."),
        Section("Only (2/3)", "Second paragraph."),
        Section("Only (3/3)", "Third paragraph."),
    ]


def test_chunk__paragraphs_fit_together__packs_multiple_into_one_window() -> None:
    # Arrange
    text = "One.\n\nTwo.\n\nThree.\n\nFour is next."
    chunker = SizeBoundChunker(_FakeChunker(), max_chars=20)

    # Act
    result = chunker.chunk(text)

    # Assert
    assert result == [
        Section("Only (1/2)", "One.\n\nTwo.\n\nThree."),
        Section("Only (2/2)", "Four is next."),
    ]


def test_chunk__single_paragraph_over_limit__hard_splits_it() -> None:
    # Arrange
    chunker = SizeBoundChunker(_FakeChunker(), max_chars=10)

    # Act
    result = chunker.chunk("abcdefghijklmnopqrstuvwxyz")

    # Assert
    assert result == [
        Section("Only (1/3)", "abcdefghij"),
        Section("Only (2/3)", "klmnopqrst"),
        Section("Only (3/3)", "uvwxyz"),
    ]
