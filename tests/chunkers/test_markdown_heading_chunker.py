from govec_bench.chunkers.base import Section
from govec_bench.chunkers.markdown_heading_chunker import MarkdownHeadingChunker


def test_chunk__no_h2_headers__returns_single_section_titled_by_h1() -> None:
    # Arrange
    text = "# My Doc\n\nJust some body text, no subsections.\n"

    # Act
    result = MarkdownHeadingChunker().chunk(text)

    # Assert
    assert result == [Section("My Doc", "Just some body text, no subsections.")]


def test_chunk__h2_headers__splits_into_one_section_per_header() -> None:
    # Arrange
    text = "# My Doc\n\n## First\n\nFirst body.\n\n## Second\n\nSecond body.\n"

    # Act
    result = MarkdownHeadingChunker().chunk(text)

    # Assert
    assert result == [Section("First", "First body."), Section("Second", "Second body.")]


def test_chunk__content_before_first_h2__creates_intro_section_titled_by_h1() -> None:
    # Arrange
    text = "# My Doc\n\nIntro text.\n\n## First\n\nFirst body.\n"

    # Act
    result = MarkdownHeadingChunker().chunk(text)

    # Assert
    assert result == [Section("My Doc", "Intro text."), Section("First", "First body.")]


def test_chunk__no_h1__uses_untitled_as_fallback_heading() -> None:
    # Arrange
    text = "Just body text, no headers at all.\n"

    # Act
    result = MarkdownHeadingChunker().chunk(text)

    # Assert
    assert result == [Section("Untitled", "Just body text, no headers at all.")]


def test_chunk__h3_headers_under_h2__splits_into_one_section_per_h3_prefixed_by_h2() -> None:
    # Arrange
    text = "# My Doc\n\n## Log\n\n### One\n\nBody one.\n\n### Two\n\nBody two.\n"

    # Act
    result = MarkdownHeadingChunker().chunk(text)

    # Assert
    assert result == [Section("Log: One", "Body one."), Section("Log: Two", "Body two.")]


def test_chunk__intro_before_first_h3__keeps_h2_heading_unprefixed() -> None:
    # Arrange
    text = "# My Doc\n\n## Log\n\nIntro under Log.\n\n### One\n\nBody one.\n"

    # Act
    result = MarkdownHeadingChunker().chunk(text)

    # Assert
    assert result == [Section("Log", "Intro under Log."), Section("Log: One", "Body one.")]


def test_chunk__h2_without_h3__is_left_as_a_single_section() -> None:
    # Arrange
    text = "# My Doc\n\n## Plain\n\nNo subsections here.\n\n## Log\n\n### One\n\nBody one.\n"

    # Act
    result = MarkdownHeadingChunker().chunk(text)

    # Assert
    assert result == [Section("Plain", "No subsections here."), Section("Log: One", "Body one.")]
