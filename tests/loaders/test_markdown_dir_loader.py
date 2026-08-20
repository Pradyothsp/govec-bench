from pathlib import Path

from govec_bench.loaders.base import RawDocument
from govec_bench.loaders.markdown_dir_loader import MarkdownDirLoader


def test_load__docs_directory__returns_one_raw_document_per_file_keyed_by_relative_path(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "a.md").write_text("# A\n\nBody.\n")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.md").write_text("# B\n\nBody.\n")

    # Act
    result = MarkdownDirLoader(tmp_path).load()

    # Assert
    assert result == [
        RawDocument(source="a", text="# A\n\nBody.\n"),
        RawDocument(source="sub_b", text="# B\n\nBody.\n"),
    ]
