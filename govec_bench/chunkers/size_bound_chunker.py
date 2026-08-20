from typing import override

from govec_bench.chunkers.base import Chunker, Section

DEFAULT_MAX_CHARS = 800


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _pack_paragraphs(text: str, max_chars: int) -> list[str]:
    paragraphs = text.split("\n\n")
    windows: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)

        if current and current_len + 2 + paragraph_len > max_chars:
            windows.append("\n\n".join(current))
            current = []
            current_len = 0

        if paragraph_len > max_chars:
            if current:
                windows.append("\n\n".join(current))
                current = []
                current_len = 0
            windows.extend(_hard_split(paragraph, max_chars))
            continue

        current.append(paragraph)
        current_len += (2 if current_len else 0) + paragraph_len

    if current:
        windows.append("\n\n".join(current))

    return windows


class SizeBoundChunker(Chunker, name="size-bound"):
    def __init__(self, inner: Chunker, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self._inner = inner
        self._max_chars = max_chars

    @override
    def chunk(self, text: str) -> list[Section]:
        sections = []

        for section in self._inner.chunk(text):
            sections.extend(self._split_section(section))

        return sections

    def _split_section(self, section: Section) -> list[Section]:
        if len(section.body) <= self._max_chars:
            return [section]

        pieces = _pack_paragraphs(section.body, self._max_chars)
        total = len(pieces)

        return [Section(f"{section.heading} ({i + 1}/{total})", piece) for i, piece in enumerate(pieces)]
