import re
from typing import override

from govec_bench.chunkers.base import Chunker, Section

_H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_H2_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_H3_RE = re.compile(r"^### (.+)$", re.MULTILINE)


def _split_at_headers(text: str, header_re: re.Pattern[str], fallback_heading: str) -> list[Section]:
    boundaries = [(m.start(), m.group(1).strip()) for m in header_re.finditer(text)]
    if not boundaries:
        body = text.strip()
        return [Section(fallback_heading, body)] if body else []

    sections: list[Section] = []

    intro = text[: boundaries[0][0]].strip()
    if intro:
        sections.append(Section(fallback_heading, intro))

    for i, (start, heading) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = header_re.sub("", text[start:end], count=1).strip()
        sections.append(Section(heading, body))

    return sections


class MarkdownHeadingChunker(Chunker, name="markdown-heading"):
    @override
    def chunk(self, text: str) -> list[Section]:
        title_match = _H1_RE.search(text)
        title = title_match.group(1).strip() if title_match else "Untitled"
        content_start = title_match.end() if title_match else 0

        h2_sections = _split_at_headers(text[content_start:], _H2_RE, title)

        sections: list[Section] = []
        for h2 in h2_sections:
            if not _H3_RE.search(h2.body):
                sections.append(h2)
                continue

            for h3 in _split_at_headers(h2.body, _H3_RE, h2.heading):
                heading = h2.heading if h3.heading == h2.heading else f"{h2.heading}: {h3.heading}"
                sections.append(Section(heading, h3.body))

        return sections
