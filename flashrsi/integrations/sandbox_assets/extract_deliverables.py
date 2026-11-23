"""Extract selected workspace deliverables inside the offline verifier.

Only selected files within the workspace are read. Format-specific libraries
are loaded on demand, and the returned text has a per-file character limit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lxml.etree import _Element

DOC_EXTENSIONS = frozenset({".docx", ".xlsx", ".pdf", ".md", ".txt", ".csv"})
MAX_CHARS = 100_000


def _element_text(element: _Element) -> str:
    return "".join(
        node.text or "" for node in element.iter() if node.tag.endswith("}t")
    )


def _docx_text(path: Path) -> tuple[str, list[str]]:
    from docx import Document

    document = Document(path)
    lines = []
    for element in document.element.body:
        if element.tag.endswith("}p"):
            paragraph = _element_text(element)
            if paragraph:
                lines.append(paragraph)
        elif element.tag.endswith("}tbl"):
            for row in element:
                if row.tag.endswith("}tr"):
                    cells = [
                        _element_text(cell) for cell in row if cell.tag.endswith("}tc")
                    ]
                    lines.append(" | ".join(cells))
    headings = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.style and paragraph.style.name.startswith("Heading")
    ]
    return "\n".join(lines), headings


def _xlsx_text(path: Path) -> tuple[str, list[str]]:
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=False)
    try:
        lines = []
        for sheet in book:
            if (sheet.max_row or 1) * (sheet.max_column or 1) > 100_000:
                raise ValueError("table too large")
            lines.append(f"## {sheet.title}")
            for row in sheet.values:
                lines.append(
                    " | ".join("" if cell is None else str(cell) for cell in row)
                )
        return "\n".join(lines), book.sheetnames
    finally:
        book.close()


def _pdf_text(path: Path) -> tuple[str, list[str]]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        content = "\n".join(page.extract_text() or "" for page in pdf.pages[:100])
        return content, [f"{len(pdf.pages)} pages"]


def text(path: Path) -> tuple[str, list[str]]:
    readers = {".docx": _docx_text, ".xlsx": _xlsx_text, ".pdf": _pdf_text}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        return path.read_text(encoding="utf-8"), []
    return reader(path)


def extract(base: Path, names: list[str]) -> list[dict[str, Any]]:
    base = base.resolve()
    if not isinstance(names, list) or len(set(names)) != len(names):
        raise ValueError("expected distinct artifact paths")
    files = []
    for name in names:
        relative = Path(name)
        path = base / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.resolve().is_relative_to(base)
            or not path.is_file()
            or path.suffix.lower() not in DOC_EXTENSIONS
            or any(
                p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(base)
            )
        ):
            raise ValueError("invalid artifact path")
        metadata = {"path": name, "bytes": path.stat().st_size}
        try:
            body, sections = text(path)
        except (ImportError, MemoryError):
            # Missing runtime dependencies and memory pressure are infrastructure failures.
            raise
        except Exception as exc:
            files.append({**metadata, "error": type(exc).__name__})
        else:
            files.append(
                {
                    **metadata,
                    "text": body[:MAX_CHARS],
                    "truncated": len(body) > MAX_CHARS,
                    "sections": sections,
                }
            )
    return files


if __name__ == "__main__":
    print(
        json.dumps(
            extract(Path("/workspace"), json.loads(sys.argv[1])), ensure_ascii=False
        )
    )
