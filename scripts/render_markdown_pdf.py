"""Render a simple text-focused Markdown file to PDF without dependencies."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT = 54
TOP = 742
BOTTOM = 54
BODY_SIZE = 10
HEADING_SIZE = 15
SUBHEADING_SIZE = 12
LINE_HEIGHT = 14
MONO_LINE_HEIGHT = 12
BODY_WRAP = 92
CODE_WRAP = 86


def render_markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> None:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    pages: list[list[tuple[str, int, str]]] = [[]]
    y = TOP
    in_code = False

    for raw_line in lines:
        if raw_line.strip().startswith("```"):
            in_code = not in_code
            continue

        chunks = _line_chunks(raw_line, in_code)
        if not chunks:
            chunks = [("", BODY_SIZE, "body")]

        for text, size, font in chunks:
            line_height = MONO_LINE_HEIGHT if font == "mono" else LINE_HEIGHT
            if y - line_height < BOTTOM:
                pages.append([])
                y = TOP
            pages[-1].append((text, size, font))
            y -= line_height

    _write_pdf(pdf_path, pages)


def _line_chunks(raw_line: str, in_code: bool) -> list[tuple[str, int, str]]:
    if in_code:
        return [
            (chunk, BODY_SIZE, "mono")
            for chunk in textwrap.wrap(raw_line, width=CODE_WRAP, replace_whitespace=False)
        ]

    stripped = raw_line.strip()
    if not stripped:
        return []

    if stripped.startswith("# "):
        return [(stripped[2:], HEADING_SIZE, "bold")]
    if stripped.startswith("## "):
        return [(stripped[3:], SUBHEADING_SIZE, "bold")]
    if stripped.startswith("### "):
        return [(stripped[4:], BODY_SIZE + 1, "bold")]

    normalized = stripped.replace("`", "")
    return [
        (chunk, BODY_SIZE, "body")
        for chunk in textwrap.wrap(normalized, width=BODY_WRAP, replace_whitespace=False)
    ]


def _write_pdf(pdf_path: Path, pages: list[list[tuple[str, int, str]]]) -> None:
    objects: list[bytes] = []
    font_regular_id = 3
    font_bold_id = 4
    font_mono_id = 5
    page_ids = []
    content_ids = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    next_id = 6
    for page in pages:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        content_ids.append(content_id)

        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R "
                f"/F3 {font_mono_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(_content_stream(page))

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _write_objects(pdf_path, objects)


def _content_stream(page: list[tuple[str, int, str]]) -> bytes:
    parts = ["BT"]
    y = TOP
    for text, size, font in page:
        font_name = {"body": "F1", "bold": "F2", "mono": "F3"}[font]
        escaped = _escape_pdf_text(text)
        parts.append(f"/{font_name} {size} Tf")
        parts.append(f"1 0 0 1 {LEFT} {y} Tm")
        parts.append(f"({escaped}) Tj")
        y -= MONO_LINE_HEIGHT if font == "mono" else LINE_HEIGHT
    parts.append("ET")
    stream = "\n".join(parts).encode("latin-1", errors="replace")
    return b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"


def _escape_pdf_text(text: str) -> str:
    return (
        text.encode("latin-1", errors="replace")
        .decode("latin-1")
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _write_objects(pdf_path: Path, objects: list[bytes]) -> None:
    offsets = [0]
    with pdf_path.open("wb") as handle:
        handle.write(b"%PDF-1.4\n")
        for index, obj in enumerate(objects, start=1):
            offsets.append(handle.tell())
            handle.write(f"{index} 0 obj\n".encode("ascii"))
            handle.write(obj)
            handle.write(b"\nendobj\n")

        xref_offset = handle.tell()
        handle.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        handle.write(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            handle.write(f"{offset:010d} 00000 n \n".encode("ascii"))

        handle.write(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    render_markdown_to_pdf(args.markdown, args.pdf)


if __name__ == "__main__":
    main()
