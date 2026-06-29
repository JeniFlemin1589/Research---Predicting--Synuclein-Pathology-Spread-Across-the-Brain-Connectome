"""Export a Markdown report to a simple text-based PDF without third-party deps."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def markdown_to_lines(markdown: str, width: int = 92) -> list[str]:
    """Convert markdown into wrapped plain-text lines."""
    lines: list[str] = []
    in_code = False
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            lines.append("")
            continue

        if in_code:
            lines.extend(textwrap.wrap(line, width=width) or [""])
            continue

        if not line.strip():
            lines.append("")
            continue

        if line.startswith("#"):
            text = line.lstrip("#").strip()
            lines.append(text.upper())
            lines.append("")
            continue

        if line.startswith("- "):
            wrapped = textwrap.wrap("- " + line[2:], width=width, subsequent_indent="  ")
            lines.extend(wrapped or ["-"])
            continue

        if line[:2].isdigit() and line[1:3] == ". ":
            wrapped = textwrap.wrap(line, width=width, subsequent_indent="   ")
            lines.extend(wrapped or [line])
            continue

        wrapped = textwrap.wrap(line, width=width)
        lines.extend(wrapped or [""])
    return lines


def pdf_escape(text: str) -> str:
    """Escape text for a PDF string literal."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_page_stream(lines: list[str]) -> bytes:
    """Create a PDF text content stream for one page."""
    commands = ["BT", "/F1 11 Tf", "50 780 Td", "14 TL"]
    for idx, line in enumerate(lines):
        if idx > 0:
            commands.append("T*")
        commands.append(f"({pdf_escape(line)}) Tj")
    commands.append("ET")
    content = "\n".join(commands).encode("latin-1", errors="replace")
    return content


def write_simple_pdf(lines: list[str], output_path: Path, lines_per_page: int = 48) -> None:
    """Write a minimal multi-page PDF."""
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]
    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    catalog_id = add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(b"<< /Type /Pages /Kids [] /Count 0 >>")
    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    content_ids: list[int] = []
    for page_lines in pages:
        stream = build_page_stream(page_lines)
        content_id = add_object(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )
        content_ids.append(content_id)
        page_id = add_object(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii")

    pdf_parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    current_offset = len(pdf_parts[0])
    for idx, obj in enumerate(objects, start=1):
        offsets.append(current_offset)
        chunk = f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
        pdf_parts.append(chunk)
        current_offset += len(chunk)

    xref_offset = current_offset
    xref_lines = [b"xref\n", f"0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for offset in offsets[1:]:
        xref_lines.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    trailer = (
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n".encode("ascii")
        + b"startxref\n"
        + f"{xref_offset}\n".encode("ascii")
        + b"%%EOF\n"
    )
    pdf_parts.extend(xref_lines)
    pdf_parts.append(trailer)
    output_path.write_bytes(b"".join(pdf_parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    markdown = source.read_text(encoding="utf-8")
    lines = markdown_to_lines(markdown)
    write_simple_pdf(lines, output)


if __name__ == "__main__":
    main()
