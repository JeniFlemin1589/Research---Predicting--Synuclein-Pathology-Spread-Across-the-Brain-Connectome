"""Export a simple Markdown report to styled HTML."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
ORDERED_RE = re.compile(r"^\d+\.\s+(.*)$")


def format_inline(text: str) -> str:
    """Apply a small subset of Markdown inline formatting."""
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def markdown_to_html(markdown: str) -> str:
    """Convert a minimal Markdown subset into HTML."""
    lines = markdown.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    in_ul = False
    in_ol = False
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            parts.append(f"<p>{format_inline(' '.join(paragraph).strip())}</p>")
            paragraph = []

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            parts.append("</ul>")
            in_ul = False
        if in_ol:
            parts.append("</ol>")
            in_ol = False

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("```"):
            flush_paragraph()
            close_lists()
            if in_code:
                code_html = html.escape("\n".join(code_lines))
                parts.append(f"<pre><code>{code_html}</code></pre>")
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(stripped)
            continue

        if not stripped:
            flush_paragraph()
            close_lists()
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph()
            close_lists()
            level = len(heading_match.group(1))
            text = format_inline(heading_match.group(2))
            parts.append(f"<h{level}>{text}</h{level}>")
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            if not in_ul:
                parts.append("<ul>")
                in_ul = True
            parts.append(f"<li>{format_inline(stripped[2:])}</li>")
            continue

        ordered_match = ORDERED_RE.match(stripped)
        if ordered_match:
            flush_paragraph()
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if not in_ol:
                parts.append("<ol>")
                in_ol = True
            parts.append(f"<li>{format_inline(ordered_match.group(1))}</li>")
            continue

        paragraph.append(stripped)

    flush_paragraph()
    close_lists()
    if in_code:
        code_html = html.escape("\n".join(code_lines))
        parts.append(f"<pre><code>{code_html}</code></pre>")

    body = "\n".join(parts)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Professor Progress Report</title>
  <style>
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      margin: 40px auto;
      max-width: 900px;
      line-height: 1.55;
      color: #1f2937;
      padding: 0 24px 40px;
    }}
    h1, h2, h3 {{
      color: #0f172a;
      page-break-after: avoid;
    }}
    h1 {{
      border-bottom: 2px solid #cbd5e1;
      padding-bottom: 8px;
    }}
    p, li {{
      font-size: 14px;
    }}
    code {{
      background: #f1f5f9;
      padding: 1px 4px;
      border-radius: 4px;
      font-family: Consolas, monospace;
      font-size: 12px;
    }}
    pre {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
    }}
    ul, ol {{
      padding-left: 24px;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    markdown = source.read_text(encoding="utf-8")
    output.write_text(markdown_to_html(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
