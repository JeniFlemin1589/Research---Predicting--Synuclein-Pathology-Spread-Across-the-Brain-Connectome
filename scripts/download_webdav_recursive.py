"""Recursively download public HTML-indexed WebDAV folders."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse

import requests


class LinkParser(HTMLParser):
    """Collect href values from simple directory listings."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def parse_links(html: str) -> list[str]:
    """Extract all href links from an HTML page."""
    parser = LinkParser()
    parser.feed(html)
    return parser.hrefs


def iter_directory(
    session: requests.Session,
    root_url: str,
    current_url: str,
) -> tuple[list[str], list[str]]:
    """Return directory and file URLs under a given directory page."""
    response = session.get(current_url, timeout=60)
    response.raise_for_status()
    directories: list[str] = []
    files: list[str] = []
    root_prefix = root_url.rstrip("/") + "/"
    for href in parse_links(response.text):
        if href.startswith("?"):
            continue
        absolute = urljoin(current_url, href)
        if not absolute.startswith(root_prefix):
            continue
        normalized = absolute.rstrip("/")
        current_normalized = current_url.rstrip("/")
        if normalized == current_normalized:
            continue
        if absolute.endswith("/"):
            directories.append(absolute)
        else:
            files.append(absolute)
    return sorted(set(directories)), sorted(set(files))


def relative_path_from_url(root_url: str, file_url: str) -> Path:
    """Map a file URL into a relative local path."""
    root_path = PurePosixPath(urlparse(root_url).path.rstrip("/"))
    file_path = PurePosixPath(urlparse(file_url).path)
    relative = file_path.relative_to(root_path)
    return Path(unquote(relative.as_posix()))


def download_file(
    session: requests.Session,
    root_url: str,
    file_url: str,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    """Download one file to disk."""
    relative_path = relative_path_from_url(root_url, file_url)
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"SKIP {relative_path}")
        return destination
    print(f"GET  {relative_path}")
    with session.get(file_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def walk_and_download(
    session: requests.Session,
    root_url: str,
    current_url: str,
    output_dir: Path,
    overwrite: bool,
    visited: set[str],
) -> None:
    """Recursively walk an HTML index and download files."""
    normalized = current_url.rstrip("/") + "/"
    if normalized in visited:
        return
    visited.add(normalized)
    directories, files = iter_directory(session, root_url, normalized)
    for file_url in files:
        download_file(session, root_url, file_url, output_dir, overwrite)
    for directory_url in directories:
        walk_and_download(session, root_url, directory_url, output_dir, overwrite, visited)


def main(argv: Iterable[str] | None = None) -> None:
    """Parse arguments and download a full WebDAV tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Root public WebDAV URL ending in a directory.")
    parser.add_argument("--output-dir", required=True, help="Local output directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist locally.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root_url = args.url.rstrip("/") + "/"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        walk_and_download(
            session=session,
            root_url=root_url,
            current_url=root_url,
            output_dir=output_dir,
            overwrite=args.overwrite,
            visited=set(),
        )


if __name__ == "__main__":
    main()
