#!/usr/bin/env python3
"""Build a small, clean UTF-8 poetry corpus for the MoonBit miniGPT demo."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


BASE_URL = (
    "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/"
    "master/%E5%85%A8%E5%94%90%E8%AF%97"
)
DEFAULT_OUTPUT = Path("data/poems_2m.txt")
DEFAULT_MAX_BYTES = 1_900_000
SOURCE_OFFSETS = range(0, 58_000, 1_000)
EDITORIAL_NOTE = re.compile(r"[（(〔\[].*?[）)〕\]]")
SPACES = re.compile(r"[ \t\u3000]+")
BAD_CHARS = str.maketrans("", "", "\ufeff�□〓●◇◆■")


def fetch_json(url: str, retries: int = 4) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def clean_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = line.translate(BAD_CHARS)
    line = EDITORIAL_NOTE.sub("", line)
    line = SPACES.sub("", line)
    return line.strip()


def cjk_count(line: str) -> int:
    count = 0
    for char in line:
        if "\u3400" <= char <= "\u9fff":
            count += 1
    return count


def poem_text(item: dict) -> str | None:
    paragraphs = item.get("paragraphs")
    if not isinstance(paragraphs, list):
        return None
    lines: list[str] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, str):
            continue
        line = clean_line(paragraph)
        if cjk_count(line) >= 4:
            lines.append(line)
    if len(lines) < 2:
        return None
    return "\n".join(lines)


def build_corpus(max_bytes: int) -> tuple[str, int, int]:
    chunks: list[str] = []
    seen: set[str] = set()
    poems = 0
    used_files = 0
    current_bytes = 0
    for offset in SOURCE_OFFSETS:
        url = f"{BASE_URL}/poet.tang.{offset}.json"
        used_files += 1
        for item in fetch_json(url):
            poem = poem_text(item)
            if poem is None or poem in seen:
                continue
            block = poem + "\n\n"
            block_bytes = len(block.encode("utf-8"))
            if current_bytes + block_bytes > max_bytes:
                return "".join(chunks).rstrip() + "\n", poems, used_files
            seen.add(poem)
            chunks.append(block)
            poems += 1
            current_bytes += block_bytes
    return "".join(chunks).rstrip() + "\n", poems, used_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args()

    corpus, poem_count, file_count = build_corpus(args.max_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8", newline="\n")

    byte_count = args.output.stat().st_size
    print(f"wrote {args.output}")
    print(f"bytes: {byte_count}")
    print(f"poems: {poem_count}")
    print(f"source files read: {file_count}")


if __name__ == "__main__":
    main()
