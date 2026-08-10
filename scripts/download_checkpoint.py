#!/usr/bin/env python3
"""Download and verify the published fine-tuned ESM-2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


DEFAULT_URL = (
    "https://github.com/yangmei25/esm2-protein-localization/releases/download/"
    "model-v1.0/best_checkpoint.pt"
)
DEFAULT_OUTPUT = Path("results/finetune/esm2_t6_8M_mean/best_checkpoint.pt")
EXPECTED_SHA256 = "698fca86ff8b973a026529b645b5d4536da2a6c30c96d797da16bc047a158fbd"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path, expected_sha256: str) -> None:
    """Download to a partial file, verify integrity, then move into place."""
    if output.exists():
        actual = file_sha256(output)
        if actual == expected_sha256:
            print(f"Checkpoint already exists and is verified: {output}")
            return
        raise ValueError(
            f"Existing checkpoint has SHA-256 {actual}, expected {expected_sha256}. "
            "Move or delete it before downloading."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    try:
        with urlopen(url) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        actual = file_sha256(partial)
        if actual != expected_sha256:
            raise ValueError(
                f"Downloaded checkpoint has SHA-256 {actual}, expected {expected_sha256}"
            )
        partial.replace(output)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"Downloaded and verified checkpoint: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sha256", default=EXPECTED_SHA256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    download(args.url, args.output, args.sha256.lower())


if __name__ == "__main__":
    main()
