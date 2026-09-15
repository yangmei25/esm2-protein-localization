#!/usr/bin/env python3
"""Prepare an optional S3 checkpoint and start the FastAPI service."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def download_checkpoint_if_needed() -> Path:
    checkpoint = Path(os.getenv("MODEL_CHECKPOINT", "/app/model/best_checkpoint.pt"))
    if checkpoint.exists():
        return checkpoint
    uri = os.getenv("MODEL_S3_URI")
    if not uri:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint}; set MODEL_S3_URI=s3://bucket/key"
        )
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("MODEL_S3_URI must have the form s3://bucket/key")
    import boto3

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(
        parsed.netloc, parsed.path.lstrip("/"), str(checkpoint)
    )
    return checkpoint


def main() -> None:
    checkpoint = download_checkpoint_if_needed()
    os.environ["MODEL_CHECKPOINT"] = str(checkpoint)
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
