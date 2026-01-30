#!/usr/bin/env python3
"""
Shared utility functions for SVT-AV1 optimization research scripts.

This module contains common functionality used across multiple scripts
to avoid code duplication and maintain consistency.
"""

import hashlib
from pathlib import Path


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate SHA256 checksum of a file.

    Reads file in chunks to handle large video files efficiently without
    loading entire file into memory.

    Args:
        file_path: Path to the file to checksum

    Returns:
        Hexadecimal string representation of SHA256 hash

    Example:
        >>> from pathlib import Path
        >>> checksum = calculate_sha256(Path("video.mp4"))
        >>> print(f"SHA256: {checksum}")
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    result: str = sha256_hash.hexdigest()
    return result
