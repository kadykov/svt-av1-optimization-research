#!/usr/bin/env python3
"""
Shared utility functions for SVT-AV1 optimization research scripts.

This module contains common functionality used across multiple scripts
to avoid code duplication and maintain consistency.
"""

import hashlib
import json
import subprocess
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


def get_video_bitrate(video_path: Path) -> float | None:
    """
    Get video stream bitrate in kbps using FFprobe.

    Returns the actual video bitrate, excluding audio and container overhead.
    This is more accurate than calculating from file size, which includes
    container metadata and any audio streams.

    Args:
        video_path: Path to the video file

    Returns:
        Video bitrate in kbps, or None if unable to determine

    Example:
        >>> from pathlib import Path
        >>> bitrate = get_video_bitrate(Path("video.mp4"))
        >>> print(f"Bitrate: {bitrate} kbps")
    """
    try:
        # First try to get stream-level bitrate
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        stream = data.get("streams", [{}])[0]
        bit_rate = stream.get("bit_rate")

        if bit_rate:
            return float(bit_rate) / 1000  # Convert to kbps

        # If bit_rate is not in stream, try format-level bitrate
        # (less accurate but better than nothing)
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=bit_rate",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        format_bit_rate = data.get("format", {}).get("bit_rate")
        if format_bit_rate:
            return float(format_bit_rate) / 1000  # Convert to kbps

        return None
    except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError):
        return None
