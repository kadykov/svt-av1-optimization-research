"""
Unit tests for metadata integrity across scripts.

These tests verify that metadata stored in JSON files matches actual file properties,
ensuring that data flowing between scripts doesn't get corrupted or out of sync.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def get_video_duration(video_path: Path) -> float | None:
    """Get video duration using FFprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def get_video_info(video_path: Path) -> dict | None:
    """Get video resolution using FFprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        if "streams" in data and len(data["streams"]) > 0:
            stream = data["streams"][0]
            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
            }
    except (subprocess.CalledProcessError, ValueError):
        pass
    return None


class TestClipMetadataIntegrity:
    """Test that clip metadata matches actual file properties."""

    @pytest.mark.requires_ffmpeg
    def test_clip_metadata_duration_matches_actual(self):
        """Test that clip metadata duration field matches actual file duration.

        This test catches a bug where extract_clips.py stores the requested
        duration instead of the actual extracted duration, causing VMAF
        comparisons to fail silently.
        """
        clip_metadata_file = Path("data/test_clips/clip_metadata.json")
        clips_dir = Path("data/test_clips")

        if not clip_metadata_file.exists():
            pytest.skip("Clip metadata not found - clips not extracted yet")

        with open(clip_metadata_file) as f:
            metadata = json.load(f)

        errors = []
        for clip in metadata["clips"]:
            clip_name = clip["clip_name"]
            stored_duration = clip.get("duration")
            actual_duration_from_file = clip.get("actual_duration")
            clip_path = clips_dir / clip_name

            # Check if clip file exists
            if not clip_path.exists():
                errors.append(f"Clip file not found: {clip_name}")
                continue

            # Get actual duration from file
            file_duration = get_video_duration(clip_path)
            if file_duration is None:
                errors.append(f"Could not determine duration of {clip_name}")
                continue

            # Check if actual_duration field exists and is accurate
            if actual_duration_from_file is None:
                errors.append(
                    f"{clip_name}: metadata missing 'actual_duration' field "
                    f"(stored={stored_duration:.2f}s, actual={file_duration:.2f}s)"
                )
            # Allow 0.1s tolerance for rounding
            elif abs(actual_duration_from_file - file_duration) > 0.1:
                errors.append(
                    f"{clip_name}: actual_duration mismatch "
                    f"(stored={actual_duration_from_file:.3f}s, file={file_duration:.3f}s)"
                )

            # Check if stored duration is reasonably close to file duration
            # (but only warn, don't fail - it's the requested vs actual)
            duration_diff = abs(stored_duration - file_duration)
            if duration_diff > 5.0:  # More than 5 seconds difference is suspicious
                print(
                    f"\n⚠️  {clip_name}: Large duration difference "
                    f"(requested={stored_duration:.2f}s, actual={file_duration:.2f}s)"
                )

        if errors:
            pytest.fail("Clip metadata integrity issues found:\n" + "\n".join(errors))

    @pytest.mark.requires_ffmpeg
    def test_clip_metadata_resolution_matches_actual(self):
        """Test that clip metadata resolution matches actual file properties."""
        clip_metadata_file = Path("data/test_clips/clip_metadata.json")
        clips_dir = Path("data/test_clips")

        if not clip_metadata_file.exists():
            pytest.skip("Clip metadata not found - clips not extracted yet")

        with open(clip_metadata_file) as f:
            metadata = json.load(f)

        errors = []
        for clip in metadata["clips"]:
            clip_name = clip["clip_name"]
            stored_width = clip.get("source_width")
            stored_height = clip.get("source_height")
            clip_path = clips_dir / clip_name

            if not clip_path.exists():
                errors.append(f"Clip file not found: {clip_name}")
                continue

            info = get_video_info(clip_path)
            if info is None:
                errors.append(f"Could not determine properties of {clip_name}")
                continue

            if stored_width != info["width"] or stored_height != info["height"]:
                errors.append(
                    f"{clip_name}: resolution mismatch "
                    f"(stored={stored_width}x{stored_height}, actual={info['width']}x{info['height']})"
                )

        if errors:
            pytest.fail("Clip metadata resolution issues found:\n" + "\n".join(errors))

    @pytest.mark.requires_ffmpeg
    def test_clip_file_size_matches_metadata(self):
        """Test that stored file size matches actual file size."""
        clip_metadata_file = Path("data/test_clips/clip_metadata.json")
        clips_dir = Path("data/test_clips")

        if not clip_metadata_file.exists():
            pytest.skip("Clip metadata not found - clips not extracted yet")

        with open(clip_metadata_file) as f:
            metadata = json.load(f)

        errors = []
        for clip in metadata["clips"]:
            clip_name = clip["clip_name"]
            stored_size = clip.get("file_size_bytes")
            clip_path = clips_dir / clip_name

            if not clip_path.exists():
                errors.append(f"Clip file not found: {clip_name}")
                continue

            actual_size = clip_path.stat().st_size
            if stored_size != actual_size:
                errors.append(
                    f"{clip_name}: file size mismatch "
                    f"(stored={stored_size} bytes, actual={actual_size} bytes)"
                )

        if errors:
            pytest.fail("Clip metadata file size issues found:\n" + "\n".join(errors))


class TestEncodingMetadataIntegrity:
    """Test that encoding metadata matches actual encoded files."""

    @pytest.mark.requires_ffmpeg
    def test_encoding_file_size_matches_metadata(self):
        """Test that stored encoding file sizes match actual files."""
        encoding_dir = Path("data/encoded/baseline_sweep")
        metadata_file = encoding_dir / "encoding_metadata.json"

        if not metadata_file.exists():
            pytest.skip("Encoding metadata not found - study not encoded yet")

        with open(metadata_file) as f:
            metadata = json.load(f)

        errors = []
        for encoding in metadata.get("encodings", []):
            output_file = encoding.get("output_file")
            stored_size = encoding.get("file_size_bytes")
            file_path = encoding_dir / output_file

            if not file_path.exists():
                errors.append(f"Encoded file not found: {output_file}")
                continue

            actual_size = file_path.stat().st_size
            if stored_size != actual_size:
                errors.append(
                    f"{output_file}: file size mismatch "
                    f"(stored={stored_size} bytes, actual={actual_size} bytes)"
                )

        if errors:
            pytest.fail("Encoding metadata file size issues found:\n" + "\n".join(errors[:10]))

    @pytest.mark.requires_ffmpeg
    def test_encoding_source_clip_matches_reference(self):
        """Test that encoding metadata references correct source clip."""
        encoding_dir = Path("data/encoded/baseline_sweep")
        metadata_file = encoding_dir / "encoding_metadata.json"
        clips_dir = Path("data/test_clips")

        if not metadata_file.exists():
            pytest.skip("Encoding metadata not found - study not encoded yet")

        with open(metadata_file) as f:
            metadata = json.load(f)

        errors = []
        checked_clips = set()

        for encoding in metadata.get("encodings", []):
            source_clip_name = encoding.get("source_clip")

            if source_clip_name in checked_clips:
                continue
            checked_clips.add(source_clip_name)

            clip_path = clips_dir / source_clip_name
            if not clip_path.exists():
                # Try common extensions
                found = False
                for ext in [".mp4", ".mkv", ".mov", ".avi"]:
                    alt_path = clips_dir / f"{Path(source_clip_name).stem}{ext}"
                    if alt_path.exists():
                        found = True
                        break
                if not found:
                    errors.append(f"Source clip not found: {source_clip_name}")

        if errors:
            pytest.fail(
                "Encoding metadata source clip references missing files:\n" + "\n".join(errors)
            )
