"""
Unit tests for shared utility functions.

These tests validate the common utilities used across scripts without
requiring external dependencies like FFmpeg.
"""

from pathlib import Path

from utils import calculate_sha256, get_video_bitrate


class TestCalculateSHA256:
    """Tests for the calculate_sha256 utility function."""

    def test_calculate_sha256_empty_file(self, tmp_path: Path):
        """Test SHA256 calculation for an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        result = calculate_sha256(test_file)

        # SHA256 of empty file is known constant
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected

    def test_calculate_sha256_small_file(self, tmp_path: Path):
        """Test SHA256 calculation for a small text file."""
        test_file = tmp_path / "test.txt"
        content = b"Hello, World!"
        test_file.write_bytes(content)

        result = calculate_sha256(test_file)

        # SHA256 of "Hello, World!" is known
        expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        assert result == expected

    def test_calculate_sha256_large_file(self, tmp_path: Path):
        """Test SHA256 calculation for a file larger than buffer size."""
        test_file = tmp_path / "large.bin"
        # Create file larger than 4096 byte buffer
        content = b"X" * 10000
        test_file.write_bytes(content)

        result = calculate_sha256(test_file)

        # Verify it returns a valid 64-character hex string
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_calculate_sha256_deterministic(self, tmp_path: Path):
        """Test that SHA256 calculation is deterministic."""
        test_file = tmp_path / "test.bin"
        content = b"Test content for determinism"
        test_file.write_bytes(content)

        result1 = calculate_sha256(test_file)
        result2 = calculate_sha256(test_file)

        assert result1 == result2

    def test_calculate_sha256_different_files(self, tmp_path: Path):
        """Test that different files produce different checksums."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_bytes(b"Content A")
        file2.write_bytes(b"Content B")

        checksum1 = calculate_sha256(file1)
        checksum2 = calculate_sha256(file2)

        assert checksum1 != checksum2

    def test_calculate_sha256_binary_file(self, tmp_path: Path):
        """Test SHA256 calculation for binary data."""
        test_file = tmp_path / "binary.dat"
        # Binary data with null bytes
        content = bytes(range(256))
        test_file.write_bytes(content)

        result = calculate_sha256(test_file)

        # Should handle binary data correctly
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestGetVideoBitrate:
    """Tests for the get_video_bitrate utility function."""

    def test_get_video_bitrate_with_fixture(self):
        """Test bitrate extraction using the fixture video file."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "test_video.mp4"

        if not fixture_path.exists():
            # Skip test if fixture doesn't exist
            return

        bitrate = get_video_bitrate(fixture_path)

        # Should return a positive bitrate value
        assert bitrate is not None
        assert bitrate > 0
        # Typical video bitrates are in reasonable range (10 kbps - 100 Mbps)
        assert 10 < bitrate < 100_000

    def test_get_video_bitrate_nonexistent_file(self):
        """Test bitrate extraction with non-existent file."""
        nonexistent = Path("/tmp/nonexistent_video.mp4")

        bitrate = get_video_bitrate(nonexistent)

        # Should return None for missing file
        assert bitrate is None

    def test_get_video_bitrate_invalid_file(self, tmp_path: Path):
        """Test bitrate extraction with invalid video file."""
        invalid_file = tmp_path / "not_a_video.txt"
        invalid_file.write_text("This is not a video file")

        bitrate = get_video_bitrate(invalid_file)

        # Should return None for invalid video
        assert bitrate is None
