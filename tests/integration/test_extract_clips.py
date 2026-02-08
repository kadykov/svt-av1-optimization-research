"""
Integration tests for extract_clips.py script.

These tests verify clip extraction functionality.
"""

import sys
from pathlib import Path

import pytest


# Import the extract_clips module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
import extract_clips


class TestFilterVideos:
    """Test video filtering functionality."""

    def test_filter_by_category(self):
        """Test filtering videos by category."""
        videos = [
            {
                "id": "1",
                "categories": ["animation"],
                "info": {"width": 1920, "height": 1080, "fps": 24},
            },
            {
                "id": "2",
                "categories": ["live-action"],
                "info": {"width": 1920, "height": 1080, "fps": 30},
            },
            {
                "id": "3",
                "categories": ["animation"],
                "info": {"width": 1280, "height": 720, "fps": 24},
            },
        ]

        result = extract_clips.filter_videos(videos, category="animation")

        assert len(result) == 2
        assert all("animation" in v["categories"] for v in result)

    def test_filter_by_max_width(self):
        """Test filtering videos by max width."""
        videos = [
            {"id": "1", "categories": [], "info": {"width": 1920, "height": 1080, "fps": 24}},
            {"id": "2", "categories": [], "info": {"width": 1280, "height": 720, "fps": 24}},
            {"id": "3", "categories": [], "info": {"width": 3840, "height": 2160, "fps": 30}},
        ]

        result = extract_clips.filter_videos(videos, max_width=1920)

        assert len(result) == 2
        assert all(v["info"]["width"] <= 1920 for v in result)

    def test_filter_by_fps_range(self):
        """Test filtering videos by FPS range."""
        videos = [
            {"id": "1", "categories": [], "info": {"width": 1920, "height": 1080, "fps": 24}},
            {"id": "2", "categories": [], "info": {"width": 1920, "height": 1080, "fps": 30}},
            {"id": "3", "categories": [], "info": {"width": 1920, "height": 1080, "fps": 60}},
        ]

        result = extract_clips.filter_videos(videos, min_fps=25, max_fps=35)

        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_filter_multiple_criteria(self):
        """Test filtering with multiple criteria."""
        videos = [
            {
                "id": "1",
                "categories": ["animation"],
                "info": {"width": 1920, "height": 1080, "fps": 24},
            },
            {
                "id": "2",
                "categories": ["live-action"],
                "info": {"width": 1920, "height": 1080, "fps": 24},
            },
            {
                "id": "3",
                "categories": ["animation"],
                "info": {"width": 1280, "height": 720, "fps": 24},
            },
            {
                "id": "4",
                "categories": ["animation"],
                "info": {"width": 1920, "height": 1080, "fps": 60},
            },
        ]

        result = extract_clips.filter_videos(
            videos, category="animation", max_width=1920, max_fps=30
        )

        assert len(result) == 2
        assert all("animation" in v["categories"] for v in result)
        assert all(v["info"]["fps"] <= 30 for v in result)

    def test_filter_no_match(self):
        """Test filtering with no matches."""
        videos = [
            {
                "id": "1",
                "categories": ["animation"],
                "info": {"width": 1920, "height": 1080, "fps": 24},
            },
        ]

        result = extract_clips.filter_videos(videos, category="documentary")

        assert len(result) == 0

    def test_filter_no_criteria(self):
        """Test filtering with no criteria returns all videos."""
        videos = [
            {
                "id": "1",
                "categories": ["animation"],
                "info": {"width": 1920, "height": 1080, "fps": 24},
            },
            {
                "id": "2",
                "categories": ["live-action"],
                "info": {"width": 1280, "height": 720, "fps": 30},
            },
        ]

        result = extract_clips.filter_videos(videos)

        assert len(result) == 2


class TestGenerateRandomClips:
    """Test random clip specification generation."""

    def test_generate_clips_basic(self):
        """Test generating clip specifications."""
        videos = [
            {
                "id": "video1",
                "path": "/path/to/video1.mp4",
                "categories": ["animation"],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 100.0},
            }
        ]

        result = extract_clips.generate_random_clips(
            videos, num_clips=3, min_duration=5.0, max_duration=10.0, seed=42
        )

        assert len(result) <= 3
        for clip in result:
            assert 5.0 <= clip["duration"] <= 10.0
            assert clip["start_time"] >= 0
            assert clip["start_time"] + clip["duration"] <= 100.0

    def test_generate_clips_with_seed(self):
        """Test that seed produces deterministic results."""
        videos = [
            {
                "id": "video1",
                "path": "/path/to/video1.mp4",
                "categories": [],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 100.0},
            }
        ]

        result1 = extract_clips.generate_random_clips(
            videos, num_clips=3, min_duration=5.0, max_duration=10.0, seed=42
        )
        result2 = extract_clips.generate_random_clips(
            videos, num_clips=3, min_duration=5.0, max_duration=10.0, seed=42
        )

        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2, strict=True):
            assert c1["start_time"] == c2["start_time"]
            assert c1["duration"] == c2["duration"]

    def test_generate_clips_proportional_distribution(self):
        """Test that clips are distributed proportionally across videos."""
        videos = [
            {
                "id": "long_video",
                "path": "/path/to/long.mp4",
                "categories": [],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 1000.0},
            },
            {
                "id": "short_video",
                "path": "/path/to/short.mp4",
                "categories": [],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 100.0},
            },
        ]

        result = extract_clips.generate_random_clips(
            videos, num_clips=10, min_duration=5.0, max_duration=10.0, seed=42
        )

        # Longer video should have more clips
        long_clips = [c for c in result if c["video_id"] == "long_video"]
        short_clips = [c for c in result if c["video_id"] == "short_video"]

        # At least one clip per video
        assert len(long_clips) >= 1
        assert len(short_clips) >= 1

    def test_clip_contains_metadata(self):
        """Test that generated clips contain source metadata."""
        videos = [
            {
                "id": "video1",
                "path": "/path/to/video1.mp4",
                "categories": ["test"],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 100.0},
            }
        ]

        result = extract_clips.generate_random_clips(
            videos, num_clips=1, min_duration=5.0, max_duration=10.0, seed=42
        )

        assert len(result) == 1
        clip = result[0]
        assert clip["video_id"] == "video1"
        assert clip["video_path"] == "/path/to/video1.mp4"
        assert clip["source_width"] == 1920
        assert clip["source_height"] == 1080
        assert clip["source_fps"] == 24
        assert clip["categories"] == ["test"]

    def test_generate_clips_with_usable_time_range(self):
        """Test that clips respect usable time range constraints."""
        videos = [
            {
                "id": "video1",
                "path": "/path/to/video1.mp4",
                "categories": ["test"],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 100.0},
                "usable_time_range": {"start": 10.0, "end": 60.0},
            }
        ]

        result = extract_clips.generate_random_clips(
            videos, num_clips=5, min_duration=5.0, max_duration=10.0, seed=42
        )

        # All clips should be within the usable time range
        for clip in result:
            assert clip["start_time"] >= 10.0
            assert clip["start_time"] + clip["duration"] <= 60.0

    def test_generate_clips_mixed_usable_ranges(self):
        """Test clip generation with some videos having usable ranges and some not."""
        videos = [
            {
                "id": "video1",
                "path": "/path/to/video1.mp4",
                "categories": [],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 100.0},
                "usable_time_range": {"start": 10.0, "end": 50.0},  # 40s usable
            },
            {
                "id": "video2",
                "path": "/path/to/video2.mp4",
                "categories": [],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 100.0},
                # No usable_time_range - entire 100s usable
            },
        ]

        result = extract_clips.generate_random_clips(
            videos, num_clips=10, min_duration=5.0, max_duration=10.0, seed=42
        )

        # Check clips from video1 respect the range
        video1_clips = [c for c in result if c["video_id"] == "video1"]
        for clip in video1_clips:
            assert clip["start_time"] >= 10.0
            assert clip["start_time"] + clip["duration"] <= 50.0

        # Check clips from video2 can use the entire duration
        video2_clips = [c for c in result if c["video_id"] == "video2"]
        for clip in video2_clips:
            assert clip["start_time"] >= 0.0
            assert clip["start_time"] + clip["duration"] <= 100.0

    def test_proportional_distribution_with_usable_range(self):
        """Test that proportional distribution uses usable duration, not total duration."""
        videos = [
            {
                "id": "video1",
                "path": "/path/to/video1.mp4",
                "categories": [],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 1000.0},
                "usable_time_range": {"start": 0.0, "end": 100.0},  # Only 100s usable
            },
            {
                "id": "video2",
                "path": "/path/to/video2.mp4",
                "categories": [],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 100.0},
                # No restriction - entire 100s usable
            },
        ]

        result = extract_clips.generate_random_clips(
            videos, num_clips=10, min_duration=5.0, max_duration=10.0, seed=42
        )

        # Both videos have 100s usable, so clips should be distributed roughly equally
        video1_clips = [c for c in result if c["video_id"] == "video1"]
        video2_clips = [c for c in result if c["video_id"] == "video2"]

        # Each should have at least 1 clip and roughly similar counts
        assert len(video1_clips) >= 1
        assert len(video2_clips) >= 1
        # They should be within a reasonable range of each other
        assert abs(len(video1_clips) - len(video2_clips)) <= 3

    def test_usable_range_shorter_than_clip_duration(self):
        """Test handling when usable range is shorter than requested clip duration."""
        videos = [
            {
                "id": "video1",
                "path": "/path/to/video1.mp4",
                "categories": [],
                "info": {"width": 1920, "height": 1080, "fps": 24, "duration": 100.0},
                "usable_time_range": {"start": 10.0, "end": 15.0},  # Only 5s usable
            }
        ]

        # Request 10s clips, but only 5s is available
        result = extract_clips.generate_random_clips(
            videos, num_clips=1, min_duration=10.0, max_duration=20.0, seed=42
        )

        assert len(result) == 1
        clip = result[0]
        # Should use the entire usable range
        assert clip["start_time"] == 10.0
        assert clip["duration"] == 5.0


@pytest.mark.integration
@pytest.mark.requires_ffmpeg
def test_get_video_info(test_video_path: Path):
    """Test getting video info using FFprobe (requires FFprobe)."""
    # Skip if test video doesn't exist
    if not test_video_path.exists():
        pytest.skip("Test video fixture not found")

    info = extract_clips.get_video_info(test_video_path)

    assert info is not None
    assert "duration" in info
    assert "width" in info
    assert "height" in info
    assert "fps" in info

    # Test video is 2 seconds, 320x240, 24fps
    assert info["duration"] == pytest.approx(2.0, abs=0.1)
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["fps"] == 24.0


@pytest.mark.integration
@pytest.mark.requires_ffmpeg
def test_extract_clip(test_video_path: Path, tmp_path: Path):
    """Test extracting a clip from a video (requires FFmpeg)."""
    # Skip if test video doesn't exist
    if not test_video_path.exists():
        pytest.skip("Test video fixture not found")

    output_path = tmp_path / "extracted_clip.mkv"

    # Extract a 0.5-second clip starting at 0.2 seconds
    success = extract_clips.extract_clip(
        video_path=test_video_path,
        start_time=0.2,
        duration=0.5,
        output_path=output_path,
    )

    assert success is True
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Verify the extracted clip has correct properties
    clip_info = extract_clips.get_video_info(output_path)
    assert clip_info is not None
    assert clip_info["width"] == 320
    assert clip_info["height"] == 240
    # Duration should be approximately 0.5 seconds (allow tolerance for codec)
    assert 0.3 <= clip_info["duration"] <= 1.0
