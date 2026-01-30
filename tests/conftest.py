"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides common fixtures
and configuration for all tests.
"""

import sys
from pathlib import Path

import pytest


# Add scripts directory to Python path so tests can import modules
REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root directory."""
    return REPO_ROOT


@pytest.fixture
def scripts_dir() -> Path:
    """Return the scripts directory."""
    return SCRIPTS_DIR


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_video_path(fixtures_dir: Path) -> Path:
    """
    Return path to test video fixture.

    This should be a very short (2-3 seconds), low resolution video
    suitable for fast testing.
    """
    return fixtures_dir / "test_video.mp4"


@pytest.fixture
def sample_study_config() -> dict:
    """Return a minimal study configuration for testing."""
    return {
        "name": "test_study",
        "description": "Test study configuration",
        "output_directory": "test_output",
        "encoder": "svt-av1",
        "parameters": {"preset": [8, 10], "crf": [35, 40]},
    }


@pytest.fixture
def sample_clip_metadata() -> dict:
    """Return sample clip metadata for testing."""
    return {
        "source_video": "test_source.mp4",
        "category": "animation",
        "resolution": "1920x1080",
        "fps": 24,
        "duration": 10.0,
        "clips": [
            {
                "filename": "test_clip_001.mp4",
                "start_time": 5.0,
                "duration": 10.0,
                "sha256": "0" * 64,
            }
        ],
    }


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "raw_videos").mkdir(parents=True)
    (data_dir / "test_clips").mkdir(parents=True)
    (data_dir / "encoded").mkdir(parents=True)
    return data_dir


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: Integration tests using real tools")
    config.addinivalue_line("markers", "unit: Fast unit tests without external dependencies")
    config.addinivalue_line("markers", "slow: Tests that take more than 10 seconds")
    config.addinivalue_line("markers", "requires_ffmpeg: Tests requiring FFmpeg installation")
    config.addinivalue_line("markers", "requires_vmaf: Tests requiring VMAF model files")
