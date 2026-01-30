"""
Integration tests for encode_study.py script.

These tests verify the encoding workflow using real tools (FFmpeg, SVT-AV1)
but with minimal test fixtures to keep tests fast.
"""

import sys
from pathlib import Path

import pytest


# Import the encode_study module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
import encode_study


class TestNormalizeParamValue:
    """Test parameter value normalization."""

    def test_normalize_single_value_to_list(self):
        """Test that single values are wrapped in a list."""
        assert encode_study.normalize_param_value(42) == [42]

    def test_normalize_boolean(self):
        """Test normalization of boolean values."""
        assert encode_study.normalize_param_value(True) == [True]
        assert encode_study.normalize_param_value(False) == [False]

    def test_normalize_string(self):
        """Test normalization of string values."""
        assert encode_study.normalize_param_value("preset") == ["preset"]

    def test_normalize_list_unchanged(self):
        """Test that lists are returned unchanged."""
        assert encode_study.normalize_param_value([1, 2, 3]) == [1, 2, 3]
        assert encode_study.normalize_param_value(["a", "b"]) == ["a", "b"]


class TestGenerateParamCombinations:
    """Test parameter combination generation."""

    def test_single_parameter_single_value(self):
        """Test with one parameter having one value."""
        params = {"preset": [8]}
        result = encode_study.generate_param_combinations(params)

        assert len(result) == 1
        assert result[0] == {"preset": 8}

    def test_single_parameter_multiple_values(self):
        """Test with one parameter having multiple values."""
        params = {"preset": [8, 10, 12]}
        result = encode_study.generate_param_combinations(params)

        assert len(result) == 3
        assert {"preset": 8} in result
        assert {"preset": 10} in result
        assert {"preset": 12} in result

    def test_multiple_parameters_cartesian_product(self):
        """Test cartesian product of multiple parameters."""
        params = {"preset": [8, 10], "crf": [30, 35]}
        result = encode_study.generate_param_combinations(params)

        assert len(result) == 4
        assert {"preset": 8, "crf": 30} in result
        assert {"preset": 8, "crf": 35} in result
        assert {"preset": 10, "crf": 30} in result
        assert {"preset": 10, "crf": 35} in result

    def test_three_parameters(self):
        """Test with three parameters."""
        params = {"preset": [8, 10], "crf": [30], "film_grain": [0, 10]}
        result = encode_study.generate_param_combinations(params)

        assert len(result) == 4  # 2 * 1 * 2
        assert {"preset": 8, "crf": 30, "film_grain": 0} in result
        assert {"preset": 10, "crf": 30, "film_grain": 10} in result

    def test_boolean_parameters(self):
        """Test with boolean parameter values."""
        params = {"enable_qm": [True, False], "preset": [8]}
        result = encode_study.generate_param_combinations(params)

        assert len(result) == 2
        assert {"enable_qm": True, "preset": 8} in result
        assert {"enable_qm": False, "preset": 8} in result

    def test_normalizes_single_values(self):
        """Test that single values (non-lists) are normalized to lists."""
        params = {"preset": 8, "crf": 35}
        result = encode_study.generate_param_combinations(params)

        assert len(result) == 1
        assert result[0] == {"preset": 8, "crf": 35}


class TestBuildOutputFilename:
    """Test structured filename generation."""

    def test_basic_filename(self):
        """Test basic filename with clip and parameters."""
        result = encode_study.build_output_filename("test_clip.mp4", {"preset": 8, "crf": 35})

        assert result.startswith("test_clip")
        assert "p8" in result
        assert "crf35" in result
        assert result.endswith(".mkv")

    def test_filename_with_film_grain(self):
        """Test filename includes optional parameters."""
        result = encode_study.build_output_filename(
            "clip.mp4", {"preset": 8, "crf": 30, "film_grain": 5}
        )

        assert "fg5" in result
        assert result.endswith(".mkv")

    def test_filename_stem_extracted(self):
        """Test that only the stem of clip name is used."""
        result = encode_study.build_output_filename("my-clip_001.mp4", {"preset": 8, "crf": 30})

        assert result.startswith("my-clip_001_")
        assert ".mp4" not in result or result.endswith(".mkv")


class TestLoadStudyConfig:
    """Test study configuration loading and validation."""

    def test_valid_config(self, tmp_path: Path):
        """Test loading a valid configuration."""
        import json

        config = {
            "study_name": "test_study",
            "description": "Test description",
            "parameters": {"preset": [8], "crf": [35]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        result = encode_study.load_study_config(config_file)

        assert result["study_name"] == "test_study"
        assert result["parameters"]["preset"] == [8]

    def test_missing_study_name(self, tmp_path: Path):
        """Test validation fails with missing study_name."""
        import json

        config = {
            "description": "Test",
            "parameters": {"preset": [8], "crf": [35]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="study_name"):
            encode_study.load_study_config(config_file)

    def test_missing_required_param(self, tmp_path: Path):
        """Test validation fails when missing required parameter."""
        import json

        config = {
            "study_name": "test",
            "description": "Test",
            "parameters": {"preset": [8]},  # Missing crf
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="crf"):
            encode_study.load_study_config(config_file)


@pytest.mark.integration
@pytest.mark.requires_ffmpeg
class TestBuildFFmpegCommand:
    """Test FFmpeg command construction."""

    def test_basic_command_structure(self, tmp_path: Path):
        """Test basic FFmpeg command structure."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mkv"
        params = {"preset": 8, "crf": 35}

        cmd = encode_study.build_ffmpeg_command(input_file, output_file, params)

        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(input_file) in cmd
        assert str(output_file) in cmd
        assert "-c:v" in cmd
        assert "libsvtav1" in cmd

    def test_command_with_film_grain(self, tmp_path: Path):
        """Test command with film_grain parameter."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mkv"
        params = {"preset": 8, "crf": 35, "film_grain": 10}

        cmd = encode_study.build_ffmpeg_command(input_file, output_file, params)

        assert "-svtav1-params" in cmd
        svtav1_idx = cmd.index("-svtav1-params")
        assert "film-grain=10" in cmd[svtav1_idx + 1]

    def test_command_overwrite_flag(self, tmp_path: Path):
        """Test that -y flag is present for overwriting."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mkv"

        cmd = encode_study.build_ffmpeg_command(input_file, output_file, {"preset": 8, "crf": 30})

        assert "-y" in cmd


@pytest.mark.integration
class TestSystemInfo:
    """Test system information gathering."""

    def test_get_system_info_structure(self):
        """Test that system info has expected structure."""
        info = encode_study.get_system_info()

        # Check required fields
        assert "os" in info
        assert "cpu" in info
        assert "cpu_cores" in info

    def test_system_info_types(self):
        """Test that system info values have correct types."""
        info = encode_study.get_system_info()

        assert isinstance(info["os"], str)
        assert isinstance(info["cpu"], str)
        assert info["cpu_cores"] is None or isinstance(info["cpu_cores"], int)


class TestFindClips:
    """Test clip discovery in directory."""

    def test_find_video_files(self, tmp_path: Path):
        """Test finding video files with various extensions."""
        (tmp_path / "clip1.mp4").touch()
        (tmp_path / "clip2.mkv").touch()
        (tmp_path / "clip3.webm").touch()
        (tmp_path / "readme.txt").touch()

        result = encode_study.find_clips(tmp_path)

        assert len(result) == 3
        assert all(p.suffix in [".mp4", ".mkv", ".webm"] for p in result)

    def test_excludes_hidden_files(self, tmp_path: Path):
        """Test that hidden files are excluded."""
        (tmp_path / "clip.mp4").touch()
        (tmp_path / ".hidden.mp4").touch()

        result = encode_study.find_clips(tmp_path)

        assert len(result) == 1
        assert result[0].name == "clip.mp4"

    def test_returns_sorted(self, tmp_path: Path):
        """Test that results are sorted alphabetically."""
        (tmp_path / "z_clip.mp4").touch()
        (tmp_path / "a_clip.mp4").touch()
        (tmp_path / "m_clip.mp4").touch()

        result = encode_study.find_clips(tmp_path)

        assert result[0].name == "a_clip.mp4"
        assert result[1].name == "m_clip.mp4"
        assert result[2].name == "z_clip.mp4"


def _has_svtav1_encoder() -> bool:
    """Check if FFmpeg has libsvtav1 encoder support."""
    import subprocess

    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
        return "libsvtav1" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Integration test that requires SVT-AV1 encoder
@pytest.mark.integration
@pytest.mark.requires_ffmpeg
@pytest.mark.skipif(
    not _has_svtav1_encoder(),
    reason="Requires FFmpeg with libsvtav1 encoder support",
)
def test_full_encoding_workflow(test_video_path: Path, tmp_path: Path):
    """
    Full integration test of encoding workflow.

    This test requires:
    1. A small test video in tests/fixtures/
    2. FFmpeg with libsvtav1 support
    3. Sufficient disk space for encoded output
    """
    if not test_video_path.exists():
        pytest.skip("Test video fixture not found")

    output_path = tmp_path / "encoded.mkv"
    params = {"preset": 12, "crf": 45}  # Fast settings for testing

    result = encode_study.encode_clip(
        clip_path=test_video_path,
        output_path=output_path,
        params=params,
        verbose=True,
    )

    assert result["success"] is True
    assert output_path.exists()
    assert result["file_size_bytes"] > 0
    assert result["encoding_time_seconds"] > 0
