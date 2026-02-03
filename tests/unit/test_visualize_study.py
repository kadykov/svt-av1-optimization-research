"""
Unit tests for visualize_study.py.

Tests the data preparation and analysis functions without generating plots.
"""

# Import the functions we want to test
import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from visualize_study import prepare_dataframe


@pytest.fixture
def sample_analysis_data():
    """Sample analysis metadata for testing."""
    return {
        "study_name": "test_study",
        "analysis_date": "2026-01-30T00:00:00Z",
        "metrics_calculated": ["vmaf", "psnr", "ssim"],
        "vmaf_model": "vmaf_v0.6.1neg",
        "clips_analyzed": 1,
        "total_encodings_analyzed": 3,
        "encodings": [
            {
                "output_file": "clip_p6_crf20.mkv",
                "source_clip": "test_clip.mp4",
                "parameters": {"preset": 6, "crf": 20},
                "metrics": {
                    "vmaf": {
                        "mean": 95.5,
                        "harmonic_mean": 94.2,
                        "min": 85.0,
                        "percentile_1": 88.0,
                        "percentile_5": 90.0,
                        "percentile_25": 94.0,
                        "percentile_50": 95.5,
                        "percentile_75": 97.0,
                        "percentile_95": 98.5,
                        "std_dev": 3.2,
                    },
                    "psnr": {"avg_mean": 42.5},
                    "ssim": {"avg_mean": 0.98},
                },
                "file_size_bytes": 5242880,  # 5 MB
                "encoding_time_seconds": 10.5,
                "encoding_fps": 25.0,
                "analysis_time_seconds": 8.2,
                "success": True,
                "error": None,
                "efficiency_metrics": {
                    "vmaf_per_mbyte": 19.1,
                    "quality_per_encoding_second": 9.1,
                },
            },
            {
                "output_file": "clip_p6_crf30.mkv",
                "source_clip": "test_clip.mp4",
                "parameters": {"preset": 6, "crf": 30},
                "metrics": {
                    "vmaf": {
                        "mean": 92.0,
                        "harmonic_mean": 90.5,
                        "min": 80.0,
                        "percentile_1": 83.0,
                        "percentile_5": 85.0,
                        "percentile_25": 90.0,
                        "percentile_50": 92.0,
                        "percentile_75": 94.0,
                        "percentile_95": 96.0,
                        "std_dev": 4.1,
                    },
                    "psnr": {"avg_mean": 40.2},
                    "ssim": {"avg_mean": 0.96},
                },
                "file_size_bytes": 3145728,  # 3 MB
                "encoding_time_seconds": 10.2,
                "encoding_fps": 26.0,
                "analysis_time_seconds": 8.0,
                "success": True,
                "error": None,
                "efficiency_metrics": {
                    "vmaf_per_mbyte": 30.7,
                    "quality_per_encoding_second": 9.0,
                },
            },
            {
                "output_file": "clip_p8_crf20.mkv",
                "source_clip": "test_clip.mp4",
                "parameters": {"preset": 8, "crf": 20},
                "metrics": {
                    "vmaf": {
                        "mean": 94.8,
                        "harmonic_mean": 93.5,
                        "min": 84.0,
                        "percentile_1": 87.0,
                        "percentile_5": 89.5,
                        "percentile_25": 93.5,
                        "percentile_50": 95.0,
                        "percentile_75": 96.5,
                        "percentile_95": 98.0,
                        "std_dev": 3.5,
                    },
                    "psnr": {"avg_mean": 42.0},
                    "ssim": {"avg_mean": 0.975},
                },
                "file_size_bytes": 5767168,  # 5.5 MB
                "encoding_time_seconds": 6.8,
                "encoding_fps": 38.0,
                "analysis_time_seconds": 8.1,
                "success": True,
                "error": None,
                "efficiency_metrics": {
                    "vmaf_per_mbyte": 17.2,
                    "quality_per_encoding_second": 13.9,
                },
            },
        ],
    }


def test_prepare_dataframe_structure(sample_analysis_data):
    """Test that prepare_dataframe creates correct structure."""
    df = prepare_dataframe(sample_analysis_data)

    # Check basic structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 3 successful encodings

    # Check key columns exist
    expected_columns = [
        "output_file",
        "source_clip",
        "preset",
        "crf",
        "file_size_mb",
        "encoding_time_s",
        "vmaf_mean",
        "vmaf_harmonic_mean",
        "psnr_avg",
        "ssim_avg",
        "bpp",
        "vmaf_per_bpp",
        "vmaf_per_time",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"


def test_prepare_dataframe_values(sample_analysis_data):
    """Test that values are correctly extracted and converted."""
    df = prepare_dataframe(sample_analysis_data)

    # Check first encoding
    first = df.iloc[0]
    assert first["preset"] == 6
    assert first["crf"] == 20
    assert first["vmaf_mean"] == 95.5
    assert first["vmaf_harmonic_mean"] == 94.2
    assert first["vmaf_min"] == 85.0
    assert first["vmaf_median"] == 95.5
    assert first["psnr_avg"] == 42.5
    assert first["ssim_avg"] == 0.98
    assert abs(first["file_size_mb"] - 5.0) < 0.01  # 5 MB
    assert first["encoding_time_s"] == 10.5


def test_prepare_dataframe_percentiles(sample_analysis_data):
    """Test that VMAF percentiles are correctly extracted."""
    df = prepare_dataframe(sample_analysis_data)

    first = df.iloc[0]
    assert first["vmaf_p1"] == 88.0
    assert first["vmaf_p5"] == 90.0
    assert first["vmaf_p25"] == 94.0
    assert first["vmaf_p75"] == 97.0
    assert first["vmaf_p95"] == 98.5
    assert first["vmaf_std"] == 3.2


def test_prepare_dataframe_filters_failures(sample_analysis_data):
    """Test that failed encodings are excluded."""
    # Add a failed encoding
    sample_analysis_data["encodings"].append(
        {
            "output_file": "failed.mkv",
            "source_clip": "test_clip.mp4",
            "parameters": {"preset": 6, "crf": 40},
            "metrics": {},
            "file_size_bytes": 0,
            "success": False,
            "error": "Encoding failed",
        }
    )

    df = prepare_dataframe(sample_analysis_data)

    # Should still have only 3 rows (failed one excluded)
    assert len(df) == 3
    assert "failed.mkv" not in df["output_file"].values


def test_prepare_dataframe_sorting(sample_analysis_data):
    """Test that data can be sorted by various metrics."""
    df = prepare_dataframe(sample_analysis_data)

    # Sort by VMAF descending
    sorted_vmaf = df.sort_values("vmaf_mean", ascending=False)
    assert sorted_vmaf.iloc[0]["vmaf_mean"] == 95.5
    assert sorted_vmaf.iloc[-1]["vmaf_mean"] == 92.0

    # Sort by file size ascending
    sorted_size = df.sort_values("file_size_mb", ascending=True)
    assert sorted_size.iloc[0]["file_size_mb"] < sorted_size.iloc[-1]["file_size_mb"]


def test_prepare_dataframe_grouping(sample_analysis_data):
    """Test that data can be grouped by parameters."""
    df = prepare_dataframe(sample_analysis_data)

    # Group by preset
    preset_stats = df.groupby("preset")["vmaf_mean"].mean()
    assert 6 in preset_stats.index
    assert 8 in preset_stats.index

    # Check preset 6 average (should be mean of 95.5 and 92.0)
    assert abs(preset_stats[6] - 93.75) < 0.01


def test_prepare_dataframe_empty_encodings():
    """Test handling of empty encodings list."""
    empty_data = {
        "study_name": "empty_study",
        "analysis_date": "2026-01-30T00:00:00Z",
        "metrics_calculated": ["vmaf"],
        "vmaf_model": "vmaf_v0.6.1neg",
        "clips_analyzed": 0,
        "total_encodings_analyzed": 0,
        "encodings": [],
    }

    df = prepare_dataframe(empty_data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_prepare_dataframe_missing_optional_metrics(sample_analysis_data):
    """Test handling when optional metrics are missing."""
    # Remove PSNR and SSIM from first encoding
    sample_analysis_data["encodings"][0]["metrics"].pop("psnr", None)
    sample_analysis_data["encodings"][0]["metrics"].pop("ssim", None)

    df = prepare_dataframe(sample_analysis_data)

    # Should still work, with NaN for missing values
    assert len(df) == 3
    assert pd.isna(df.iloc[0]["psnr_avg"])
    assert pd.isna(df.iloc[0]["ssim_avg"])
    # But VMAF should still be there
    assert df.iloc[0]["vmaf_mean"] == 95.5


def test_prepare_dataframe_efficiency_calculations(sample_analysis_data):
    """Test that efficiency metrics are correctly calculated."""
    df = prepare_dataframe(sample_analysis_data)

    # Check efficiency metrics exist
    assert "vmaf_per_bpp" in df.columns
    assert "vmaf_per_time" in df.columns
    assert "vmaf_per_bpp_per_time" in df.columns

    # Verify vmaf_per_time is calculated correctly
    # vmaf_per_time = vmaf_mean / encoding_time_s
    first = df.iloc[0]
    expected_vmaf_per_time = first["vmaf_mean"] / first["encoding_time_s"]
    assert abs(first["vmaf_per_time"] - expected_vmaf_per_time) < 0.01


def test_dataframe_can_export_csv(sample_analysis_data, tmp_path):
    """Test that the dataframe can be exported to CSV."""
    df = prepare_dataframe(sample_analysis_data)

    # Export to CSV
    csv_path = tmp_path / "test_export.csv"
    df.to_csv(csv_path, index=False)

    # Read back and verify
    df_loaded = pd.read_csv(csv_path)
    assert len(df_loaded) == 3
    assert "vmaf_mean" in df_loaded.columns
    assert df_loaded.iloc[0]["vmaf_mean"] == 95.5
