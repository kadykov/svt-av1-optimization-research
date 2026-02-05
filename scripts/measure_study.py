#!/usr/bin/env python3
"""
Measure quality metrics for encoded videos from a study.

This script calculates VMAF (NEG mode), PSNR, and SSIM for all encodings
in a study by comparing them against the original source clips. Results
are stored in measurements.json alongside the encoding metadata.

This is the "measurement" phase of the workflow:
  1. encode_study.py - Encode clips with various parameters
  2. measure_study.py - Calculate quality metrics (VMAF, PSNR, SSIM) [THIS SCRIPT]
  3. analyze_study.py - Analyze results and generate visualizations

Quality Metrics:
- VMAF NEG: Netflix's perceptual quality metric (No Enhancement Gain mode)
  * NEG mode disables enhancement gain, making it ideal for codec evaluation
  * Industry standard for measuring compression quality
- PSNR: Peak Signal-to-Noise Ratio (traditional pixel difference metric)
- SSIM: Structural Similarity Index (perceptual similarity metric)

Usage:
    python measure_study.py baseline_sweep
    python measure_study.py baseline_sweep --metrics vmaf
    python measure_study.py film_grain --continue-on-error
    python measure_study.py baseline_sweep --threads 8 -v
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils import get_video_bitrate


def load_encoding_metadata(study_dir: Path) -> dict[str, Any]:
    """Load encoding metadata for a study."""
    metadata_file = study_dir / "encoding_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Encoding metadata not found: {metadata_file}\n"
            f"Run encoding study first: just encode-study {study_dir.name}"
        )

    with open(metadata_file) as f:
        data: dict[str, Any] = json.load(f)
        return data


def count_video_frames(video_path: Path) -> int | None:
    """Count actual frames in a video by decoding (more reliable than nb_frames metadata).

    This is slower than reading nb_frames metadata but more accurate, especially
    for clips extracted with stream copy that may have incorrect metadata.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def find_source_clip(clip_name: str, clips_dir: Path) -> Path | None:
    """Find the source clip file in the clips directory."""
    # Try exact match first
    clip_path = clips_dir / clip_name
    if clip_path.exists():
        return clip_path

    # Try with common extensions
    for ext in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
        clip_path = clips_dir / f"{Path(clip_name).stem}{ext}"
        if clip_path.exists():
            return clip_path

    return None


def get_video_info(video_path: Path) -> dict[str, Any] | None:
    """Get video information (duration, frame count, resolution) using FFprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate:format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        stream = data.get("streams", [{}])[0]
        format_info = data.get("format", {})

        # Parse frame rate (e.g., "30/1" -> 30.0)
        frame_rate_str = stream.get("r_frame_rate", "0/1")
        num, denom = map(int, frame_rate_str.split("/"))
        frame_rate = num / denom if denom > 0 else 0

        # Get frame count (may not be available for all formats)
        nb_frames = stream.get("nb_frames")
        if nb_frames:
            frame_count = int(nb_frames)
        else:
            # Estimate from duration and frame rate
            duration = float(format_info.get("duration", 0))
            frame_count = int(duration * frame_rate) if duration > 0 else 0

        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "frame_count": frame_count,
            "frame_rate": frame_rate,
            "duration": float(format_info.get("duration", 0)),
        }
    except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError):
        return None


def get_video_duration(video_path: Path) -> float | None:
    """Get video duration in seconds using FFprobe (simplified version)."""
    info = get_video_info(video_path)
    return info["duration"] if info else None


def calculate_work_units(video_info: dict[str, Any]) -> int:
    """Calculate work units for a video based on frame count and resolution.

    Work units = frame_count * width * height
    This gives a rough measure of encoding/analysis complexity.
    """
    if not video_info:
        return 0

    frame_count = video_info.get("frame_count", 0)
    width = video_info.get("width", 0)
    height = video_info.get("height", 0)

    return int(frame_count * width * height)


def format_time_remaining(seconds: float) -> str:
    """Format seconds into human-readable time (e.g., '2h 15m', '45s')."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    return f"{hours}h {minutes}m"


def calculate_vmaf(
    reference: Path,
    distorted: Path,
    model: str = "version=vmaf_v0.6.1neg",
    threads: int = 4,
    verbose: bool = False,
) -> dict[str, float] | None:
    """
    Calculate VMAF score using FFmpeg libvmaf filter.

    Args:
        reference: Path to original/reference video
        distorted: Path to encoded/distorted video
        model: VMAF model to use (default: version=vmaf_v0.6.1neg for NEG mode)
        threads: Number of threads for VMAF calculation
        verbose: Show FFmpeg output

    Returns:
        Dictionary with VMAF statistics or None on error
    """
    # Create temporary file for VMAF JSON output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        log_file = Path(tmp.name)

    try:
        # Build FFmpeg command for VMAF calculation
        # Format: ffmpeg -i distorted -i reference -lavfi libvmaf -f null -
        # NOTE: We use setpts=PTS-STARTPTS to reset timestamps to 0 for both streams.
        # This is critical because:
        # 1. Source clips extracted with -c copy may have non-zero start times
        # 2. Encoded files may have different start times than sources
        # 3. Without alignment, libvmaf compares wrong frames, giving incorrect scores
        filter_complex = (
            f"[0:v]setpts=PTS-STARTPTS[dist];"
            f"[1:v]setpts=PTS-STARTPTS[ref];"
            f"[dist][ref]libvmaf=model={model}:log_path={log_file}:log_fmt=json:n_threads={threads}"
        )
        cmd = [
            "ffmpeg",
            "-i",
            str(distorted),
            "-i",
            str(reference),
            "-lavfi",
            filter_complex,
            "-f",
            "null",
            "-",
        ]

        if verbose:
            print(f"  Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )

        if result.returncode != 0:
            print("  ERROR: VMAF calculation failed")
            if verbose:
                print(f"  stderr: {result.stderr}")
            return None

        # Parse VMAF JSON output
        with open(log_file) as f:
            vmaf_data = json.load(f)

        # Extract frame scores
        frames = vmaf_data.get("frames", [])
        if not frames:
            print("  ERROR: No VMAF frames found in output")
            return None

        scores = [frame["metrics"]["vmaf"] for frame in frames]

        # Calculate statistics
        scores_sorted = sorted(scores)
        n = len(scores)

        mean = sum(scores) / n

        stats = {
            "mean": mean,
            "harmonic_mean": n / sum(1 / s if s > 0 else 0 for s in scores),
            "min": min(scores),
            "max": max(scores),
            "percentile_1": scores_sorted[int(n * 0.01)],
            "percentile_5": scores_sorted[int(n * 0.05)],
            "percentile_25": scores_sorted[int(n * 0.25)],
            "percentile_50": scores_sorted[int(n * 0.50)],
            "percentile_75": scores_sorted[int(n * 0.75)],
            "percentile_95": scores_sorted[int(n * 0.95)],
            "std_dev": (sum((s - mean) ** 2 for s in scores) / n) ** 0.5,
        }

        # Round to 2 decimal places
        stats = {k: round(v, 2) for k, v in stats.items()}

        return stats

    except subprocess.TimeoutExpired:
        print("  ERROR: VMAF calculation timed out")
        return None
    except Exception as e:
        print(f"  ERROR: VMAF calculation failed: {e}")
        return None
    finally:
        # Clean up temporary file
        if log_file.exists():
            log_file.unlink()


def calculate_psnr_ssim(
    reference: Path,
    distorted: Path,
    calculate_psnr: bool = True,
    calculate_ssim: bool = True,
    verbose: bool = False,
) -> tuple[dict | None, dict | None]:
    """
    Calculate PSNR and/or SSIM using FFmpeg filters.

    Returns:
        Tuple of (psnr_stats, ssim_stats), each can be None if not calculated
    """
    if not calculate_psnr and not calculate_ssim:
        return None, None

    # Build filter chain with timestamp alignment
    # NOTE: We use setpts=PTS-STARTPTS to reset timestamps to 0 for both streams.
    # This ensures proper frame-to-frame comparison even when source clips have
    # non-zero start times (common with -c copy extraction).
    #
    # When calculating both PSNR and SSIM, we must use split filters to provide
    # separate copies of each stream to each metric filter, since each filter
    # consumes its input streams.
    if calculate_psnr and calculate_ssim:
        # Need to split both streams for both metrics
        filter_chain = (
            "[0:v]setpts=PTS-STARTPTS,split=2[dist1][dist2];"
            "[1:v]setpts=PTS-STARTPTS,split=2[ref1][ref2];"
            "[dist1][ref1]psnr=stats_file=-;"
            "[dist2][ref2]ssim=stats_file=-"
        )
    elif calculate_psnr:
        filter_chain = (
            "[0:v]setpts=PTS-STARTPTS[dist];"
            "[1:v]setpts=PTS-STARTPTS[ref];"
            "[dist][ref]psnr=stats_file=-"
        )
    else:  # calculate_ssim only
        filter_chain = (
            "[0:v]setpts=PTS-STARTPTS[dist];"
            "[1:v]setpts=PTS-STARTPTS[ref];"
            "[dist][ref]ssim=stats_file=-"
        )

    try:
        cmd = [
            "ffmpeg",
            "-i",
            str(distorted),
            "-i",
            str(reference),
            "-lavfi",
            filter_chain,
            "-f",
            "null",
            "-",
        ]

        if verbose:
            print(f"  Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            print("  ERROR: PSNR/SSIM calculation failed")
            if verbose:
                print(f"  stderr: {result.stderr}")
            return None, None

        # Parse PSNR and SSIM from stderr
        psnr_stats = None
        ssim_stats = None

        for line in result.stderr.split("\n"):
            if calculate_psnr and "PSNR" in line and "average:" in line:
                # Example: [Parsed_psnr_0 @ 0x...] PSNR y:42.3 u:45.1 v:44.8 average:42.9 min:38.5 max:48.2
                try:
                    parts = line.split("PSNR")[1].strip()
                    values = {}
                    for part in parts.split():
                        if ":" in part:
                            k, v = part.split(":")
                            values[k] = float(v)

                    psnr_stats = {
                        "y_mean": round(values.get("y", 0), 2),
                        "u_mean": round(values.get("u", 0), 2),
                        "v_mean": round(values.get("v", 0), 2),
                        "avg_mean": round(values.get("average", 0), 2),
                    }
                except (ValueError, KeyError) as e:
                    if verbose:
                        print(f"  Warning: Failed to parse PSNR: {e}")

            if calculate_ssim and "SSIM" in line and "All:" in line and "[Parsed_ssim" in line:
                # Example: [Parsed_ssim_0 @ 0x...] SSIM Y:0.982 U:0.991 V:0.990 All:0.985 (18.234)
                # NOTE: We must check for "[Parsed_ssim" to avoid matching per-frame output lines
                # which also contain "All:" but have format like "n:450 Y:0.967 ... All:0.968"
                try:
                    parts = line.split("SSIM")[1].strip()
                    values = {}
                    for part in parts.split():
                        if ":" in part:
                            k, v = part.split(":")
                            # Remove parentheses if present
                            v = v.strip("()")
                            with contextlib.suppress(ValueError):
                                values[k] = float(v)

                    ssim_stats = {
                        "y_mean": round(values.get("Y", 0), 4),
                        "u_mean": round(values.get("U", 0), 4),
                        "v_mean": round(values.get("V", 0), 4),
                        "avg_mean": round(values.get("All", 0), 4),
                    }
                except (ValueError, KeyError) as e:
                    if verbose:
                        print(f"  Warning: Failed to parse SSIM: {e}")

        return psnr_stats, ssim_stats

    except subprocess.TimeoutExpired:
        print("  ERROR: PSNR/SSIM calculation timed out")
        return None, None
    except Exception as e:
        print(f"  ERROR: PSNR/SSIM calculation failed: {e}")
        return None, None


def analyze_encoding(
    encoding: dict[str, Any],
    study_dir: Path,
    clips_dir: Path,
    metrics: list[str],
    threads: int,
    verbose: bool,
) -> dict[str, Any]:
    """Analyze a single encoding and return results."""
    output_file = encoding["output_file"]
    source_clip_name = encoding["source_clip"]

    print(f"\nAnalyzing: {output_file}")
    print(f"  Source: {source_clip_name}")

    # Find source clip
    source_clip = find_source_clip(source_clip_name, clips_dir)
    if source_clip is None:
        return {
            "output_file": output_file,
            "source_clip": source_clip_name,
            "parameters": encoding["parameters"],
            "metrics": {},
            "success": False,
            "error": f"Source clip not found: {source_clip_name}",
        }

    # Check if encoded file exists
    encoded_file = study_dir / output_file
    if not encoded_file.exists():
        return {
            "output_file": output_file,
            "source_clip": source_clip_name,
            "parameters": encoding["parameters"],
            "metrics": {},
            "success": False,
            "error": f"Encoded file not found: {encoded_file}",
        }

    # Validate that source and encoded files have compatible durations
    source_duration = get_video_duration(source_clip)
    encoded_duration = get_video_duration(encoded_file)

    if source_duration is None:
        return {
            "output_file": output_file,
            "source_clip": source_clip_name,
            "parameters": encoding["parameters"],
            "metrics": {},
            "success": False,
            "error": f"Could not determine source clip duration: {source_clip.name}",
        }

    if encoded_duration is None:
        return {
            "output_file": output_file,
            "source_clip": source_clip_name,
            "parameters": encoding["parameters"],
            "metrics": {},
            "success": False,
            "error": f"Could not determine encoded file duration: {output_file}",
        }

    # Check for duration mismatch (allow 1% tolerance for codec overhead)
    duration_tolerance = 1.0  # 1 second tolerance
    if abs(source_duration - encoded_duration) > duration_tolerance:
        print(
            f"  ⚠️  Duration mismatch detected:"
            f" source={source_duration:.3f}s, encoded={encoded_duration:.3f}s"
        )
        if source_duration > encoded_duration:
            print(
                "  ⚠️  Encoded file is shorter than source. "
                "This may indicate incorrect clip extraction or encoding."
            )

    # Get frame counts for validation (helps detect B-frame ordering issues)
    source_info = get_video_info(source_clip)
    encoded_info = get_video_info(encoded_file)
    source_frames = source_info.get("frame_count", 0) if source_info else 0
    encoded_frames = encoded_info.get("frame_count", 0) if encoded_info else 0

    if source_frames > 0 and encoded_frames > 0:
        frame_diff = abs(source_frames - encoded_frames)
        if frame_diff > 1:  # Allow 1 frame tolerance
            print(
                f"  ⚠️  Frame count mismatch: source={source_frames}, encoded={encoded_frames} "
                f"(diff={frame_diff})"
            )
            print(
                "  ⚠️  This may cause inaccurate VMAF scores. Consider re-extracting clips "
                "with lossless encoding (without --fast flag)."
            )

    start_time = time.time()

    # Calculate bitrate from encoded file (video stream only)
    bitrate_kbps = get_video_bitrate(encoded_file)
    if bitrate_kbps is None:
        print("  Warning: Could not determine video bitrate")

    # Get video info for the encoded file
    encoded_info = get_video_info(encoded_file)

    # Check duration match
    duration_tolerance = 1.0  # 1 second tolerance
    duration_match = abs(source_duration - encoded_duration) <= duration_tolerance

    # Build result - only store measurement-related data
    # Encoding parameters and file info are in encoding_metadata.json
    result = {
        "output_file": output_file,
        "source_clip": source_clip_name,
        "parameters": encoding["parameters"],  # Keep for convenience
        "metrics": {},
        "video_info": {
            "duration_seconds": encoded_duration,
            "width": encoded_info.get("width") if encoded_info else None,
            "height": encoded_info.get("height") if encoded_info else None,
            "frame_count": encoded_info.get("frame_count") if encoded_info else None,
            "frame_rate": encoded_info.get("frame_rate") if encoded_info else None,
            "bitrate_kbps": bitrate_kbps,
        },
        "duration_match": duration_match,
        "success": True,
    }

    # VMAF
    if "vmaf" in metrics or "vmaf_neg" in metrics:
        print("  Calculating VMAF (NEG mode)...")
        vmaf_model = "version=vmaf_v0.6.1neg"
        vmaf_stats = calculate_vmaf(
            reference=source_clip,
            distorted=encoded_file,
            model=vmaf_model,
            threads=threads,
            verbose=verbose,
        )

        if vmaf_stats:
            result["metrics"]["vmaf"] = vmaf_stats
            print(
                f"    Mean: {vmaf_stats['mean']:.2f}, "
                f"Harmonic: {vmaf_stats['harmonic_mean']:.2f}, "
                f"Min: {vmaf_stats['min']:.2f}"
            )

            # Warn about potential frame alignment issues
            # High std_dev with low min scores often indicates B-frame ordering problems
            if vmaf_stats["std_dev"] > 25 and vmaf_stats["min"] < 10:
                print("  ⚠️  WARNING: High VMAF variance with very low minimum scores detected!")
                print("      This typically indicates frame misalignment (comparing wrong frames).")
                print(
                    "      Consider re-extracting source clips with lossless encoding "
                    "(without --fast flag)."
                )
                result["frame_alignment_warning"] = True
        else:
            result["success"] = False
            result["error"] = "VMAF calculation failed"

    # PSNR and SSIM
    calculate_psnr = "psnr" in metrics
    calculate_ssim = "ssim" in metrics

    if calculate_psnr or calculate_ssim:
        if calculate_psnr:
            print("  Calculating PSNR...")
        if calculate_ssim:
            print("  Calculating SSIM...")

        psnr_stats, ssim_stats = calculate_psnr_ssim(
            reference=source_clip,
            distorted=encoded_file,
            calculate_psnr=calculate_psnr,
            calculate_ssim=calculate_ssim,
            verbose=verbose,
        )

        if psnr_stats:
            result["metrics"]["psnr"] = psnr_stats
            print(
                f"    PSNR Y: {psnr_stats['y_mean']:.2f} dB, Avg: {psnr_stats['avg_mean']:.2f} dB"
            )

        if ssim_stats:
            result["metrics"]["ssim"] = ssim_stats
            print(f"    SSIM Y: {ssim_stats['y_mean']:.4f}, Avg: {ssim_stats['avg_mean']:.4f}")

    measurement_time = time.time() - start_time
    result["measurement_time_seconds"] = round(measurement_time, 2)
    print(f"  Measurement time: {measurement_time:.1f}s")

    return result


def calculate_summary(measurements: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Calculate summary statistics across all measurements."""
    # Convert dict to list with output_file included for processing
    encodings = [{"output_file": k, **v} for k, v in measurements.items()]
    successful = [e for e in encodings if e.get("success")]

    if not successful:
        return {}

    vmaf_means = [
        e["metrics"].get("vmaf", {}).get("mean") for e in successful if "vmaf" in e["metrics"]
    ]
    bitrates = [
        e["video_info"]["bitrate_kbps"]
        for e in successful
        if e.get("video_info", {}).get("bitrate_kbps")
    ]

    summary = {
        "total_measurement_time_seconds": sum(
            e.get("measurement_time_seconds", 0) for e in encodings
        )
    }

    if vmaf_means:
        summary["vmaf_range"] = {
            "min_mean": round(min(vmaf_means), 2),
            "max_mean": round(max(vmaf_means), 2),
        }

    if bitrates:
        summary["bitrate_range_kbps"] = {
            "min": round(min(bitrates), 2),
            "max": round(max(bitrates), 2),
        }

    # Find best efficiency (VMAF per kbps) - Note: this uses legacy metric
    # Full efficiency metrics are calculated in the analysis phase
    with_efficiency = [
        e
        for e in successful
        if e.get("video_info", {}).get("bitrate_kbps") and e.get("metrics", {}).get("vmaf")
    ]
    if with_efficiency:
        # Calculate efficiency inline
        for e in with_efficiency:
            vmaf = e["metrics"]["vmaf"]["mean"]
            bitrate = e["video_info"]["bitrate_kbps"]
            e["_vmaf_per_kbps"] = vmaf / bitrate if bitrate > 0 else 0

        best_eff = max(with_efficiency, key=lambda e: e["_vmaf_per_kbps"])
        summary["best_efficiency"] = {
            "output_file": best_eff["output_file"],
            "parameters": best_eff["parameters"],
            "vmaf_mean": best_eff["metrics"]["vmaf"]["mean"],
            "vmaf_per_kbps": round(best_eff["_vmaf_per_kbps"], 4),
            "bitrate_kbps": best_eff["video_info"]["bitrate_kbps"],
        }

    # Find best quality (highest VMAF)
    if vmaf_means:
        with_vmaf = [e for e in successful if "vmaf" in e["metrics"]]
        best_quality = max(with_vmaf, key=lambda e: e["metrics"]["vmaf"]["mean"])
        summary["best_quality"] = {
            "output_file": best_quality["output_file"],
            "parameters": best_quality["parameters"],
            "vmaf_mean": best_quality["metrics"]["vmaf"]["mean"],
            "bitrate_kbps": best_quality.get("video_info", {}).get("bitrate_kbps"),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze encoded videos from a study by calculating quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s baseline_sweep
  %(prog)s baseline_sweep --metrics vmaf
  %(prog)s film_grain --metrics vmaf psnr ssim
  %(prog)s baseline_sweep --threads 8 -v
  %(prog)s baseline_sweep --continue-on-error
        """,
    )
    parser.add_argument("study_name", help="Name of the study to analyze (e.g., baseline_sweep)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["vmaf", "vmaf_neg", "psnr", "ssim"],
        default=["vmaf", "psnr", "ssim"],
        help="Metrics to calculate (default: vmaf psnr ssim). vmaf and vmaf_neg are equivalent (both use NEG mode)",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("data/test_clips"),
        help="Directory containing source clips (default: data/test_clips)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads for VMAF calculation (default: auto-detect all CPU cores)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue analyzing remaining encodings if one fails",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed FFmpeg output")

    args = parser.parse_args()

    # Auto-detect number of threads if not specified
    if args.threads is None:
        args.threads = os.cpu_count() or 4  # Fallback to 4 if cpu_count() returns None

    # Find study directory
    study_dir = Path("data/encoded") / args.study_name
    if not study_dir.exists():
        print(f"Error: Study directory not found: {study_dir}")
        print("Available studies:")
        encoded_dir = Path("data/encoded")
        if encoded_dir.exists():
            for d in sorted(encoded_dir.iterdir()):
                if d.is_dir() and (d / "encoding_metadata.json").exists():
                    print(f"  {d.name}")
        sys.exit(1)

    # Normalize metrics (vmaf and vmaf_neg are the same)
    metrics = list(set(args.metrics))
    if "vmaf_neg" in metrics:
        metrics.remove("vmaf_neg")
        if "vmaf" not in metrics:
            metrics.append("vmaf")

    print(f"Analyzing study: {args.study_name}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Source clips: {args.clips_dir}")
    print(f"VMAF threads: {args.threads}")

    # Load encoding metadata
    try:
        encoding_metadata = load_encoding_metadata(study_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Support both new format (encodings as dict) and legacy format (encodings as list)
    encodings_raw = encoding_metadata.get("encodings", {})
    if isinstance(encodings_raw, dict):
        # New format: dict keyed by output_file
        encodings_to_analyze = [{"output_file": k, **v} for k, v in encodings_raw.items()]
    else:
        # Legacy format: list of dicts
        encodings_to_analyze = encodings_raw

    successful_encodings = [e for e in encodings_to_analyze if e.get("success")]

    if not successful_encodings:
        print("\nNo successful encodings found in study")
        sys.exit(1)

    print(f"\nEncodings to analyze: {len(successful_encodings)}")

    # Get unique source clips
    unique_clips = {e["source_clip"] for e in successful_encodings}
    print(f"Unique source clips: {len(unique_clips)}")

    # Calculate work units for progress tracking
    print("\nCalculating analysis workload...")
    clip_work_units = {}
    total_work_units = 0

    for clip_name in unique_clips:
        source_clip = find_source_clip(clip_name, args.clips_dir)
        if source_clip:
            video_info = get_video_info(source_clip)
            if video_info:
                work_units = calculate_work_units(video_info)
                clip_work_units[clip_name] = work_units
                if args.verbose:
                    print(
                        f"  {clip_name}: {work_units:,} work units ({video_info['frame_count']} frames, {video_info['width']}x{video_info['height']})"
                    )
            else:
                clip_work_units[clip_name] = 1_000_000
        else:
            clip_work_units[clip_name] = 1_000_000

    # Calculate total work (each encoding processes one clip)
    for encoding in successful_encodings:
        total_work_units += clip_work_units.get(encoding["source_clip"], 1_000_000)

    print(
        f"Total workload: {total_work_units:,} work units across {len(successful_encodings)} encodings"
    )

    # Analyze each encoding
    analysis_results: dict[str, Any] = {}
    failed_count = 0
    completed_work_units = 0
    overall_start_time = time.time()

    for i, encoding in enumerate(successful_encodings, 1):
        # Calculate progress
        progress_pct = (
            (completed_work_units / total_work_units * 100) if total_work_units > 0 else 0
        )
        elapsed = time.time() - overall_start_time

        # Estimate time remaining
        if completed_work_units > 0:
            avg_time_per_unit = elapsed / completed_work_units
            remaining_units = total_work_units - completed_work_units
            eta_seconds = avg_time_per_unit * remaining_units
            eta_str = format_time_remaining(eta_seconds)
            progress_str = f"[{progress_pct:.1f}%, ETA: {eta_str}]"
        else:
            progress_str = "[0.0%]"

        print(f"\n[{i}/{len(successful_encodings)}] {progress_str}", end=" ")

        try:
            result = analyze_encoding(
                encoding=encoding,
                study_dir=study_dir,
                clips_dir=args.clips_dir,
                metrics=metrics,
                threads=args.threads,
                verbose=args.verbose,
            )
            # Store in dict with output_file as key
            output_file = result.pop("output_file")
            analysis_results[output_file] = result

            # Update completed work units
            clip_work = clip_work_units.get(encoding["source_clip"], 1_000_000)
            completed_work_units += clip_work

            if not result["success"]:
                failed_count += 1
                if not args.continue_on_error:
                    print(f"\nError: {result['error']}")
                    print("Use --continue-on-error to continue despite failures")
                    sys.exit(1)

        except KeyboardInterrupt:
            print("\n\nAnalysis interrupted by user")
            sys.exit(1)
        except Exception as e:
            failed_count += 1
            print(f"  ERROR: {e}")
            if not args.continue_on_error:
                raise

    # Calculate summary
    print("\n" + "=" * 70)
    print("Calculating summary statistics...")
    summary = calculate_summary(analysis_results)

    # Save results
    output_file = study_dir / "measurements.json"
    output_data = {
        "study_name": args.study_name,
        "measurement_date": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "metrics_calculated": metrics,
        "vmaf_model": "vmaf_v0.6.1neg",
        "clips_measured": len(unique_clips),
        "total_measurements": len(analysis_results),
        "measurements": analysis_results,
        "summary": summary,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\nMeasurement complete!")
    print(f"  Total measured: {len(analysis_results)}")
    print(f"  Successful: {len(analysis_results) - failed_count}")
    print(f"  Failed: {failed_count}")

    if summary:
        if "vmaf_range" in summary:
            print(
                f"\nVMAF range: {summary['vmaf_range']['min_mean']:.2f} - "
                f"{summary['vmaf_range']['max_mean']:.2f}"
            )

        if "best_efficiency" in summary:
            best = summary["best_efficiency"]
            print("\nBest efficiency (VMAF/kbps):")
            print(f"  File: {best['output_file']}")
            print(f"  Parameters: {best['parameters']}")
            print(f"  VMAF: {best['vmaf_mean']:.2f}")
            if best["bitrate_kbps"] is not None:
                print(f"  Bitrate: {best['bitrate_kbps']:.1f} kbps")
            else:
                print("  Bitrate: N/A")
            print(f"  Efficiency: {best['vmaf_per_kbps']:.4f}")

        if "best_quality" in summary:
            best = summary["best_quality"]
            print("\nBest quality (highest VMAF):")
            print(f"  File: {best['output_file']}")
            print(f"  Parameters: {best['parameters']}")
            print(f"  VMAF: {best['vmaf_mean']:.2f}")
            if best["bitrate_kbps"] is not None:
                print(f"  Bitrate: {best['bitrate_kbps']:.1f} kbps")
            else:
                print("  Bitrate: N/A")

        print(f"\nTotal measurement time: {summary['total_measurement_time_seconds']:.1f}s")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
