#!/usr/bin/env python3
"""
Encode test clips according to a study configuration.

This script reads a study configuration file that defines parameter sweeps
(e.g., preset, CRF, film grain settings) and encodes all clips in the test_clips
directory using those parameters. It tracks encoding time, resource usage, and
outputs structured metadata.

Usage:
    python encode_study.py config/studies/baseline_sweep.json
    python encode_study.py config/studies/film_grain.json --continue-on-error
    python encode_study.py config/studies/baseline_sweep.json --dry-run
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from utils import calculate_sha256


def get_system_info() -> dict[str, Any]:
    """Gather system information for reproducibility."""
    info: dict[str, Any] = {
        "os": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "cpu_cores": os.cpu_count(),
    }

    # Get memory info (Linux)
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["memory_gb"] = round(kb / 1024 / 1024, 2)
                    break
    except (FileNotFoundError, ValueError):
        info["memory_gb"] = None

    # Get SVT-AV1 version
    try:
        result = subprocess.run(
            ["SvtAv1EncApp", "--version"], capture_output=True, text=True, timeout=5
        )
        # Parse version from output
        for line in result.stdout.split("\n"):
            if "SVT" in line or "version" in line.lower():
                info["svt_av1_version"] = line.strip()
                break
        else:
            info["svt_av1_version"] = "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info["svt_av1_version"] = "not found"

    # Get FFmpeg version
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        first_line = result.stdout.split("\n")[0]
        info["ffmpeg_version"] = first_line.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info["ffmpeg_version"] = "not found"

    return info


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


def normalize_param_value(value: Any) -> list[Any]:
    """Convert single value or array to list for uniform handling."""
    if isinstance(value, list):
        return value
    return [value]


def generate_param_combinations(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all combinations of parameters from the study config.

    Parameters are ordered to vary 'preset' in the inner loop (fastest changing)
    and 'crf' in the outer loop (slowest changing). This provides better ETA
    estimates by averaging over different preset speeds rather than completing
    all CRFs for one preset before moving to the next.

    Args:
        params: Parameter dict where values can be single values or lists

    Returns:
        List of parameter dicts, each representing one encoding configuration
    """
    # Normalize all params to lists
    normalized = {k: normalize_param_value(v) for k, v in params.items()}

    # Order parameters: CRF first (outer loop), preset last (inner loop)
    # This makes preset vary fastest, providing better ETA by averaging speeds
    param_names = list(normalized.keys())

    # Reorder: move 'preset' to end and 'crf' to beginning if they exist
    if "preset" in param_names and "crf" in param_names:
        param_names.remove("preset")
        param_names.remove("crf")
        # CRF first, then other params, then preset last
        param_names = ["crf", *param_names, "preset"]
    elif "preset" in param_names:
        # Just move preset to end
        param_names.remove("preset")
        param_names.append("preset")
    elif "crf" in param_names:
        # Just move CRF to beginning
        param_names.remove("crf")
        param_names.insert(0, "crf")

    param_values = [normalized[name] for name in param_names]

    # Generate cartesian product of all parameter values
    # The rightmost parameter changes fastest (inner loop)
    combinations = []
    for values in product(*param_values):
        combo = dict(zip(param_names, values, strict=True))
        combinations.append(combo)

    return combinations


def build_output_filename(clip_name: str, params: dict[str, Any]) -> str:
    """Build output filename from clip name and encoding parameters.

    Format: {clip_stem}_p{preset}_crf{crf}[_fg{film_grain}][_fgd{denoise}][_scm{scm}]...mkv
    """
    clip_stem = Path(clip_name).stem
    parts = [clip_stem, f"p{params['preset']}", f"crf{params['crf']}"]

    # Add optional parameters in consistent order
    optional_params = [
        ("film_grain", "fg"),
        ("film_grain_denoise", "fgd"),
        ("scm", "scm"),
        ("tune", "tune"),
        ("tile_rows", "tr"),
        ("tile_columns", "tc"),
        ("enable_qm", "qm"),
    ]

    for param_name, short_name in optional_params:
        if param_name in params:
            parts.append(f"{short_name}{params[param_name]}")

    return "_".join(parts) + ".mkv"


def build_ffmpeg_command(input_file: Path, output_file: Path, params: dict[str, Any]) -> list[str]:
    """Build FFmpeg command with SVT-AV1 encoding parameters."""
    cmd = [
        "ffmpeg",
        "-i",
        str(input_file),
        "-c:v",
        "libsvtav1",
        "-preset",
        str(params["preset"]),
        "-crf",
        str(params["crf"]),
    ]

    # Add optional parameters

    # Build svtav1-params string
    svt_params = []

    if "film_grain" in params:
        svt_params.append(f"film-grain={params['film_grain']}")

    if "film_grain_denoise" in params:
        svt_params.append(f"film-grain-denoise={params['film_grain_denoise']}")

    if "scm" in params:
        svt_params.append(f"scm={params['scm']}")

    if "tune" in params:
        svt_params.append(f"tune={params['tune']}")

    if "tile_rows" in params:
        svt_params.append(f"tile-rows={params['tile_rows']}")

    if "tile_columns" in params:
        svt_params.append(f"tile-columns={params['tile_columns']}")

    if "enable_qm" in params:
        svt_params.append(f"enable-qm={params['enable_qm']}")

    if svt_params:
        cmd.extend(["-svtav1-params", ":".join(svt_params)])

    # Remove audio streams (we only need video for quality analysis)
    # This reduces file size and ensures accurate video-only bitrate calculation
    cmd.extend(["-an"])

    # Overwrite output file
    cmd.extend(["-y", str(output_file)])

    return cmd


def encode_clip(
    clip_path: Path, output_path: Path, params: dict[str, Any], verbose: bool = False
) -> dict[str, Any]:
    """Encode a single clip with specified parameters.

    Returns:
        Dict with encoding results including timing, file size, checksum
    """
    cmd = build_ffmpeg_command(clip_path, output_path, params)

    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    result = {
        "output_file": output_path.name,
        "source_clip": clip_path.name,
        "parameters": params.copy(),
        "success": False,
        "error": None,
    }

    # Run encoding and measure time
    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per clip
        )

        end_time = time.time()
        result["encoding_time_seconds"] = round(end_time - start_time, 2)

        if proc.returncode != 0:
            result["error"] = f"FFmpeg exited with code {proc.returncode}"
            if verbose:
                print(f"  STDERR: {proc.stderr}")
            return result

        # Parse encoding stats from FFmpeg output
        result["fps"] = parse_fps_from_output(proc.stderr)

        # Get output file info
        result["file_size_bytes"] = output_path.stat().st_size
        result["sha256"] = calculate_sha256(output_path)

        # Duration will be calculated during analysis phase
        result["duration_seconds"] = None
        # Bitrate will be calculated during analysis phase
        result["bitrate_kbps"] = None

        result["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        result["success"] = True

    except subprocess.TimeoutExpired:
        result["error"] = "Encoding timeout (>1 hour)"
        result["encoding_time_seconds"] = 3600
    except Exception as e:
        result["error"] = str(e)
        result["encoding_time_seconds"] = round(time.time() - start_time, 2)

    # Set null for unavailable metrics
    result["cpu_time_seconds"] = None
    result["peak_memory_mb"] = None

    return result


def parse_fps_from_output(stderr: str) -> float | None:
    """Parse encoding FPS from FFmpeg stderr output."""
    # Look for line like: "frame= 1234 fps=45.6 ..."
    for line in stderr.split("\n"):
        if "fps=" in line:
            try:
                parts = line.split("fps=")
                if len(parts) > 1:
                    fps_str = parts[1].split()[0]
                    return float(fps_str)
            except (ValueError, IndexError):
                continue
    return None


def load_study_config(config_path: Path) -> dict[str, Any]:
    """Load and validate study configuration."""
    with open(config_path) as f:
        config: dict[str, Any] = json.load(f)

    # Basic validation
    required_fields = ["study_name", "description", "parameters"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Study config missing required field: {field}")

    required_params = ["preset", "crf"]
    for param in required_params:
        if param not in config["parameters"]:
            raise ValueError(f"Study config missing required parameter: {param}")

    return config


def find_clips(clips_dir: Path) -> list[Path]:
    """Find all video clips in the clips directory."""
    extensions = [".mp4", ".mkv", ".webm", ".mov", ".avi"]
    clips: list[Path] = []

    for ext in extensions:
        clips.extend(clips_dir.glob(f"*{ext}"))

    # Filter out any files starting with '.'
    clips = [c for c in clips if not c.name.startswith(".")]

    return sorted(clips)


def main():
    parser = argparse.ArgumentParser(
        description="Encode test clips according to study configuration"
    )
    parser.add_argument("study_config", type=Path, help="Path to study configuration JSON file")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue encoding remaining clips if one fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be encoded without actually encoding",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed encoding commands and output"
    )

    args = parser.parse_args()

    # Load study config
    if not args.study_config.exists():
        print(f"Error: Study config not found: {args.study_config}")
        sys.exit(1)

    try:
        config = load_study_config(args.study_config)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading study config: {e}")
        sys.exit(1)

    # Setup paths
    project_root = Path(__file__).parent.parent
    clips_dir = project_root / config.get("clips_dir", "data/test_clips")

    output_dir_name = config.get("output_dir") or config["study_name"]
    output_dir = project_root / "data" / "encoded" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find clips
    clips = find_clips(clips_dir)
    if not clips:
        print(f"Error: No clips found in {clips_dir}")
        print("Run 'just extract-clips' first to create test clips.")
        sys.exit(1)

    print(f"Study: {config['study_name']}")
    print(f"Description: {config['description']}")
    print(f"Clips directory: {clips_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(clips)} clips")

    # Generate parameter combinations
    param_combos = generate_param_combinations(config["parameters"])
    total_encodings = len(clips) * len(param_combos)

    print(f"Parameter combinations: {len(param_combos)}")
    print(f"Total encodings: {total_encodings}")

    if args.dry_run:
        print("\nDRY RUN - Would encode:")
        for clip in clips[:3]:  # Show first 3 clips as example
            print(f"\n  Clip: {clip.name}")
            for params in param_combos[:3]:  # Show first 3 param combos
                output_name = build_output_filename(clip.name, params)
                print(f"    → {output_name}")
                print(f"       params: {params}")
            if len(param_combos) > 3:
                print(f"    ... and {len(param_combos) - 3} more parameter combinations")

        if len(clips) > 3:
            print(f"\n  ... and {len(clips) - 3} more clips")

        print(f"\nTotal: {total_encodings} encodings")
        return

    # Gather system info
    print("\nGathering system information...")
    system_info = get_system_info()
    if args.verbose:
        print(f"  CPU: {system_info['cpu']}")
        print(f"  Cores: {system_info['cpu_cores']}")
        print(f"  Memory: {system_info['memory_gb']} GB")
        print(f"  SVT-AV1: {system_info['svt_av1_version']}")
        print(f"  FFmpeg: {system_info['ffmpeg_version']}")

    # Initialize metadata
    metadata: dict[str, Any] = {
        "study_config": config,
        "system_info": system_info,
        "start_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "end_time": None,
        "encodings": [],
    }

    # Calculate work units for progress tracking
    print("\nCalculating encoding workload...")
    clip_work_units = {}
    total_work_units = 0

    for clip in clips:
        video_info = get_video_info(clip)
        if video_info:
            work_units = calculate_work_units(video_info)
            clip_work_units[clip.name] = work_units
            total_work_units += work_units * len(
                param_combos
            )  # Each param combo processes the clip
            if args.verbose:
                print(
                    f"  {clip.name}: {work_units:,} work units ({video_info['frame_count']} frames, {video_info['width']}x{video_info['height']})"
                )
        else:
            # Fallback: assume average complexity
            clip_work_units[clip.name] = 1_000_000
            total_work_units += 1_000_000 * len(param_combos)

    print(f"Total workload: {total_work_units:,} work units across {total_encodings} encodings")

    # Encode all clips with all parameter combinations
    print("\nStarting encoding...")
    completed = 0
    failed = 0
    completed_work_units = 0
    overall_start_time = time.time()

    for clip_idx, clip in enumerate(clips, 1):
        clip_work = clip_work_units.get(clip.name, 1_000_000)
        print(f"\n[Clip {clip_idx}/{len(clips)}] {clip.name}")

        for param_idx, params in enumerate(param_combos, 1):
            output_name = build_output_filename(clip.name, params)
            output_path = output_dir / output_name

            param_str = ", ".join(f"{k}={v}" for k, v in params.items())

            # Calculate and display progress

            progress_pct = (
                (completed_work_units / total_work_units * 100) if total_work_units > 0 else 0
            )
            elapsed = time.time() - overall_start_time

            # Estimate time remaining based on work units completed
            if completed_work_units > 0:
                avg_time_per_unit = elapsed / completed_work_units
                remaining_units = total_work_units - completed_work_units
                eta_seconds = avg_time_per_unit * remaining_units
                eta_str = format_time_remaining(eta_seconds)
                progress_str = f"[{progress_pct:.1f}%, ETA: {eta_str}]"
            else:
                progress_str = "[0.0%]"

            print(
                f"  [{param_idx}/{len(param_combos)}] {progress_str} {param_str}... ",
                end="",
                flush=True,
            )

            result = encode_clip(clip, output_path, params, verbose=args.verbose)
            metadata["encodings"].append(result)

            if result["success"]:
                print(
                    f"✓ ({result['encoding_time_seconds']:.1f}s, "
                    f"{result['file_size_bytes'] / 1024 / 1024:.2f} MB)"
                )
                completed += 1
                completed_work_units += clip_work
            else:
                print(f"✗ {result['error']}")
                failed += 1
                completed_work_units += clip_work  # Count failed ones too for progress tracking

                if not args.continue_on_error:
                    print(
                        "\nEncoding failed. Use --continue-on-error to continue despite failures."
                    )
                    break

        if failed > 0 and not args.continue_on_error:
            break

    # Finalize metadata
    metadata["end_time"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    metadata["summary"] = {
        "total_encodings": total_encodings,
        "successful_encodings": completed,
        "failed_encodings": failed,
        "total_time_seconds": sum(e["encoding_time_seconds"] for e in metadata["encodings"]),
        "total_output_size_bytes": sum(
            e.get("file_size_bytes", 0) for e in metadata["encodings"] if e["success"]
        ),
    }

    # Save metadata
    metadata_path = output_dir / "encoding_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Study complete!")
    print(f"  Successful: {completed}/{total_encodings}")
    print(f"  Failed: {failed}/{total_encodings}")
    print(f"  Total time: {metadata['summary']['total_time_seconds']:.1f}s")
    print(
        f"  Total output size: {metadata['summary']['total_output_size_bytes'] / 1024 / 1024 / 1024:.2f} GB"
    )
    print(f"  Metadata: {metadata_path}")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
