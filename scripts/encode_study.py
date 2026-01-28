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
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_system_info() -> Dict[str, Any]:
    """Gather system information for reproducibility."""
    info = {
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
            ["SvtAv1EncApp", "--version"],
            capture_output=True,
            text=True,
            timeout=5
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
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        first_line = result.stdout.split("\n")[0]
        info["ffmpeg_version"] = first_line.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info["ffmpeg_version"] = "not found"
    
    return info


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def normalize_param_value(value: Any) -> List[Any]:
    """Convert single value or array to list for uniform handling."""
    if isinstance(value, list):
        return value
    return [value]


def generate_param_combinations(params: Dict[str, Any]) -> List[Dict[str, int]]:
    """Generate all combinations of parameters from the study config.
    
    Args:
        params: Parameter dict where values can be single values or lists
        
    Returns:
        List of parameter dicts, each representing one encoding configuration
    """
    # Normalize all params to lists
    normalized = {k: normalize_param_value(v) for k, v in params.items()}
    
    # Get all parameter names and their value lists
    param_names = list(normalized.keys())
    param_values = [normalized[name] for name in param_names]
    
    # Generate cartesian product of all parameter values
    combinations = []
    for values in product(*param_values):
        combo = dict(zip(param_names, values))
        combinations.append(combo)
    
    return combinations


def build_output_filename(clip_name: str, params: Dict[str, int]) -> str:
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


def build_ffmpeg_command(
    input_file: Path,
    output_file: Path,
    params: Dict[str, int]
) -> List[str]:
    """Build FFmpeg command with SVT-AV1 encoding parameters."""
    cmd = [
        "ffmpeg",
        "-i", str(input_file),
        "-c:v", "libsvtav1",
        "-preset", str(params["preset"]),
        "-crf", str(params["crf"]),
    ]
    
    # Add optional parameters
    optional_mappings = {
        "film_grain": "-svtav1-params",
        "film_grain_denoise": None,  # Handled with film_grain
        "scm": "-svtav1-params",
        "tune": "-svtav1-params",
        "tile_rows": "-svtav1-params",
        "tile_columns": "-svtav1-params",
        "enable_qm": "-svtav1-params",
    }
    
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
    
    # Copy audio without re-encoding
    cmd.extend(["-c:a", "copy"])
    
    # Overwrite output file
    cmd.extend(["-y", str(output_file)])
    
    return cmd


def encode_clip(
    clip_path: Path,
    output_path: Path,
    params: Dict[str, int],
    verbose: bool = False
) -> Dict[str, Any]:
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
            timeout=3600  # 1 hour timeout per clip
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
        
        # Calculate bitrate (need clip duration - get from clip metadata)
        # For now, leave as null - can be calculated in analysis phase
        result["bitrate_kbps"] = None
        
        result["timestamp"] = datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
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


def parse_fps_from_output(stderr: str) -> Optional[float]:
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


def load_study_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate study configuration."""
    with open(config_path) as f:
        config = json.load(f)
    
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


def find_clips(clips_dir: Path) -> List[Path]:
    """Find all video clips in the clips directory."""
    extensions = [".mp4", ".mkv", ".webm", ".mov", ".avi"]
    clips = []
    
    for ext in extensions:
        clips.extend(clips_dir.glob(f"*{ext}"))
    
    # Filter out any files starting with '.'
    clips = [c for c in clips if not c.name.startswith(".")]
    
    return sorted(clips)


def main():
    parser = argparse.ArgumentParser(
        description="Encode test clips according to study configuration"
    )
    parser.add_argument(
        "study_config",
        type=Path,
        help="Path to study configuration JSON file"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue encoding remaining clips if one fails"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be encoded without actually encoding"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed encoding commands and output"
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
    metadata = {
        "study_config": config,
        "system_info": system_info,
        "start_time": datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "end_time": None,
        "encodings": []
    }
    
    # Encode all clips with all parameter combinations
    print(f"\nStarting encoding ({total_encodings} total)...")
    completed = 0
    failed = 0
    
    for clip in clips:
        print(f"\nClip: {clip.name}")
        
        for params in param_combos:
            output_name = build_output_filename(clip.name, params)
            output_path = output_dir / output_name
            
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"  Encoding with {param_str}... ", end="", flush=True)
            
            result = encode_clip(clip, output_path, params, verbose=args.verbose)
            metadata["encodings"].append(result)
            
            if result["success"]:
                print(f"✓ ({result['encoding_time_seconds']:.1f}s, "
                      f"{result['file_size_bytes'] / 1024 / 1024:.2f} MB)")
                completed += 1
            else:
                print(f"✗ {result['error']}")
                failed += 1
                
                if not args.continue_on_error:
                    print("\nEncoding failed. Use --continue-on-error to continue despite failures.")
                    break
        
        if failed > 0 and not args.continue_on_error:
            break
    
    # Finalize metadata
    metadata["end_time"] = datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
    metadata["summary"] = {
        "total_encodings": total_encodings,
        "successful_encodings": completed,
        "failed_encodings": failed,
        "total_time_seconds": sum(
            e["encoding_time_seconds"] for e in metadata["encodings"]
        ),
        "total_output_size_bytes": sum(
            e.get("file_size_bytes", 0) for e in metadata["encodings"] if e["success"]
        )
    }
    
    # Save metadata
    metadata_path = output_dir / "encoding_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Study complete!")
    print(f"  Successful: {completed}/{total_encodings}")
    print(f"  Failed: {failed}/{total_encodings}")
    print(f"  Total time: {metadata['summary']['total_time_seconds']:.1f}s")
    print(f"  Total output size: {metadata['summary']['total_output_size_bytes'] / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Metadata: {metadata_path}")
    print(f"{'='*60}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
