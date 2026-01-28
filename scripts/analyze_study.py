#!/usr/bin/env python3
"""
Analyze encoded videos from a study by calculating quality metrics.

This script calculates VMAF (NEG mode), PSNR, and SSIM for all encodings
in a study by comparing them against the original source clips. Results
are stored in analysis_metadata.json alongside the encoding metadata.

Quality Metrics:
- VMAF NEG: Netflix's perceptual quality metric (No Enhancement Gain mode)
  * NEG mode disables enhancement gain, making it ideal for codec evaluation
  * Industry standard for measuring compression quality
- PSNR: Peak Signal-to-Noise Ratio (traditional pixel difference metric)
- SSIM: Structural Similarity Index (perceptual similarity metric)

Usage:
    python analyze_study.py baseline_sweep
    python analyze_study.py baseline_sweep --metrics vmaf
    python analyze_study.py film_grain --continue-on-error
    python analyze_study.py baseline_sweep --threads 8 -v
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tempfile


def load_encoding_metadata(study_dir: Path) -> Dict[str, Any]:
    """Load encoding metadata for a study."""
    metadata_file = study_dir / "encoding_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Encoding metadata not found: {metadata_file}\n"
            f"Run encoding study first: just encode-study {study_dir.name}"
        )
    
    with open(metadata_file) as f:
        return json.load(f)


def find_source_clip(clip_name: str, clips_dir: Path) -> Optional[Path]:
    """Find the source clip file in the clips directory."""
    # Try exact match first
    clip_path = clips_dir / clip_name
    if clip_path.exists():
        return clip_path
    
    # Try with common extensions
    for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']:
        clip_path = clips_dir / f"{Path(clip_name).stem}{ext}"
        if clip_path.exists():
            return clip_path
    
    return None


def calculate_vmaf(
    reference: Path,
    distorted: Path,
    model: str = "version=vmaf_v0.6.1neg",
    threads: int = 4,
    verbose: bool = False
) -> Optional[Dict[str, float]]:
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        log_file = Path(tmp.name)
    
    try:
        # Build FFmpeg command for VMAF calculation
        # Format: ffmpeg -i distorted -i reference -lavfi libvmaf -f null -
        cmd = [
            "ffmpeg",
            "-i", str(distorted),
            "-i", str(reference),
            "-lavfi",
            f"[0:v][1:v]libvmaf=model={model}:log_path={log_file}:log_fmt=json:n_threads={threads}",
            "-f", "null",
            "-"
        ]
        
        if verbose:
            print(f"  Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max
        )
        
        if result.returncode != 0:
            print(f"  ERROR: VMAF calculation failed")
            if verbose:
                print(f"  stderr: {result.stderr}")
            return None
        
        # Parse VMAF JSON output
        with open(log_file) as f:
            vmaf_data = json.load(f)
        
        # Extract frame scores
        frames = vmaf_data.get("frames", [])
        if not frames:
            print(f"  ERROR: No VMAF frames found in output")
            return None
        
        scores = [frame["metrics"]["vmaf"] for frame in frames]
        
        # Calculate statistics
        scores_sorted = sorted(scores)
        n = len(scores)
        
        mean = sum(scores) / n
        
        stats = {
            "mean": mean,
            "harmonic_mean": n / sum(1/s if s > 0 else 0 for s in scores),
            "min": min(scores),
            "max": max(scores),
            "percentile_1": scores_sorted[int(n * 0.01)],
            "percentile_5": scores_sorted[int(n * 0.05)],
            "percentile_25": scores_sorted[int(n * 0.25)],
            "percentile_50": scores_sorted[int(n * 0.50)],
            "percentile_75": scores_sorted[int(n * 0.75)],
            "percentile_95": scores_sorted[int(n * 0.95)],
            "std_dev": (sum((s - mean)**2 for s in scores) / n) ** 0.5
        }
        
        # Round to 2 decimal places
        stats = {k: round(v, 2) for k, v in stats.items()}
        
        return stats
        
    except subprocess.TimeoutExpired:
        print(f"  ERROR: VMAF calculation timed out")
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
    verbose: bool = False
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Calculate PSNR and/or SSIM using FFmpeg filters.
    
    Returns:
        Tuple of (psnr_stats, ssim_stats), each can be None if not calculated
    """
    if not calculate_psnr and not calculate_ssim:
        return None, None
    
    # Build filter chain
    filters = []
    if calculate_psnr:
        filters.append("[0:v][1:v]psnr=stats_file=-")
    if calculate_ssim:
        filters.append("[0:v][1:v]ssim=stats_file=-")
    
    filter_chain = ";".join(filters)
    
    try:
        cmd = [
            "ffmpeg",
            "-i", str(distorted),
            "-i", str(reference),
            "-lavfi", filter_chain,
            "-f", "null",
            "-"
        ]
        
        if verbose:
            print(f"  Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode != 0:
            print(f"  ERROR: PSNR/SSIM calculation failed")
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
                        "avg_mean": round(values.get("average", 0), 2)
                    }
                except (ValueError, KeyError) as e:
                    if verbose:
                        print(f"  Warning: Failed to parse PSNR: {e}")
            
            if calculate_ssim and "SSIM" in line and "All:" in line:
                # Example: [Parsed_ssim_0 @ 0x...] SSIM Y:0.982 U:0.991 V:0.990 All:0.985 (18.234)
                try:
                    parts = line.split("SSIM")[1].strip()
                    values = {}
                    for part in parts.split():
                        if ":" in part:
                            k, v = part.split(":")
                            # Remove parentheses if present
                            v = v.strip("()")
                            try:
                                values[k] = float(v)
                            except ValueError:
                                pass
                    
                    ssim_stats = {
                        "y_mean": round(values.get("Y", 0), 4),
                        "u_mean": round(values.get("U", 0), 4),
                        "v_mean": round(values.get("V", 0), 4),
                        "avg_mean": round(values.get("All", 0), 4)
                    }
                except (ValueError, KeyError) as e:
                    if verbose:
                        print(f"  Warning: Failed to parse SSIM: {e}")
        
        return psnr_stats, ssim_stats
        
    except subprocess.TimeoutExpired:
        print(f"  ERROR: PSNR/SSIM calculation timed out")
        return None, None
    except Exception as e:
        print(f"  ERROR: PSNR/SSIM calculation failed: {e}")
        return None, None


def calculate_efficiency_metrics(
    vmaf_mean: Optional[float],
    file_size_bytes: int,
    bitrate_kbps: Optional[float],
    encoding_time: Optional[float]
) -> Dict[str, float]:
    """Calculate derived efficiency metrics."""
    metrics = {}
    
    if vmaf_mean is not None:
        # VMAF per megabyte
        size_mb = file_size_bytes / (1024 * 1024)
        if size_mb > 0:
            metrics["vmaf_per_mbyte"] = round(vmaf_mean / size_mb, 2)
        
        # VMAF per kbps (quality per bitrate unit)
        if bitrate_kbps is not None and bitrate_kbps > 0:
            metrics["vmaf_per_kbps"] = round(vmaf_mean / bitrate_kbps, 4)
        
        # Quality per encoding second (speed/quality tradeoff)
        if encoding_time and encoding_time > 0:
            metrics["quality_per_encoding_second"] = round(vmaf_mean / encoding_time, 2)
    
    return metrics


def analyze_encoding(
    encoding: Dict[str, Any],
    study_dir: Path,
    clips_dir: Path,
    metrics: List[str],
    threads: int,
    verbose: bool
) -> Dict[str, Any]:
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
            "file_size_bytes": encoding["file_size_bytes"],
            "bitrate_kbps": encoding.get("bitrate_kbps"),
            "success": False,
            "error": f"Source clip not found: {source_clip_name}"
        }
    
    # Check if encoded file exists
    encoded_file = study_dir / output_file
    if not encoded_file.exists():
        return {
            "output_file": output_file,
            "source_clip": source_clip_name,
            "parameters": encoding["parameters"],
            "metrics": {},
            "file_size_bytes": encoding["file_size_bytes"],
            "bitrate_kbps": encoding.get("bitrate_kbps"),
            "success": False,
            "error": f"Encoded file not found: {encoded_file}"
        }
    
    start_time = time.time()
    
    # Calculate metrics
    result = {
        "output_file": output_file,
        "source_clip": source_clip_name,
        "parameters": encoding["parameters"],
        "metrics": {},
        "file_size_bytes": encoding["file_size_bytes"],
        "bitrate_kbps": encoding.get("bitrate_kbps"),
        "duration_seconds": encoding.get("duration_seconds"),
        "encoding_time_seconds": encoding.get("encoding_time_seconds"),
        "encoding_fps": encoding.get("fps"),
        "success": True,
        "error": None
    }
    
    # VMAF
    if "vmaf" in metrics or "vmaf_neg" in metrics:
        print(f"  Calculating VMAF (NEG mode)...")
        vmaf_model = "version=vmaf_v0.6.1neg"
        vmaf_stats = calculate_vmaf(
            reference=source_clip,
            distorted=encoded_file,
            model=vmaf_model,
            threads=threads,
            verbose=verbose
        )
        
        if vmaf_stats:
            result["metrics"]["vmaf"] = vmaf_stats
            print(f"    Mean: {vmaf_stats['mean']:.2f}, "
                  f"Harmonic: {vmaf_stats['harmonic_mean']:.2f}, "
                  f"Min: {vmaf_stats['min']:.2f}")
        else:
            result["success"] = False
            result["error"] = "VMAF calculation failed"
    
    # PSNR and SSIM
    calculate_psnr = "psnr" in metrics
    calculate_ssim = "ssim" in metrics
    
    if calculate_psnr or calculate_ssim:
        if calculate_psnr:
            print(f"  Calculating PSNR...")
        if calculate_ssim:
            print(f"  Calculating SSIM...")
        
        psnr_stats, ssim_stats = calculate_psnr_ssim(
            reference=source_clip,
            distorted=encoded_file,
            calculate_psnr=calculate_psnr,
            calculate_ssim=calculate_ssim,
            verbose=verbose
        )
        
        if psnr_stats:
            result["metrics"]["psnr"] = psnr_stats
            print(f"    PSNR Y: {psnr_stats['y_mean']:.2f} dB, "
                  f"Avg: {psnr_stats['avg_mean']:.2f} dB")
        
        if ssim_stats:
            result["metrics"]["ssim"] = ssim_stats
            print(f"    SSIM Y: {ssim_stats['y_mean']:.4f}, "
                  f"Avg: {ssim_stats['avg_mean']:.4f}")
    
    # Calculate efficiency metrics
    vmaf_mean = result["metrics"].get("vmaf", {}).get("mean")
    if vmaf_mean:
        result["efficiency_metrics"] = calculate_efficiency_metrics(
            vmaf_mean=vmaf_mean,
            file_size_bytes=result["file_size_bytes"],
            bitrate_kbps=result["bitrate_kbps"],
            encoding_time=result.get("encoding_time_seconds")
        )
        
        if result["efficiency_metrics"]:
            print(f"  Efficiency:")
            if "vmaf_per_kbps" in result["efficiency_metrics"]:
                print(f"    VMAF/kbps: {result['efficiency_metrics']['vmaf_per_kbps']:.4f}")
            if "vmaf_per_mbyte" in result["efficiency_metrics"]:
                print(f"    VMAF/MB: {result['efficiency_metrics']['vmaf_per_mbyte']:.2f}")
    
    analysis_time = time.time() - start_time
    result["analysis_time_seconds"] = round(analysis_time, 2)
    print(f"  Analysis time: {analysis_time:.1f}s")
    
    return result


def calculate_summary(encodings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics across all encodings."""
    successful = [e for e in encodings if e.get("success")]
    
    if not successful:
        return {}
    
    vmaf_means = [e["metrics"].get("vmaf", {}).get("mean") 
                  for e in successful if "vmaf" in e["metrics"]]
    bitrates = [e["bitrate_kbps"] for e in successful if e.get("bitrate_kbps")]
    
    summary = {
        "total_analysis_time_seconds": sum(
            e.get("analysis_time_seconds", 0) for e in encodings
        )
    }
    
    if vmaf_means:
        summary["vmaf_range"] = {
            "min_mean": round(min(vmaf_means), 2),
            "max_mean": round(max(vmaf_means), 2)
        }
    
    if bitrates:
        summary["bitrate_range_kbps"] = {
            "min": round(min(bitrates), 2),
            "max": round(max(bitrates), 2)
        }
    
    # Find best efficiency (VMAF per kbps)
    with_efficiency = [
        e for e in successful 
        if e.get("efficiency_metrics", {}).get("vmaf_per_kbps")
    ]
    if with_efficiency:
        best_eff = max(with_efficiency, 
                      key=lambda e: e["efficiency_metrics"]["vmaf_per_kbps"])
        summary["best_efficiency"] = {
            "output_file": best_eff["output_file"],
            "parameters": best_eff["parameters"],
            "vmaf_mean": best_eff["metrics"]["vmaf"]["mean"],
            "vmaf_per_kbps": best_eff["efficiency_metrics"]["vmaf_per_kbps"],
            "bitrate_kbps": best_eff["bitrate_kbps"]
        }
    
    # Find best quality (highest VMAF)
    if vmaf_means:
        with_vmaf = [e for e in successful if "vmaf" in e["metrics"]]
        best_quality = max(with_vmaf, 
                          key=lambda e: e["metrics"]["vmaf"]["mean"])
        summary["best_quality"] = {
            "output_file": best_quality["output_file"],
            "parameters": best_quality["parameters"],
            "vmaf_mean": best_quality["metrics"]["vmaf"]["mean"],
            "bitrate_kbps": best_quality["bitrate_kbps"]
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
        """
    )
    parser.add_argument(
        "study_name",
        help="Name of the study to analyze (e.g., baseline_sweep)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["vmaf", "vmaf_neg", "psnr", "ssim"],
        default=["vmaf", "psnr", "ssim"],
        help="Metrics to calculate (default: vmaf psnr ssim). vmaf and vmaf_neg are equivalent (both use NEG mode)"
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("data/test_clips"),
        help="Directory containing source clips (default: data/test_clips)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads for VMAF calculation (default: auto-detect all CPU cores)"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue analyzing remaining encodings if one fails"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed FFmpeg output"
    )
    
    args = parser.parse_args()
    
    # Auto-detect number of threads if not specified
    if args.threads is None:
        args.threads = os.cpu_count() or 4  # Fallback to 4 if cpu_count() returns None
    
    # Find study directory
    study_dir = Path("data/encoded") / args.study_name
    if not study_dir.exists():
        print(f"Error: Study directory not found: {study_dir}")
        print(f"Available studies:")
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
    
    encodings_to_analyze = encoding_metadata.get("encodings", [])
    successful_encodings = [e for e in encodings_to_analyze if e.get("success")]
    
    if not successful_encodings:
        print(f"\nNo successful encodings found in study")
        sys.exit(1)
    
    print(f"\nEncodings to analyze: {len(successful_encodings)}")
    
    # Get unique source clips
    unique_clips = set(e["source_clip"] for e in successful_encodings)
    print(f"Unique source clips: {len(unique_clips)}")
    
    # Analyze each encoding
    analysis_results = []
    failed_count = 0
    
    for i, encoding in enumerate(successful_encodings, 1):
        print(f"\n[{i}/{len(successful_encodings)}]", end=" ")
        
        try:
            result = analyze_encoding(
                encoding=encoding,
                study_dir=study_dir,
                clips_dir=args.clips_dir,
                metrics=metrics,
                threads=args.threads,
                verbose=args.verbose
            )
            analysis_results.append(result)
            
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
    print("\n" + "="*70)
    print("Calculating summary statistics...")
    summary = calculate_summary(analysis_results)
    
    # Save results
    output_file = study_dir / "analysis_metadata.json"
    output_data = {
        "study_name": args.study_name,
        "analysis_date": datetime.utcnow().isoformat() + "Z",
        "metrics_calculated": metrics,
        "vmaf_model": "vmaf_v0.6.1neg",
        "clips_analyzed": len(unique_clips),
        "total_encodings_analyzed": len(analysis_results),
        "encodings": analysis_results,
        "summary": summary
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"  Total analyzed: {len(analysis_results)}")
    print(f"  Successful: {len(analysis_results) - failed_count}")
    print(f"  Failed: {failed_count}")
    
    if summary:
        if "vmaf_range" in summary:
            print(f"\nVMAF range: {summary['vmaf_range']['min_mean']:.2f} - "
                  f"{summary['vmaf_range']['max_mean']:.2f}")
        
        if "best_efficiency" in summary:
            best = summary["best_efficiency"]
            print(f"\nBest efficiency (VMAF/kbps):")
            print(f"  File: {best['output_file']}")
            print(f"  Parameters: {best['parameters']}")
            print(f"  VMAF: {best['vmaf_mean']:.2f}")
            print(f"  Bitrate: {best['bitrate_kbps']:.1f} kbps")
            print(f"  Efficiency: {best['vmaf_per_kbps']:.4f}")
        
        if "best_quality" in summary:
            best = summary["best_quality"]
            print(f"\nBest quality (highest VMAF):")
            print(f"  File: {best['output_file']}")
            print(f"  Parameters: {best['parameters']}")
            print(f"  VMAF: {best['vmaf_mean']:.2f}")
            print(f"  Bitrate: {best['bitrate_kbps']:.1f} kbps")
        
        print(f"\nTotal analysis time: {summary['total_analysis_time_seconds']:.1f}s")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
