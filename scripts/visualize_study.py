#!/usr/bin/env python3
"""
Visualize and analyze encoding study results.

Reads analysis_metadata.json and encoding_metadata.json from a study,
generates plots and exports data to CSV for further analysis.

Key visualizations:
- Rate-distortion curves (bitrate vs VMAF)
- Speed vs quality tradeoffs
- Preset/CRF parameter impact
- Per-clip comparisons
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
except ImportError:
    print(
        "Error: Required packages not installed. Install with:",
        file=sys.stderr,
    )
    print("  pip install pandas numpy matplotlib seaborn", file=sys.stderr)
    print("Or use: just install-dev", file=sys.stderr)
    sys.exit(1)


# Set plotting style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_json(file_path: Path) -> dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def prepare_dataframe(analysis_data: dict[str, Any]) -> pd.DataFrame:
    """
    Convert analysis metadata to pandas DataFrame.

    Extracts key metrics and parameters into a flat structure for analysis.
    """
    rows = []
    for encoding in analysis_data["encodings"]:
        if not encoding.get("success", False):
            continue

        row = {
            # Identifiers
            "output_file": encoding["output_file"],
            "source_clip": encoding["source_clip"],
            # Parameters
            "preset": encoding["parameters"]["preset"],
            "crf": encoding["parameters"]["crf"],
            # File metrics
            "file_size_mb": encoding["file_size_bytes"] / (1024 * 1024),
            "bitrate_kbps": encoding.get("bitrate_kbps"),
            # Encoding performance
            "encoding_time_s": encoding.get("encoding_time_seconds"),
            "encoding_fps": encoding.get("encoding_fps", 0),
            "analysis_time_s": encoding.get("analysis_time_seconds"),
        }

        # VMAF metrics
        if vmaf := encoding["metrics"].get("vmaf"):
            row.update(
                {
                    "vmaf_mean": vmaf["mean"],
                    "vmaf_harmonic_mean": vmaf["harmonic_mean"],
                    "vmaf_min": vmaf["min"],
                    "vmaf_p1": vmaf["percentile_1"],
                    "vmaf_p5": vmaf["percentile_5"],
                    "vmaf_p25": vmaf["percentile_25"],
                    "vmaf_median": vmaf["percentile_50"],
                    "vmaf_p75": vmaf["percentile_75"],
                    "vmaf_p95": vmaf["percentile_95"],
                    "vmaf_std": vmaf["std_dev"],
                }
            )

        # PSNR metrics
        if psnr := encoding["metrics"].get("psnr"):
            row["psnr_avg"] = psnr["avg_mean"]

        # SSIM metrics
        if ssim := encoding["metrics"].get("ssim"):
            row["ssim_avg"] = ssim["avg_mean"]

        # Efficiency metrics
        if eff := encoding.get("efficiency_metrics"):
            row["vmaf_per_mb"] = eff.get("vmaf_per_mbyte")
            row["quality_per_encode_s"] = eff.get("quality_per_encoding_second")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate bitrate if not present (estimate from file size)
    # Assuming typical clip duration, this is rough
    if not df.empty and df["bitrate_kbps"].isna().all():
        print(
            "Warning: bitrate_kbps not available, using file size as proxy",
            file=sys.stderr,
        )
        # Normalize by file size instead
        df["bitrate_proxy"] = df["file_size_mb"]

    return df


def plot_rate_distortion(df: pd.DataFrame, output_dir: Path, study_name: str) -> None:
    """
    Plot rate-distortion curves: VMAF vs bitrate/file size.

    Separate curves for each preset, with CRF values as points.
    """
    _fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Use file size as x-axis if bitrate not available
    x_col = "file_size_mb"
    x_label = "File Size (MB)"

    # Plot 1: VMAF mean vs file size, colored by preset
    for preset in sorted(df["preset"].unique()):
        preset_data = df[df["preset"] == preset].sort_values("crf", ascending=False)
        axes[0].plot(
            preset_data[x_col],
            preset_data["vmaf_mean"],
            marker="o",
            label=f"Preset {preset}",
            linewidth=2,
            markersize=8,
        )

        # Annotate CRF values
        for _, row in preset_data.iterrows():
            axes[0].annotate(
                f"CRF{int(row['crf'])}",
                (row[x_col], row["vmaf_mean"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                alpha=0.7,
            )

    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("VMAF Mean Score")
    axes[0].set_title(f"{study_name}: Rate-Distortion (VMAF Mean)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: VMAF harmonic mean vs file size (worst-case quality)
    for preset in sorted(df["preset"].unique()):
        preset_data = df[df["preset"] == preset].sort_values("crf", ascending=False)
        axes[1].plot(
            preset_data[x_col],
            preset_data["vmaf_harmonic_mean"],
            marker="s",
            label=f"Preset {preset}",
            linewidth=2,
            markersize=8,
        )

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("VMAF Harmonic Mean Score")
    axes[1].set_title(f"{study_name}: Rate-Distortion (VMAF Harmonic Mean)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"{study_name}_rate_distortion.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_speed_quality_tradeoff(df: pd.DataFrame, output_dir: Path, study_name: str) -> None:
    """
    Plot encoding speed vs quality tradeoffs.

    Shows how preset affects encoding time vs VMAF score.
    """
    _fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Encoding time vs VMAF, colored by preset
    for preset in sorted(df["preset"].unique()):
        preset_data = df[df["preset"] == preset]
        axes[0].scatter(
            preset_data["encoding_time_s"],
            preset_data["vmaf_mean"],
            label=f"Preset {preset}",
            s=100,
            alpha=0.7,
        )

    axes[0].set_xlabel("Encoding Time (seconds)")
    axes[0].set_ylabel("VMAF Mean Score")
    axes[0].set_title(f"{study_name}: Speed vs Quality Tradeoff")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Quality per encoding second (efficiency)
    if "quality_per_encode_s" in df.columns:
        preset_means = df.groupby("preset")["quality_per_encode_s"].mean().sort_index()
        axes[1].bar(
            [f"P{p}" for p in preset_means.index],
            preset_means.values,
            color=sns.color_palette("deep", len(preset_means)),
        )
        axes[1].set_xlabel("Preset")
        axes[1].set_ylabel("VMAF Mean / Encoding Time (score/s)")
        axes[1].set_title(f"{study_name}: Encoding Efficiency by Preset")
        axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / f"{study_name}_speed_quality.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_parameter_impact(df: pd.DataFrame, output_dir: Path, study_name: str) -> None:
    """
    Plot how parameters (preset, CRF) affect metrics.

    Heatmaps showing parameter combinations and their results.
    """
    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Aggregate by preset and CRF (mean across clips)
    grouped = df.groupby(["preset", "crf"]).agg(
        {
            "vmaf_mean": "mean",
            "file_size_mb": "mean",
            "encoding_time_s": "mean",
            "vmaf_per_mb": "mean",
        }
    )

    # Create pivot tables for heatmaps
    metrics = [
        ("vmaf_mean", "VMAF Mean Score", "RdYlGn"),
        ("file_size_mb", "File Size (MB)", "YlOrRd"),
        ("encoding_time_s", "Encoding Time (s)", "YlOrBr"),
        ("vmaf_per_mb", "VMAF per MB (efficiency)", "RdYlGn"),
    ]

    for idx, (metric, title, cmap) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        pivot = grouped[metric].reset_index().pivot(index="crf", columns="preset", values=metric)

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            ax=ax,
            cbar_kws={"label": title},
        )
        ax.set_title(f"{study_name}: {title} by Preset and CRF")
        ax.set_xlabel("Preset")
        ax.set_ylabel("CRF")

    plt.tight_layout()
    output_path = output_dir / f"{study_name}_parameter_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_clip_comparison(df: pd.DataFrame, output_dir: Path, study_name: str) -> None:
    """
    Compare how different clips respond to encoding parameters.

    Shows content-dependent behavior.
    """
    clips = df["source_clip"].unique()
    if len(clips) < 2:
        print("Skipping clip comparison (only one clip)", file=sys.stderr)
        return

    _fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: VMAF by clip for different presets
    clip_data = []
    for clip in clips:
        for preset in sorted(df["preset"].unique()):
            subset = df[(df["source_clip"] == clip) & (df["preset"] == preset)]
            if not subset.empty:
                # Keep clip identifier unique by preserving clip number
                clip_name = clip.replace(".mp4", "").replace(".mkv", "")
                # Shorten long names but keep the clip number
                if len(clip_name) > 25:
                    parts = clip_name.split("_clip_")
                    if len(parts) == 2:
                        clip_name = f"{parts[0][:15]}_c{parts[1]}"
                    else:
                        clip_name = clip_name[:25]
                clip_data.append(
                    {
                        "clip": clip_name,
                        "preset": f"P{preset}",
                        "vmaf_mean": subset["vmaf_mean"].mean(),
                    }
                )

    clip_df = pd.DataFrame(clip_data)
    clip_pivot = clip_df.pivot(index="clip", columns="preset", values="vmaf_mean")

    clip_pivot.plot(kind="bar", ax=axes[0], width=0.8)
    axes[0].set_xlabel("Source Clip")
    axes[0].set_ylabel("VMAF Mean Score")
    axes[0].set_title(f"{study_name}: Quality by Clip and Preset")
    axes[0].legend(title="Preset")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=45)

    # Plot 2: File size by clip
    clip_size_data = []
    for clip in clips:
        for preset in sorted(df["preset"].unique()):
            subset = df[(df["source_clip"] == clip) & (df["preset"] == preset)]
            if not subset.empty:
                # Keep clip identifier unique by preserving clip number
                clip_name = clip.replace(".mp4", "").replace(".mkv", "")
                # Shorten long names but keep the clip number
                if len(clip_name) > 25:
                    parts = clip_name.split("_clip_")
                    if len(parts) == 2:
                        clip_name = f"{parts[0][:15]}_c{parts[1]}"
                    else:
                        clip_name = clip_name[:25]
                clip_size_data.append(
                    {
                        "clip": clip_name,
                        "preset": f"P{preset}",
                        "file_size_mb": subset["file_size_mb"].mean(),
                    }
                )

    size_df = pd.DataFrame(clip_size_data)
    size_pivot = size_df.pivot(index="clip", columns="preset", values="file_size_mb")

    size_pivot.plot(kind="bar", ax=axes[1], width=0.8)
    axes[1].set_xlabel("Source Clip")
    axes[1].set_ylabel("File Size (MB)")
    axes[1].set_title(f"{study_name}: File Size by Clip and Preset")
    axes[1].legend(title="Preset")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = output_dir / f"{study_name}_clip_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_vmaf_distribution(df: pd.DataFrame, output_dir: Path, study_name: str) -> None:
    """
    Plot VMAF score distributions (percentiles) for different configurations.

    Shows consistency of quality across frames.
    """
    _fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Box plot of VMAF statistics by preset
    vmaf_stats = []
    for _, row in df.iterrows():
        vmaf_stats.append(
            {
                "preset": f"P{int(row['preset'])}\nCRF{int(row['crf'])}",
                "Min": row["vmaf_min"],
                "P5": row["vmaf_p5"],
                "P25": row["vmaf_p25"],
                "Median": row["vmaf_median"],
                "P75": row["vmaf_p75"],
                "P95": row["vmaf_p95"],
                "Mean": row["vmaf_mean"],
            }
        )

    stats_df = pd.DataFrame(vmaf_stats)

    # Create box plot manually using percentiles
    positions = range(len(stats_df))
    axes[0].boxplot(
        [[s["P5"], s["P25"], s["Median"], s["P75"], s["P95"]] for _, s in stats_df.iterrows()],
        positions=positions,
        widths=0.6,
        showfliers=False,
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(stats_df["preset"], rotation=45, ha="right")
    axes[0].set_ylabel("VMAF Score")
    axes[0].set_title(f"{study_name}: VMAF Distribution (P5-P95)")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Plot 2: Mean vs Harmonic Mean (shows quality consistency)
    for preset in sorted(df["preset"].unique()):
        preset_data = df[df["preset"] == preset]
        axes[1].scatter(
            preset_data["vmaf_mean"],
            preset_data["vmaf_harmonic_mean"],
            label=f"Preset {preset}",
            s=100,
            alpha=0.7,
        )

    # Add diagonal line (mean = harmonic mean would be perfect consistency)
    lims = [
        max(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
        min(axes[1].get_xlim()[1], axes[1].get_ylim()[1]),
    ]
    axes[1].plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    axes[1].set_xlabel("VMAF Mean Score")
    axes[1].set_ylabel("VMAF Harmonic Mean Score")
    axes[1].set_title(f"{study_name}: Quality Consistency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"{study_name}_vmaf_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def export_csv(df: pd.DataFrame, output_dir: Path, study_name: str) -> None:
    """Export dataframe to CSV for further analysis."""
    output_path = output_dir / f"{study_name}_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def generate_summary_report(
    df: pd.DataFrame,
    analysis_data: dict[str, Any],
    output_dir: Path,
    study_name: str,
) -> None:
    """Generate a text summary report."""
    report_path = output_dir / f"{study_name}_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Analysis Report: {study_name}\n")
        f.write("=" * 80 + "\n\n")

        # Study metadata
        f.write(f"Analysis Date: {analysis_data['analysis_date']}\n")
        f.write(f"VMAF Model: {analysis_data['vmaf_model']}\n")
        f.write(f"Clips Analyzed: {analysis_data['clips_analyzed']}\n")
        f.write(f"Total Encodings: {analysis_data['total_encodings_analyzed']}\n")
        f.write(f"Metrics: {', '.join(analysis_data['metrics_calculated']).upper()}\n")
        f.write("\n")

        # Overall statistics
        f.write("Overall Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"VMAF Mean Range: {df['vmaf_mean'].min():.2f} - {df['vmaf_mean'].max():.2f}\n")
        f.write(
            f"File Size Range: {df['file_size_mb'].min():.2f} - {df['file_size_mb'].max():.2f} MB\n"
        )
        if "encoding_time_s" in df.columns:
            f.write(
                f"Encoding Time Range: {df['encoding_time_s'].min():.2f} - "
                f"{df['encoding_time_s'].max():.2f} seconds\n"
            )
        f.write("\n")

        # Best configurations
        f.write("Best Configurations\n")
        f.write("-" * 80 + "\n")

        best_quality = df.loc[df["vmaf_mean"].idxmax()]
        f.write("Highest Quality:\n")
        f.write(f"  Preset: {int(best_quality['preset'])}, CRF: {int(best_quality['crf'])}\n")
        f.write(f"  VMAF: {best_quality['vmaf_mean']:.2f}\n")
        f.write(f"  File Size: {best_quality['file_size_mb']:.2f} MB\n")
        f.write(f"  Encoding Time: {best_quality['encoding_time_s']:.2f}s\n")
        f.write("\n")

        if "vmaf_per_mb" in df.columns:
            best_efficiency = df.loc[df["vmaf_per_mb"].idxmax()]
            f.write("Best Efficiency (VMAF per MB):\n")
            f.write(
                f"  Preset: {int(best_efficiency['preset'])}, CRF: {int(best_efficiency['crf'])}\n"
            )
            f.write(f"  VMAF: {best_efficiency['vmaf_mean']:.2f}\n")
            f.write(f"  VMAF per MB: {best_efficiency['vmaf_per_mb']:.2f}\n")
            f.write(f"  File Size: {best_efficiency['file_size_mb']:.2f} MB\n")
            f.write("\n")

        smallest_file = df.loc[df["file_size_mb"].idxmin()]
        f.write("Smallest File:\n")
        f.write(f"  Preset: {int(smallest_file['preset'])}, CRF: {int(smallest_file['crf'])}\n")
        f.write(f"  VMAF: {smallest_file['vmaf_mean']:.2f}\n")
        f.write(f"  File Size: {smallest_file['file_size_mb']:.2f} MB\n")
        f.write("\n")

        # Parameter impact summary
        f.write("Parameter Impact Summary\n")
        f.write("-" * 80 + "\n")

        preset_stats = (
            df.groupby("preset")
            .agg(
                {
                    "vmaf_mean": ["mean", "std"],
                    "file_size_mb": ["mean", "std"],
                    "encoding_time_s": ["mean", "std"],
                }
            )
            .round(2)
        )

        f.write("\nBy Preset:\n")
        f.write(preset_stats.to_string())
        f.write("\n\n")

        crf_stats = (
            df.groupby("crf")
            .agg(
                {
                    "vmaf_mean": ["mean", "std"],
                    "file_size_mb": ["mean", "std"],
                }
            )
            .round(2)
        )

        f.write("By CRF:\n")
        f.write(crf_stats.to_string())
        f.write("\n")

    print(f"Saved: {report_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize and analyze encoding study results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze baseline_sweep study with all plots
  %(prog)s baseline_sweep

  # Analyze with custom output directory
  %(prog)s baseline_sweep --output results/baseline_analysis

  # Only generate specific plot types
  %(prog)s baseline_sweep --plots rate-distortion speed-quality

Available plot types:
  - rate-distortion: VMAF vs bitrate/file size curves
  - speed-quality: Encoding time vs quality tradeoffs
  - parameter-impact: Heatmaps of parameter effects
  - clip-comparison: Per-clip quality comparison
  - vmaf-distribution: VMAF score distributions and consistency
  - all: Generate all plots (default)
        """,
    )

    parser.add_argument(
        "study_name",
        help="Name of the study to analyze (e.g., 'baseline_sweep')",
    )

    parser.add_argument(
        "--encoded-dir",
        type=Path,
        default=Path("data/encoded"),
        help="Directory containing encoded study results (default: data/encoded)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for plots and CSV (default: results/<study_name>)",
    )

    parser.add_argument(
        "--plots",
        nargs="+",
        choices=[
            "rate-distortion",
            "speed-quality",
            "parameter-impact",
            "clip-comparison",
            "vmaf-distribution",
            "all",
        ],
        default=["all"],
        help="Which plots to generate (default: all)",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV export",
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip text summary report",
    )

    args = parser.parse_args()

    # Resolve paths
    study_dir = args.encoded_dir / args.study_name
    if not study_dir.exists():
        print(f"Error: Study directory not found: {study_dir}", file=sys.stderr)
        print("\nAvailable studies:", file=sys.stderr)
        if args.encoded_dir.exists():
            for d in args.encoded_dir.iterdir():
                if d.is_dir() and (d / "analysis_metadata.json").exists():
                    print(f"  - {d.name}", file=sys.stderr)
        sys.exit(1)

    analysis_file = study_dir / "analysis_metadata.json"
    if not analysis_file.exists():
        print(f"Error: Analysis not found: {analysis_file}", file=sys.stderr)
        print(f"Run: just analyze-study {args.study_name}", file=sys.stderr)
        sys.exit(1)

    # Set output directory
    output_dir = args.output or Path("results") / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing study: {args.study_name}")
    print(f"Reading: {analysis_file}")
    print(f"Output directory: {output_dir}\n")

    # Load data
    analysis_data = load_json(analysis_file)
    df = prepare_dataframe(analysis_data)

    if df.empty:
        print("Error: No successful encodings found in analysis", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} successful encodings\n")

    # Determine which plots to generate
    plot_types = args.plots
    if "all" in plot_types:
        plot_types = [
            "rate-distortion",
            "speed-quality",
            "parameter-impact",
            "clip-comparison",
            "vmaf-distribution",
        ]

    # Generate plots
    plot_functions = {
        "rate-distortion": plot_rate_distortion,
        "speed-quality": plot_speed_quality_tradeoff,
        "parameter-impact": plot_parameter_impact,
        "clip-comparison": plot_clip_comparison,
        "vmaf-distribution": plot_vmaf_distribution,
    }

    for plot_type in plot_types:
        if plot_type in plot_functions:
            print(f"Generating {plot_type} plot...")
            plot_functions[plot_type](df, output_dir, args.study_name)

    # Export CSV
    if not args.no_csv:
        print("\nExporting CSV...")
        export_csv(df, output_dir, args.study_name)

    # Generate summary report
    if not args.no_report:
        print("\nGenerating summary report...")
        generate_summary_report(df, analysis_data, output_dir, args.study_name)

    print(f"\n✅ Analysis complete! Results in: {output_dir}")


if __name__ == "__main__":
    main()
