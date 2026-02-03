#!/usr/bin/env python3
"""
Visualize and analyze encoding study results.

Reads analysis_metadata.json from a study and generates clear, focused plots.
Each metric gets three visualizations:
  1. Heatmap: CRF vs Preset
  2. Line chart: Metric vs CRF (one line per preset)
  3. Line chart: Metric vs Preset (one line per CRF)

Design principles:
  - One figure per file (no subfigures)
  - Colorblind-friendly palettes (viridis)
  - Aggregate statistics across clips first
  - Focus on actionable metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


try:
    import matplotlib.pyplot as plt
    import numpy as np
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


# Colorblind-friendly palette: viridis (dark purple/blue to yellow)
COLORMAP = "viridis"
# For line plots, use colorful discrete palette with distinct colors
# Markers will provide additional distinction for colorblind accessibility
LINE_PALETTE = "tab10"  # Standard colorful palette
# Marker shapes to cycle through for line plots
LINE_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

# Configure matplotlib defaults
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# Use a clean style
sns.set_theme(style="whitegrid")


# Metrics to analyze with their display names and higher_is_better flag
# Special metric: "vmaf_combined" generates plots with both mean and p5
METRICS = [
    ("vmaf_combined", "VMAF Score (Mean and P5)", True),  # Combined plot
    ("bpp", "Bitrate per Pixel (bpp)", False),
    ("vmaf_per_bpp", "VMAF per bpp (Quality Efficiency)", True),
    ("encoding_time_s", "Encoding Time (seconds)", False),
    ("vmaf_per_time", "VMAF per Encoding Second", True),
    ("vmaf_per_bpp_per_time", "VMAF per bpp per Encoding Second", True),
]


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


def load_clip_metadata() -> dict[str, dict[str, Any]]:
    """Load clip metadata to get resolution information."""
    clip_metadata_path = Path("data/test_clips/clip_metadata.json")
    if not clip_metadata_path.exists():
        return {}

    try:
        with open(clip_metadata_path, encoding="utf-8") as f:
            data = json.load(f)
            return {clip["clip_name"]: clip for clip in data.get("clips", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def prepare_dataframe(analysis_data: dict[str, Any]) -> pd.DataFrame:
    """
    Convert analysis metadata to pandas DataFrame.

    Extracts key metrics and parameters into a flat structure for analysis.
    Calculates derived metrics like bpp, vmaf_per_bpp, etc.
    """
    clip_metadata = load_clip_metadata()

    rows = []
    for encoding in analysis_data["encodings"]:
        if not encoding.get("success", False):
            continue

        # Get resolution from clip metadata
        source_clip = encoding["source_clip"]
        clip_info = clip_metadata.get(source_clip, {})
        width = clip_info.get("source_width", 0)
        height = clip_info.get("source_height", 0)
        fps = clip_info.get("source_fps", 0)

        file_size_bytes = encoding["file_size_bytes"]
        bitrate_kbps = encoding.get("bitrate_kbps", 0) or 0
        encoding_time_s = encoding.get("encoding_time_seconds", 0) or 0

        # Calculate bitrate per pixel (bpp)
        # bpp = bitrate / (width * height * fps)
        bpp = 0.0
        if bitrate_kbps and width and height and fps:
            bitrate_bps = bitrate_kbps * 1000
            pixels_per_second = width * height * fps
            bpp = bitrate_bps / pixels_per_second

        row = {
            # Identifiers
            "output_file": encoding["output_file"],
            "source_clip": source_clip,
            # Parameters
            "preset": encoding["parameters"]["preset"],
            "crf": encoding["parameters"]["crf"],
            # Resolution
            "width": width,
            "height": height,
            "fps": fps,
            # File metrics
            "file_size_mb": file_size_bytes / (1024 * 1024),
            "bitrate_kbps": bitrate_kbps,
            "bpp": bpp,
            # Encoding time
            "encoding_time_s": encoding_time_s,
            "encoding_fps": encoding.get("encoding_fps", 0),
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

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Calculate derived metrics
    # VMAF per bpp (quality efficiency)
    df["vmaf_per_bpp"] = np.where(
        df["bpp"] > 0,
        df["vmaf_mean"] / df["bpp"],
        np.nan,
    )

    # VMAF per encoding time
    df["vmaf_per_time"] = np.where(
        df["encoding_time_s"] > 0,
        df["vmaf_mean"] / df["encoding_time_s"],
        np.nan,
    )

    # VMAF per bpp per encoding time (combined efficiency)
    df["vmaf_per_bpp_per_time"] = np.where(
        (df["bpp"] > 0) & (df["encoding_time_s"] > 0),
        df["vmaf_mean"] / df["bpp"] / df["encoding_time_s"],
        np.nan,
    )

    return df


def aggregate_by_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data by preset and CRF, computing mean across all clips.

    Returns a DataFrame with one row per (preset, crf) combination.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove preset and crf from aggregation columns
    agg_cols = [c for c in numeric_cols if c not in ["preset", "crf"]]

    aggregated = df.groupby(["preset", "crf"])[agg_cols].mean().reset_index()
    return aggregated


def get_line_colors_and_markers(n: int) -> tuple[list, list]:
    """Get n distinct colors and markers for line plots.

    Returns:
        tuple: (colors, markers) where colors are from tab10 and markers cycle through shapes
    """
    cmap = plt.colormaps.get_cmap(LINE_PALETTE)
    colors = [cmap(i % 10) for i in range(n)]  # tab10 has 10 colors
    markers = [LINE_MARKERS[i % len(LINE_MARKERS)] for i in range(n)]
    return colors, markers


def plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    output_dir: Path,
    study_name: str,
    higher_is_better: bool = True,
) -> None:
    """
    Plot a heatmap of the metric vs CRF (y-axis) and Preset (x-axis).

    Uses viridis colormap for colorblind accessibility.
    """
    pivot = df.pivot(index="crf", columns="preset", values=metric)

    if pivot.empty or pivot.isna().all().all():
        print(f"  Skipping heatmap for {metric}: no data available", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Use viridis, reversed if lower is better
    cmap = COLORMAP if higher_is_better else f"{COLORMAP}_r"

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": metric_label},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Preset (lower = slower, higher quality)")
    ax.set_ylabel("CRF (lower = higher bitrate/quality)")
    ax.set_title(f"{study_name}: {metric_label}\n(Preset vs CRF)")

    output_path = output_dir / f"{study_name}_heatmap_{metric}.png"
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_vs_crf(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    output_dir: Path,
    study_name: str,
) -> None:
    """
    Plot metric vs CRF, with one line per preset.

    Uses colorful palette with varying markers for accessibility.
    """
    presets = sorted(df["preset"].unique())
    colors, markers = get_line_colors_and_markers(len(presets))

    fig, ax = plt.subplots(figsize=(10, 7))

    for preset, color, marker in zip(presets, colors, markers, strict=True):
        preset_data = df[df["preset"] == preset].sort_values("crf")
        if preset_data[metric].isna().all():
            continue
        ax.plot(
            preset_data["crf"],
            preset_data[metric],
            marker=marker,
            linewidth=2,
            markersize=8,
            label=f"Preset {preset}",
            color=color,
        )

    ax.set_xlabel("CRF (lower = higher bitrate/quality)")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{study_name}: {metric_label} vs CRF")
    ax.legend(title="Preset", loc="best")
    ax.grid(True, alpha=0.3)

    # Set integer ticks for CRF
    ax.set_xticks(sorted(df["crf"].unique()))

    output_path = output_dir / f"{study_name}_vs_crf_{metric}.png"
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_vs_preset(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    output_dir: Path,
    study_name: str,
) -> None:
    """
    Plot metric vs preset, with one line per CRF.

    Uses colorful palette with varying markers for accessibility.
    """
    crfs = sorted(df["crf"].unique())
    colors, markers = get_line_colors_and_markers(len(crfs))

    fig, ax = plt.subplots(figsize=(10, 7))

    for crf, color, marker in zip(crfs, colors, markers, strict=True):
        crf_data = df[df["crf"] == crf].sort_values("preset")
        if crf_data[metric].isna().all():
            continue
        ax.plot(
            crf_data["preset"],
            crf_data[metric],
            marker=marker,
            linewidth=2,
            markersize=8,
            label=f"CRF {crf}",
            color=color,
        )

    ax.set_xlabel("Preset (lower = slower, higher quality)")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{study_name}: {metric_label} vs Preset")
    ax.legend(title="CRF", loc="best")
    ax.grid(True, alpha=0.3)

    # Set integer ticks for preset
    ax.set_xticks(sorted(df["preset"].unique()))

    output_path = output_dir / f"{study_name}_vs_preset_{metric}.png"
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_vmaf_combined_vs_crf(
    df: pd.DataFrame,
    output_dir: Path,
    study_name: str,
) -> None:
    """
    Plot VMAF mean and P5 vs CRF on the same plot.

    Mean: solid lines with filled markers
    P5: dashed lines with open (unfilled) markers
    Same color and marker shape for each preset across both metrics.
    """
    presets = sorted(df["preset"].unique())
    colors, markers = get_line_colors_and_markers(len(presets))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot VMAF mean (solid lines, filled markers)
    for preset, color, marker in zip(presets, colors, markers, strict=True):
        preset_data = df[df["preset"] == preset].sort_values("crf")
        if preset_data["vmaf_mean"].isna().all():
            continue
        ax.plot(
            preset_data["crf"],
            preset_data["vmaf_mean"],
            marker=marker,
            linestyle="-",
            linewidth=2,
            markersize=8,
            label=f"Preset {preset} (Mean)",
            color=color,
            markerfacecolor=color,
        )

    # Plot VMAF P5 (dashed lines, open markers)
    for preset, color, marker in zip(presets, colors, markers, strict=True):
        preset_data = df[df["preset"] == preset].sort_values("crf")
        if preset_data["vmaf_p5"].isna().all():
            continue
        ax.plot(
            preset_data["crf"],
            preset_data["vmaf_p5"],
            marker=marker,
            linestyle="--",
            linewidth=2,
            markersize=8,
            label=f"Preset {preset} (P5)",
            color=color,
            markerfacecolor="none",
            markeredgewidth=2,
        )

    ax.set_xlabel("CRF (lower = higher bitrate/quality)")
    ax.set_ylabel("VMAF Score")
    ax.set_title(f"{study_name}: VMAF Score (Mean and P5) vs CRF")
    ax.legend(title="Metric", loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(df["crf"].unique()))

    output_path = output_dir / f"{study_name}_vs_crf_vmaf_combined.png"
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_vmaf_combined_vs_preset(
    df: pd.DataFrame,
    output_dir: Path,
    study_name: str,
) -> None:
    """
    Plot VMAF mean and P5 vs preset on the same plot.

    Mean: solid lines with filled markers
    P5: dashed lines with open (unfilled) markers
    Same color and marker shape for each CRF across both metrics.
    """
    crfs = sorted(df["crf"].unique())
    colors, markers = get_line_colors_and_markers(len(crfs))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot VMAF mean (solid lines, filled markers)
    for crf, color, marker in zip(crfs, colors, markers, strict=True):
        crf_data = df[df["crf"] == crf].sort_values("preset")
        if crf_data["vmaf_mean"].isna().all():
            continue
        ax.plot(
            crf_data["preset"],
            crf_data["vmaf_mean"],
            marker=marker,
            linestyle="-",
            linewidth=2,
            markersize=8,
            label=f"CRF {crf} (Mean)",
            color=color,
            markerfacecolor=color,
        )

    # Plot VMAF P5 (dashed lines, open markers)
    for crf, color, marker in zip(crfs, colors, markers, strict=True):
        crf_data = df[df["crf"] == crf].sort_values("preset")
        if crf_data["vmaf_p5"].isna().all():
            continue
        ax.plot(
            crf_data["preset"],
            crf_data["vmaf_p5"],
            marker=marker,
            linestyle="--",
            linewidth=2,
            markersize=8,
            label=f"CRF {crf} (P5)",
            color=color,
            markerfacecolor="none",
            markeredgewidth=2,
        )

    ax.set_xlabel("Preset (lower = slower, higher quality)")
    ax.set_ylabel("VMAF Score")
    ax.set_title(f"{study_name}: VMAF Score (Mean and P5) vs Preset")
    ax.legend(title="Metric", loc="best", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(df["preset"].unique()))

    output_path = output_dir / f"{study_name}_vs_preset_vmaf_combined.png"
    plt.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_metric_trio(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    output_dir: Path,
    study_name: str,
    higher_is_better: bool = True,
) -> None:
    """
    Generate the trio of plots for a metric:
      1. Heatmap (CRF vs Preset)
      2. Line chart vs CRF
      3. Line chart vs Preset

    Special case: vmaf_combined generates combined mean+p5 plots.
    """
    # Special handling for combined VMAF plot
    if metric == "vmaf_combined":
        if "vmaf_mean" not in df.columns or "vmaf_p5" not in df.columns:
            print(f"Skipping {metric}: vmaf_mean or vmaf_p5 not available", file=sys.stderr)
            return
        print(f"\nGenerating plots for: {metric_label}")
        # Still generate heatmaps for mean and p5 separately
        plot_heatmap(df, "vmaf_mean", "VMAF Mean Score", output_dir, study_name, higher_is_better)
        plot_heatmap(df, "vmaf_p5", "VMAF 5th Percentile", output_dir, study_name, higher_is_better)
        # Combined line plots
        plot_vmaf_combined_vs_crf(df, output_dir, study_name)
        plot_vmaf_combined_vs_preset(df, output_dir, study_name)
        return

    if metric not in df.columns or df[metric].isna().all():
        print(f"Skipping {metric}: no data available", file=sys.stderr)
        return

    print(f"\nGenerating plots for: {metric_label}")

    plot_heatmap(df, metric, metric_label, output_dir, study_name, higher_is_better)
    plot_vs_crf(df, metric, metric_label, output_dir, study_name)
    plot_vs_preset(df, metric, metric_label, output_dir, study_name)


def plot_clip_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    study_name: str,
) -> None:
    """
    Plot per-clip comparison: one line per clip, x-axis = preset.

    Uses line plots instead of bar charts for clearer trend comparison.
    Generates separate figures for key metrics.
    """
    clips = df["source_clip"].unique()
    if len(clips) < 2:
        print("Skipping clip comparison (only one clip)", file=sys.stderr)
        return

    # Metrics to compare per clip
    clip_metrics = [
        ("vmaf_mean", "VMAF Mean Score"),
        ("vmaf_per_bpp", "VMAF per bpp"),
        ("bpp", "Bitrate per Pixel"),
    ]

    colors, markers = get_line_colors_and_markers(len(clips))

    for metric, metric_label in clip_metrics:
        if metric not in df.columns:
            continue

        # Aggregate by clip and preset (mean over CRF values)
        clip_data = df.groupby(["source_clip", "preset"])[metric].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 7))

        for clip, color, marker in zip(clips, colors, markers, strict=True):
            subset = clip_data[clip_data["source_clip"] == clip].sort_values("preset")
            if subset.empty:
                continue
            # Shorten clip name for legend
            clip_short = clip.replace(".mp4", "").replace(".mkv", "")
            ax.plot(
                subset["preset"],
                subset[metric],
                marker=marker,
                linewidth=2,
                markersize=8,
                label=clip_short,
                color=color,
            )

        ax.set_xlabel("Preset (lower = slower, higher quality)")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{study_name}: {metric_label} by Clip vs Preset")
        ax.legend(title="Clip", loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(df["preset"].unique()))

        output_path = output_dir / f"{study_name}_clip_{metric}.png"
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
        plt.close(fig)


def export_csv(df: pd.DataFrame, output_dir: Path, study_name: str) -> None:
    """Export raw dataframe to CSV for further analysis."""
    output_path = output_dir / f"{study_name}_raw_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def export_aggregated_csv(df: pd.DataFrame, output_dir: Path, study_name: str) -> None:
    """Export aggregated dataframe to CSV."""
    output_path = output_dir / f"{study_name}_aggregated.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def generate_summary_report(
    df: pd.DataFrame,
    agg_df: pd.DataFrame,
    analysis_data: dict[str, Any],
    output_dir: Path,
    study_name: str,
) -> None:
    """Generate a text summary report with key findings."""
    report_path = output_dir / f"{study_name}_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Analysis Report: {study_name}\n")
        f.write("=" * 80 + "\n\n")

        # Study metadata
        f.write("Study Metadata\n")
        f.write("-" * 40 + "\n")
        f.write(f"Analysis Date: {analysis_data['analysis_date']}\n")
        f.write(f"VMAF Model: {analysis_data['vmaf_model']}\n")
        f.write(f"Clips Analyzed: {analysis_data['clips_analyzed']}\n")
        f.write(f"Total Encodings: {analysis_data['total_encodings_analyzed']}\n")
        f.write(f"Metrics: {', '.join(analysis_data['metrics_calculated']).upper()}\n")
        f.write("\n")

        # Parameter ranges
        f.write("Parameter Ranges\n")
        f.write("-" * 40 + "\n")
        f.write(f"Presets: {sorted(int(p) for p in df['preset'].unique())}\n")
        f.write(f"CRF values: {sorted(int(c) for c in df['crf'].unique())}\n")
        f.write("\n")

        # Key statistics (aggregated across clips)
        f.write("Aggregated Statistics (Mean Across Clips)\n")
        f.write("-" * 40 + "\n")

        for metric, label, _ in METRICS:
            if metric in agg_df.columns and not agg_df[metric].isna().all():
                min_val = agg_df[metric].min()
                max_val = agg_df[metric].max()
                f.write(f"{label}:\n")
                f.write(f"  Range: {min_val:.3f} - {max_val:.3f}\n")

        f.write("\n")

        # Best configurations
        f.write("Best Configurations (Aggregated)\n")
        f.write("-" * 40 + "\n")

        if "vmaf_mean" in agg_df.columns:
            best = agg_df.loc[agg_df["vmaf_mean"].idxmax()]
            f.write("Highest VMAF Mean:\n")
            f.write(f"  Preset {int(best['preset'])}, CRF {int(best['crf'])}\n")
            f.write(f"  VMAF: {best['vmaf_mean']:.2f}\n")
            if "bpp" in agg_df.columns:
                f.write(f"  bpp: {best['bpp']:.4f}\n")
            f.write("\n")

        if "vmaf_per_bpp" in agg_df.columns and not agg_df["vmaf_per_bpp"].isna().all():
            best = agg_df.loc[agg_df["vmaf_per_bpp"].idxmax()]
            f.write("Best Quality Efficiency (VMAF per bpp):\n")
            f.write(f"  Preset {int(best['preset'])}, CRF {int(best['crf'])}\n")
            f.write(f"  VMAF per bpp: {best['vmaf_per_bpp']:.2f}\n")
            f.write(f"  VMAF: {best['vmaf_mean']:.2f}, bpp: {best['bpp']:.4f}\n")
            f.write("\n")

        if "bpp" in agg_df.columns:
            smallest = agg_df.loc[agg_df["bpp"].idxmin()]
            f.write("Lowest Bitrate (bpp):\n")
            f.write(f"  Preset {int(smallest['preset'])}, CRF {int(smallest['crf'])}\n")
            f.write(f"  bpp: {smallest['bpp']:.4f}\n")
            if "vmaf_mean" in agg_df.columns:
                f.write(f"  VMAF: {smallest['vmaf_mean']:.2f}\n")
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

  # Analyze specific metrics only
  %(prog)s baseline_sweep --metrics vmaf_mean vmaf_per_bpp

  # Skip per-clip comparison plots
  %(prog)s baseline_sweep --no-clip-plots

Available metrics:
  vmaf_combined       - VMAF Mean and P5 (combined plot)
  bpp                 - Bitrate per Pixel
  vmaf_per_bpp        - VMAF per bpp (quality efficiency)
  encoding_time_s     - Encoding Time
  vmaf_per_time       - VMAF per Encoding Second
  vmaf_per_bpp_per_time - Combined efficiency metric
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
        "--metrics",
        nargs="+",
        choices=[m[0] for m in METRICS],
        help="Which metrics to plot (default: all)",
    )

    parser.add_argument(
        "--no-clip-plots",
        action="store_true",
        help="Skip per-clip comparison plots",
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
                if d.is_dir():
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
    print(f"Output directory: {output_dir}")

    # Load data
    analysis_data = load_json(analysis_file)
    df = prepare_dataframe(analysis_data)

    if df.empty:
        print("Error: No successful encodings found in analysis", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} successful encodings")
    print(f"Clips: {df['source_clip'].nunique()}")
    print(f"Presets: {sorted(df['preset'].unique())}")
    print(f"CRF values: {sorted(df['crf'].unique())}")

    # Aggregate by parameters (mean across clips)
    agg_df = aggregate_by_params(df)
    print(f"Aggregated to {len(agg_df)} parameter combinations")

    # Determine which metrics to plot
    metrics_to_plot = args.metrics or [m[0] for m in METRICS]
    metric_info = {m[0]: (m[1], m[2]) for m in METRICS}

    # Generate metric trio plots (heatmap + 2 line charts)
    for metric in metrics_to_plot:
        if metric in metric_info:
            label, higher_is_better = metric_info[metric]
            plot_metric_trio(agg_df, metric, label, output_dir, args.study_name, higher_is_better)

    # Per-clip comparison plots (optional)
    if not args.no_clip_plots:
        print("\nGenerating per-clip comparison plots...")
        plot_clip_comparison(df, output_dir, args.study_name)

    # Export CSV
    if not args.no_csv:
        print("\nExporting CSV files...")
        export_csv(df, output_dir, args.study_name)
        export_aggregated_csv(agg_df, output_dir, args.study_name)

    # Generate summary report
    if not args.no_report:
        print("\nGenerating summary report...")
        generate_summary_report(df, agg_df, analysis_data, output_dir, args.study_name)

    print(f"\n✅ Analysis complete! Results in: {output_dir}")


if __name__ == "__main__":
    main()
