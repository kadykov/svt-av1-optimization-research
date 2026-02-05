# Visualization Guide

This guide explains how to analyze and visualize encoding study results.

## Quick Start

```bash
# After running a study and analysis:
just visualize-study baseline_sweep
```

This generates:
- **WebP plots**: For each metric, three plots (heatmap, vs CRF, vs preset)
- **CSV exports**: Raw data and aggregated data in tabular format
- **Text report**: Summary statistics and best configurations

## Output Structure

Results are saved to `results/<study_name>/`. The visualization system generates a comprehensive set of plots:

### Metric Plots (Trio Format)

For each metric, three complementary visualizations are generated:

1. **Heatmap**: `<study>_heatmap_<metric>.webp` - Shows full parameter space (preset × CRF)
2. **vs CRF**: `<study>_vs_crf_<metric>.webp` - Line plot with one line per preset
3. **vs Preset**: `<study>_vs_preset_<metric>.webp` - Line plot with one line per CRF

### Available Metrics

**All metrics are normalized per-frame-per-pixel for fair comparison across different resolutions, durations, and frame rates.**

- **vmaf_combined** - VMAF Mean and 5th Percentile (combined plot)
- **bytes_per_frame_per_pixel** - File size efficiency (bytes per pixel per frame)
- **bytes_per_vmaf_per_frame_per_pixel** - Inverted efficiency: "How many bytes per pixel per frame do I need to achieve this VMAF score?" (lower is better)
- **bytes_per_p5_vmaf_per_frame_per_pixel** - Same as above, but for P5-VMAF (worst-case quality)
- **encoding_time_per_frame_per_pixel** - Computational cost (milliseconds per megapixel per frame)
- **bytes_per_vmaf_per_encoding_time** - Combined efficiency: bytes per VMAF point per encoding second (lower is better)
- **bytes_per_p5_vmaf_per_encoding_time** - Combined P5-VMAF efficiency

### Per-Clip Analysis

- `<study>_clip_vmaf_combined.webp` - VMAF (Mean and P5) by clip vs preset
- `<study>_clip_bytes_per_frame_per_pixel.webp` - Bytes per frame per pixel by clip vs preset
- `<study>_clip_bytes_per_vmaf_per_frame_per_pixel.webp` - Quality efficiency by clip vs preset
- `<study>_clip_vs_crf_vmaf_combined.webp` - VMAF (Mean and P5) by clip vs CRF
- `<study>_clip_vs_crf_bytes_per_frame_per_pixel.webp` - Bytes per frame per pixel by clip vs CRF
- `<study>_clip_vs_crf_bytes_per_vmaf_per_frame_per_pixel.webp` - Quality efficiency by clip vs CRF

### Duration Analysis

- `<study>_duration_vmaf_per_bpp_frames.webp` - Efficiency vs clip length (frames)
- `<study>_duration_vmaf_per_bpp_pixels.webp` - Efficiency vs resolution (total pixels)
- `<study>_duration_p5_vmaf_per_bpp_frames.webp` - P5-VMAF efficiency vs frames
- `<study>_duration_p5_vmaf_per_bpp_pixels.webp` - P5-VMAF efficiency vs pixels

### Data Exports

- `<study>_raw_data.csv` - All per-encoding metrics
- `<study>_aggregated.csv` - Metrics averaged across clips per parameter combination
- `<study>_report.txt` - Human-readable summary with best configurations

### Example File List

```
results/baseline_sweep/
├── baseline_sweep_heatmap_vmaf_mean.webp
├── baseline_sweep_heatmap_vmaf_p5.webp
├── baseline_sweep_heatmap_bpp.webp
├── baseline_sweep_vs_crf_vmaf_combined.webp
├── baseline_sweep_vs_preset_vmaf_combined.webp
├── baseline_sweep_vs_crf_vmaf_per_bpp.webp
├── baseline_sweep_vs_preset_vmaf_per_bpp.webp
├── baseline_sweep_clip_vmaf_combined.webp
├── baseline_sweep_clip_bytes_per_frame_per_pixel.webp
├── baseline_sweep_clip_bytes_per_vmaf_per_frame_per_pixel.webp
├── baseline_sweep_clip_vs_crf_vmaf_combined.webp
├── baseline_sweep_clip_vs_crf_bytes_per_frame_per_pixel.webp
├── baseline_sweep_clip_vs_crf_bytes_per_vmaf_per_frame_per_pixel.webp
├── baseline_sweep_duration_vmaf_per_bpp_frames.webp
├── baseline_sweep_raw_data.csv
├── baseline_sweep_aggregated.csv
└── baseline_sweep_report.txt
```

## Understanding the Plots

### Heatmap Views
Shows the full parameter space at once. Darker/brighter colors indicate different metric values depending on whether higher is better (e.g., VMAF) or lower is better (e.g., bpp, encoding time).

**Key insights:**
- Visualize entire parameter space at once
- Identify optimal parameter ranges
- Spot patterns and trade-off zones
- Find "sweet spots" where quality plateaus

**Best for:** Understanding the overall landscape of parameter combinations.

### Line Plots vs CRF
Shows how metrics change with CRF values, with separate lines for each preset.

**Key insights:**
- Lower CRF = higher quality but larger files (higher bpp)
- Compare preset efficiency at same CRF level
- Identify where quality/efficiency starts to drop
- See preset-specific compression curves

**Best for:** Choosing CRF value for a given preset and quality target.

### Line Plots vs Preset
Shows how metrics change across presets, with separate lines for each CRF.

**Key insights:**
- Faster presets (higher numbers) encode quicker but may be less efficient
- Compare encoding speed vs quality trade-offs
- See consistency across different quality levels
- Identify which presets work well together

**Best for:** Choosing preset for a given quality target and time budget.

### Per-Clip Comparison
Shows how different source clips respond to encoding parameters. Useful for understanding content-dependent behavior.

**Two complementary views:**
- **vs Preset plots**: Show how clips respond to different presets (averaged over CRF values)
- **vs CRF plots**: Show how clips respond to different CRF values (averaged over presets)

**Key insights:**
- Content-dependent encoding behavior
- Which clips are "harder" to encode efficiently
- Consistency across different content types
- Identify outliers or problematic clips
- Compare behavior across parameter dimensions

**Best for:** Understanding how your parameter choices generalize across content and identifying which parameter (preset vs CRF) has more content-dependent effects.

### Duration Analysis
Shows relationship between clip characteristics (length, resolution) and encoding efficiency.

**Key insights:**
- How clip duration affects quality efficiency
- Resolution impact on compression
- Identify if efficiency varies with video complexity
- Spot trends across different content lengths

**Best for:** Understanding if results generalize to different video durations/resolutions.

## Command Reference

### Basic Usage

```bash
# Generate all plots and exports
just visualize-study baseline_sweep

# Custom output directory
python scripts/visualize_study.py baseline_sweep --output custom_analysis/

# Generate specific metrics only
python scripts/visualize_study.py baseline_sweep --metrics vmaf_combined bpp

# Skip per-clip comparison plots
python scripts/visualize_study.py baseline_sweep --no-clip-plots

# Skip duration analysis
python scripts/visualize_study.py baseline_sweep --no-duration-analysis

# Skip CSV export
python scripts/visualize_study.py baseline_sweep --no-csv

# Skip text report
python scripts/visualize_study.py baseline_sweep --no-report

# Minimal output (just aggregated metrics)
python scripts/visualize_study.py baseline_sweep --no-clip-plots --no-duration-analysis
```

### Available Metrics

Use with `--metrics` flag to generate plots for specific metrics only:

- `vmaf_combined` - VMAF Mean and P5 (combined plot)
- `bytes_per_frame_per_pixel` - File size efficiency (bytes per pixel per frame)
- `bytes_per_vmaf_per_frame_per_pixel` - Inverted efficiency: bytes needed per VMAF point (lower is better)
- `bytes_per_p5_vmaf_per_frame_per_pixel` - Bytes needed per P5-VMAF point (worst-case quality)
- `encoding_time_per_frame_per_pixel` - Computational cost (ms per megapixel per frame)
- `bytes_per_vmaf_per_encoding_time` - Combined efficiency: bytes per VMAF per second (lower is better)
- `bytes_per_p5_vmaf_per_encoding_time` - Combined P5-VMAF efficiency

**Example:**
```bash
# Only generate quality efficiency plots
python scripts/visualize_study.py baseline_sweep --metrics bytes_per_vmaf_per_frame_per_pixel bytes_per_p5_vmaf_per_frame_per_pixel
```

## CSV Export

The visualization system exports two CSV files for detailed analysis:

### Raw Data CSV (`<study>_raw_data.csv`)

Contains per-encoding metrics for all individual encodings:

**Identifiers:**
- `output_file` - Encoded filename
- `source_clip` - Source clip name

**Parameters:**
- `preset` - Encoder preset used
- `crf` - CRF quality setting

**Video Properties:**
- `width`, `height` - Resolution in pixels
- `fps` - Frame rate
- `duration` - Clip duration in seconds
- `num_frames` - Total number of frames
- `total_pixels` - Total pixels across all frames (num_frames * width * height)

**File Metrics:**
- `file_size_mb` - Output file size in MB
- `file_size_bytes` - Output file size in bytes
- `bitrate_kbps` - Average bitrate (if available)
- `bpp` - Legacy metric: bits per pixel per frame (for backward compatibility)
- `bytes_per_frame_per_pixel` - **New normalized metric**: bytes per pixel per frame

**Performance:**
- `encoding_time_s` - Wall clock encoding time
- `encoding_fps` - Frames per second during encoding
- `encoding_time_per_frame_per_pixel` - **New normalized metric**: encoding time per megapixel per frame (ms)

**VMAF Metrics:**
- `vmaf_mean` - Arithmetic mean VMAF score
- `vmaf_harmonic_mean` - Worst-case quality indicator
- `vmaf_min` / `vmaf_p1` / `vmaf_p5` - Lowest quality frames
- `vmaf_median` / `vmaf_p25` / `vmaf_p75` / `vmaf_p95` - Percentiles
- `vmaf_std` - Standard deviation

**Other Quality Metrics:**
- `psnr_avg` - Average PSNR score
- `ssim_avg` - Average SSIM score

**Calculated Efficiency Metrics (inverted - lower is better):**
- `bytes_per_vmaf_per_frame_per_pixel` - **New**: bytes per quality point per pixel per frame
- `bytes_per_p5_vmaf_per_frame_per_pixel` - **New**: worst-case quality efficiency
- `bytes_per_vmaf_per_encoding_time` - **New**: combined efficiency metric
- `bytes_per_p5_vmaf_per_encoding_time` - **New**: combined P5-VMAF efficiency
- `vmaf_per_bpp` - Legacy metric (kept for backward compatibility)
- `p5_vmaf_per_bpp` - Legacy metric (kept for backward compatibility)

### Aggregated CSV (`<study>_aggregated.csv`)

Contains metrics averaged across all clips for each parameter combination (preset × CRF). This represents the "typical" performance for each configuration.

**Structure:** Same columns as raw data, but with values averaged across clips.

**Usage:** Use aggregated data for:
- Comparing parameter configurations
- Finding optimal settings for general use
- Plotting overall trends

Use raw data for:
- Understanding per-clip variation
- Identifying problematic clips
- Content-specific analysis

## Using CSV Data for Custom Analysis

**Example uses:**
```python
import pandas as pd

# Load aggregated data
df = pd.read_csv('results/baseline_sweep/baseline_sweep_aggregated.csv')

# Find best efficiency configurations (VMAF > 90, low bpp)
efficient = df[(df['vmaf_mean'] > 90) & (df['bpp'] < 0.03)]
print(efficient[['preset', 'crf', 'vmaf_mean', 'bpp', 'vmaf_per_bpp']])

# Compare presets at same CRF
p6_crf30 = df[(df['preset'] == 6) & (df['crf'] == 30)]
p10_crf30 = df[(df['preset'] == 10) & (df['crf'] == 30)]
print(f"Preset 6: {p6_crf30['encoding_time_s'].values[0]:.2f}s")
print(f"Preset 10: {p10_crf30['encoding_time_s'].values[0]:.2f}s")

# Find sweet spot: good quality with reasonable encoding time
sweet_spot = df[
    (df['vmaf_mean'] > 92) &
    (df['vmaf_p5'] > 88) &  # Good worst-case quality
    (df['encoding_time_s'] < 10)
].sort_values('vmaf_per_bpp', ascending=False).head(5)
print(sweet_spot[['preset', 'crf', 'vmaf_mean', 'vmaf_p5', 'bpp', 'encoding_time_s']])
```

## Analysis Workflow

### 1. Complete the Encoding Pipeline

```bash
# Extract clips
just extract-clips 10 15 30

# Run encoding study
just encode-study baseline_sweep

# Calculate quality metrics
just analyze-study baseline_sweep
```

### 2. Generate Visualizations

```bash
# Generate all plots
just visualize-study baseline_sweep

# View results
ls -lh results/baseline_sweep/
```

### 3. Interpret Results

**Check the text report first:**
```bash
cat results/baseline_sweep/baseline_sweep_report.txt
```

This gives you:
- Study metadata (date, VMAF model, number of clips)
- Parameter ranges tested
- Aggregated statistics for all metrics
- Best configurations for different goals

**Review plots systematically:**

1. **Start with VMAF heatmaps** - Get overall sense of parameter space
   - `baseline_sweep_heatmap_vmaf_mean.webp` - Overall quality
   - `baseline_sweep_heatmap_vmaf_p5.webp` - Worst-case quality

2. **Check efficiency metrics** - Find sweet spots
   - `baseline_sweep_heatmap_vmaf_per_bpp.webp` - Quality per compression
   - `baseline_sweep_heatmap_vmaf_per_bpp_per_time.webp` - Combined efficiency

3. **Examine line plots for trends** - Understand parameter behavior
   - `baseline_sweep_vs_crf_vmaf_combined.webp` - How CRF affects quality
   - `baseline_sweep_vs_preset_vmaf_combined.webp` - How preset affects quality

4. **Review per-clip analysis** - Check generalization
   - `baseline_sweep_clip_vmaf_mean.webp` - Content-dependent behavior
   - `baseline_sweep_clip_vmaf_per_bpp.webp` - Efficiency across content

**Use CSV for custom analysis:**
```python
import pandas as pd

# Load aggregated data (averaged across clips)
df = pd.read_csv('results/baseline_sweep/baseline_sweep_aggregated.csv')

# Find configurations with VMAF > 90 and efficient compression
efficient = df[(df['vmaf_mean'] > 90) & (df['bytes_per_frame_per_pixel'] < 0.004)]
print(efficient[['preset', 'crf', 'vmaf_mean', 'bytes_per_frame_per_pixel', 'bytes_per_vmaf_per_frame_per_pixel']].head())

# Compare encoding time across presets (normalized)
print(df.groupby('preset')['encoding_time_per_frame_per_pixel'].mean())

# Find best overall efficiency (lower is better)
best_efficiency = df.nsmallest(5, 'bytes_per_vmaf_per_encoding_time')
print(best_efficiency[['preset', 'crf', 'vmaf_mean', 'bytes_per_vmaf_per_encoding_time']])
```

## Integration with Other Studies

The visualization system works with any study configuration:

```bash
# Film grain study
just encode-study film_grain
just analyze-study film_grain
just visualize-study film_grain

# Screen content study
just encode-study screen_content
just analyze-study screen_content
just visualize-study screen_content
```

Each study generates its own visualizations in `results/<study_name>/`.

## Customization

The visualization script can be extended:

**Add new plot types:**
Edit `scripts/visualize_study.py` and add a new plot function following the existing pattern.

**Modify plot appearance:**
Seaborn theme and matplotlib settings are at the top of the script.

**Change data aggregation:**
The `prepare_dataframe()` function controls how analysis metadata is converted to tabular format.

## Troubleshooting

**Missing dependencies:**
```bash
pip install pandas numpy matplotlib seaborn
# Or: just install-dev
```

**"Study not found" error:**
```bash
# List available studies
ls data/encoded/

# Make sure analysis is complete
just analyze-study baseline_sweep
```

**"No successful encodings" error:**
Check that encodings succeeded and analysis completed without errors. Review the analysis metadata:
```bash
cat data/encoded/baseline_sweep/analysis_metadata.json | head -50
```

**Plots look wrong or incomplete:**
- Ensure analysis completed successfully for all encodings
- Check that clips have resolution metadata in `data/test_clips/clip_metadata.json`
- Verify bpp calculations are correct (requires bitrate and resolution data)

**WebP files not opening:**
WebP is a modern image format. Most image viewers support it, but if yours doesn't:
- Use a web browser to view the files
- Convert to PNG: `for f in *.webp; do ffmpeg -i "$f" "${f%.webp}.png"; done`

## Key Metrics Explained

### Quality Metrics
- **VMAF Mean**: Average quality across all frames (0-100 scale, higher is better)
- **VMAF P5**: 5th percentile - worst 5% of frames (indicates quality floor)
- **VMAF Harmonic Mean**: Another worst-case indicator (emphasizes low scores)

### File Size Metrics (normalized per-frame-per-pixel)
- **bytes_per_frame_per_pixel**: File size divided by total pixels across all frames
  - Formula: `file_size_bytes / (num_frames * width * height)`
  - Allows fair comparison across different resolutions, durations, and frame rates
  - Lower values = better compression

### Efficiency Metrics (inverted form - lower is better)
- **bytes_per_vmaf_per_frame_per_pixel**: "How many bytes per pixel per frame do I need to achieve this VMAF score?"
  - Formula: `bytes_per_frame_per_pixel / vmaf_mean`
  - Lower values = more efficient (fewer bytes needed for same quality)
  - Intuitive interpretation: cost per quality point

- **bytes_per_p5_vmaf_per_frame_per_pixel**: Same as above, but for worst-case (P5) VMAF
  - Ensures good quality even in difficult frames

### Encoding Time Metrics (normalized)
- **encoding_time_per_frame_per_pixel**: Computational cost per frame per megapixel (in milliseconds)
  - Formula: `(encoding_time_s * 1000) / (total_pixels / 1_000_000)`
  - Allows comparison of encoding speed across different video complexities
  - Lower values = faster encoding

### Combined Efficiency Metrics
- **bytes_per_vmaf_per_encoding_time**: Overall efficiency considering both file size and encoding time
  - Formula: `file_size_bytes / vmaf_mean / encoding_time_s`
  - The per-frame-per-pixel terms cancel out mathematically
  - Lower values = better overall efficiency
  - Represents: "bytes per quality point per second of encoding time"

### Why Normalized Metrics?

Without normalization, raw metrics are misleading:
- **Encoding time**: A 4K 60fps video naturally takes longer than 720p 30fps
- **Bitrate**: Higher frame rates require more bits for same quality
- **File size**: Longer videos have larger files

Normalized metrics allow fair comparisons:
- Same preset on 1080p vs 4K clips
- Same CRF on 24fps vs 60fps content
- Same configuration on 5s vs 30s clips

### Interpreting Values
- **bytes_per_frame_per_pixel** (0.002 - 0.012): Typical range for AV1 encoding
  - < 0.003: Highly compressed (check quality!)
  - 0.003-0.006: Good compression with decent quality
  - > 0.010: High bitrate, near-lossless

- **bytes_per_vmaf_per_frame_per_pixel** (~0.00003-0.0001): Efficiency metric
  - Lower is better - fewer bytes needed per quality point
  - Compare different configurations to find sweet spots

- **encoding_time_per_frame_per_pixel** (7-130 ms/megapixel): Computational cost
  - < 20: Fast presets (8-10)
  - 20-60: Balanced presets (5-7)
  - > 100: Slow, high-quality presets (2-4)

## Future Enhancements

Planned additions:
- Interactive HTML reports with Plotly
- Pareto frontier visualization (optimal configurations)
- Convex hull analysis for rate-distortion curves
- Statistical significance testing across clips
- Automated recommendation system
- Multi-study comparison plots
- Automated recommendations based on use case
- Content complexity analysis integration
