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

- **vmaf_combined** - VMAF Mean and 5th Percentile (combined plot)
- **bpp** - Bitrate per Pixel (compression rate)
- **vmaf_per_bpp** - VMAF per bpp (quality efficiency)
- **p5_vmaf_per_bpp** - P5-VMAF per bpp (worst-case quality efficiency)
- **encoding_time_s** - Encoding time in seconds
- **vmaf_per_time** - VMAF per encoding second (speed efficiency)
- **vmaf_per_bpp_per_time** - Combined efficiency: VMAF per bpp per second
- **p5_vmaf_per_bpp_per_time** - P5-VMAF combined efficiency

### Per-Clip Analysis

- `<study>_clip_vmaf_mean.webp` - VMAF by clip and preset
- `<study>_clip_bpp.webp` - Bitrate per pixel by clip and preset
- `<study>_clip_vmaf_per_bpp.webp` - Quality efficiency by clip

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
├── baseline_sweep_clip_vmaf_mean.webp
├── baseline_sweep_clip_bpp.webp
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

**Key insights:**
- Content-dependent encoding behavior
- Which clips are "harder" to encode efficiently
- Consistency across different content types
- Identify outliers or problematic clips

**Best for:** Understanding how your parameter choices generalize across content.

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
- `bpp` - Bitrate per Pixel
- `vmaf_per_bpp` - VMAF per bpp (quality efficiency)
- `p5_vmaf_per_bpp` - P5-VMAF per bpp (quality efficiency)
- `encoding_time_s` - Encoding Time
- `vmaf_per_time` - VMAF per Encoding Second
- `vmaf_per_bpp_per_time` - Combined efficiency metric
- `p5_vmaf_per_bpp_per_time` - P5-VMAF combined efficiency metric

**Example:**
```bash
# Only generate quality efficiency plots
python scripts/visualize_study.py baseline_sweep --metrics vmaf_per_bpp p5_vmaf_per_bpp
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

**File Metrics:**
- `file_size_mb` - Output file size in MB
- `bitrate_kbps` - Average bitrate (if available)
- `bpp` - Bitrate per pixel (calculated from bitrate and resolution)

**Performance:**
- `encoding_time_s` - Wall clock encoding time
- `encoding_fps` - Frames per second during encoding
- `analysis_time_s` - VMAF calculation time

**VMAF Metrics:**
- `vmaf_mean` - Arithmetic mean VMAF score
- `vmaf_harmonic_mean` - Worst-case quality indicator
- `vmaf_min` / `vmaf_p1` / `vmaf_p5` - Lowest quality frames
- `vmaf_median` / `vmaf_p25` / `vmaf_p75` / `vmaf_p95` - Percentiles
- `vmaf_std` - Standard deviation

**Other Quality Metrics:**
- `psnr_avg` - Average PSNR score
- `ssim_avg` - Average SSIM score

**Calculated Efficiency Metrics:**
- `vmaf_per_bpp` - Quality per compression (VMAF / bpp)
- `p5_vmaf_per_bpp` - Worst-case quality per compression
- `vmaf_per_time` - Quality per encoding second
- `vmaf_per_bpp_per_time` - Combined efficiency metric
- `p5_vmaf_per_bpp_per_time` - P5-VMAF combined efficiency

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

# Find configurations with VMAF > 90 and low bpp
efficient = df[(df['vmaf_mean'] > 90) & (df['bpp'] < 0.03)]
print(efficient[['preset', 'crf', 'vmaf_mean', 'bpp', 'vmaf_per_bpp']].head())

# Compare encoding time across presets
print(df.groupby('preset')['encoding_time_s'].mean())
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

### Efficiency Metrics
- **bpp (bits per pixel)**: Bitrate / (width × height × fps) - measures compression rate
- **vmaf_per_bpp**: Quality efficiency - how much quality per bit (higher is better)
- **p5_vmaf_per_bpp**: Worst-case quality efficiency
- **vmaf_per_time**: Quality per encoding second (speed efficiency)
- **vmaf_per_bpp_per_time**: Combined efficiency - quality per compression per time

### Interpreting Values
- **High vmaf_per_bpp** (>3000): Very efficient compression
- **High vmaf_per_time** (>20): Fast encoding with good quality
- **Low bpp** (<0.03): Highly compressed, check quality metrics
- **High VMAF** (>95): Transparent or near-transparent quality
- **Low P5-VMAF gap** (mean - p5 < 5): Consistent quality across frames

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
