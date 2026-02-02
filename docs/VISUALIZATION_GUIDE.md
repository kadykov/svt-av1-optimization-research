# Visualization Guide

This guide explains how to analyze and visualize encoding study results.

## Quick Start

```bash
# After running a study and analysis:
just visualize-study baseline_sweep
```

This generates:
- **PNG plots**: Rate-distortion curves, speed-quality tradeoffs, parameter heatmaps
- **CSV export**: All metrics in tabular format for custom analysis
- **Text report**: Summary statistics and best configurations

## Output Structure

Results are saved to `results/<study_name>/`:

```
results/baseline_sweep/
├── baseline_sweep_rate_distortion.png    # VMAF vs file size curves
├── baseline_sweep_speed_quality.png       # Encoding time vs quality
├── baseline_sweep_parameter_impact.png    # Preset/CRF heatmaps
├── baseline_sweep_clip_comparison.png     # Per-clip analysis
├── baseline_sweep_vmaf_distribution.png   # Quality consistency
├── baseline_sweep_analysis.csv            # All metrics in CSV
└── baseline_sweep_report.txt              # Summary report
```

## Plot Types

### Rate-Distortion Curves
Shows VMAF quality vs file size for different presets.

**Key insights:**
- Lower CRF = higher quality but larger files
- Faster presets (higher numbers) may sacrifice efficiency
- Identifies "sweet spots" for quality/size tradeoff

### Speed vs Quality
Compares encoding time to resulting quality.

**Key insights:**
- Preset impact on encoding speed
- Quality per encoding second (efficiency metric)
- Time investment vs quality gain

### Parameter Impact (Heatmaps)
Shows how preset and CRF combinations affect:
- VMAF mean score
- File size
- Encoding time
- VMAF per MB (efficiency)

**Key insights:**
- Visualizes entire parameter space at once
- Identifies optimal parameter ranges
- Highlights trade-off zones

### Clip Comparison
Compares how different source clips respond to encoding.

**Key insights:**
- Content-dependent behavior
- Which clips are harder to encode
- Consistency across content types

### VMAF Distribution
Shows quality consistency through percentiles and mean vs harmonic mean.

**Key insights:**
- Frame-level quality variation
- Worst-case quality (P5, harmonic mean)
- Encoding stability

## Command Reference

### Basic Usage

```bash
# Generate all plots and exports
just visualize-study baseline_sweep

# Custom output directory
just visualize-to baseline_sweep custom_analysis/

# Generate specific plots only
just visualize-plots baseline_sweep rate-distortion speed-quality

# Skip CSV export
python scripts/visualize_study.py baseline_sweep --no-csv

# Skip text report
python scripts/visualize_study.py baseline_sweep --no-report
```

### Available Plot Types

Use with `--plots` flag:
- `rate-distortion` - VMAF vs file size curves
- `speed-quality` - Encoding time vs quality
- `parameter-impact` - Preset/CRF heatmaps
- `clip-comparison` - Per-clip quality comparison
- `vmaf-distribution` - Quality consistency analysis
- `all` - Generate all plots (default)

## CSV Export

The CSV export includes all metrics for custom analysis:

**Identifiers:**
- `output_file` - Encoded filename
- `source_clip` - Source clip name

**Parameters:**
- `preset` - Encoder preset used
- `crf` - CRF quality setting

**File Metrics:**
- `file_size_mb` - Output file size
- `bitrate_kbps` - Average bitrate (if available)

**Performance:**
- `encoding_time_s` - Wall clock encoding time
- `encoding_fps` - Frames per second
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

**Efficiency Metrics:**
- `vmaf_per_mb` - Quality per file size
- `quality_per_encode_s` - Quality per encoding time

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
- Overall quality and file size ranges
- Best quality configuration
- Best efficiency configuration
- Smallest file configuration
- Parameter impact summary

**Review plots:**
- Start with rate-distortion to understand quality/size tradeoffs
- Check parameter impact heatmaps for full parameter space view
- Review speed-quality if encoding time is a concern
- Examine clip comparison for content-dependent insights

**Use CSV for custom analysis:**
```bash
# Example: Load in Python
import pandas as pd
df = pd.read_csv('results/baseline_sweep/baseline_sweep_analysis.csv')

# Find configurations with VMAF > 90 and small file size
efficient = df[(df['vmaf_mean'] > 90) & (df['file_size_mb'] < 5)]
print(efficient[['preset', 'crf', 'vmaf_mean', 'file_size_mb']])
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
just list-encoded

# Make sure analysis is complete
just analyze-study baseline_sweep
```

**"No successful encodings" error:**
Check that encodings succeeded and analysis completed without errors.

**Display issues in dev containers:**
The "Authorization required" warning is normal in headless environments. PNG files are still generated correctly.

## Future Enhancements

Planned additions:
- Interactive HTML reports with Plotly
- Pareto frontier visualization (optimal configurations)
- Multi-study comparison plots
- Automated recommendations based on use case
- Content complexity analysis integration
