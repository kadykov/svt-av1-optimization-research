# End-to-End Analysis Workflow Example

This document demonstrates a complete workflow from encoding to visualization for the baseline_sweep study.

## Complete Workflow

### 1. Prepare Video Clips

```bash
# Download source videos
just fetch-videos

# Extract test clips (10 clips, 15-30 seconds each)
just extract-clips 10 15 30

# Verify clips were created
ls -lh data/test_clips/*.mp4
```

### 2. Run Encoding Study

```bash
# Preview what will be encoded
just dry-run-study baseline_sweep

# Run the encoding (this will take time)
just encode-study baseline_sweep

# Check encoding results
ls -lh data/encoded/baseline_sweep/
cat data/encoded/baseline_sweep/encoding_metadata.json | head -30
```

### 3. Calculate Quality Metrics

```bash
# Analyze with VMAF, PSNR, and SSIM
just analyze-study baseline_sweep

# Or faster: only VMAF
just analyze-vmaf baseline_sweep

# Check analysis results
head -50 data/encoded/baseline_sweep/analysis_metadata.json
```

### 4. Generate Visualizations

```bash
# Generate all plots, CSV, and report
just visualize-study baseline_sweep

# View results
ls -lh results/baseline_sweep/
```

## What Gets Generated

### Visual Analysis (WebP files)

The visualization system generates comprehensive plot sets organized by metric and view type.

#### Metric Trio Plots

For each metric, three complementary views are generated:

**1. Heatmap View** (`<study>_heatmap_<metric>.webp`)
- Shows full parameter space (preset × CRF grid)
- Color intensity indicates metric value
- Example: `baseline_sweep_heatmap_vmaf_mean.webp`

**2. vs CRF Line Plot** (`<study>_vs_crf_<metric>.webp`)
- X-axis: CRF values
- One line per preset
- Shows how metric changes with CRF at each preset
- Example: `baseline_sweep_vs_crf_vmaf_combined.webp`

**3. vs Preset Line Plot** (`<study>_vs_preset_<metric>.webp`)
- X-axis: Preset values
- One line per CRF
- Shows how metric changes across presets at each CRF
- Example: `baseline_sweep_vs_preset_vmaf_per_bpp.webp`

#### Available Metrics

Each metric trio covers (all normalized per-frame-per-pixel):
- **vmaf_combined** - VMAF Mean and P5 (combined plot showing both)
- **bytes_per_frame_per_pixel** - File size efficiency (bytes per pixel per frame)
- **bytes_per_vmaf_per_frame_per_pixel** - Inverted efficiency: bytes needed per VMAF point (lower is better)
- **bytes_per_p5_vmaf_per_frame_per_pixel** - Worst-case quality efficiency
- **encoding_time_per_frame_per_pixel** - Computational cost (ms per megapixel per frame)
- **bytes_per_vmaf_per_encoding_time** - Combined efficiency: bytes per VMAF per second (lower is better)
- **bytes_per_p5_vmaf_per_encoding_time** - Combined P5-VMAF efficiency

#### Per-Clip Comparison Plots

These show content-dependent behavior:
- `baseline_sweep_clip_vmaf_mean.webp` - Quality by clip and preset
- `baseline_sweep_clip_bytes_per_frame_per_pixel.webp` - Compression by clip and preset
- `baseline_sweep_clip_bytes_per_vmaf_per_frame_per_pixel.webp` - Efficiency by clip and preset

**Insights:**
- Which clips are "hard to encode"
- Consistency across different content types
- Outliers or problematic clips

#### Duration Analysis Plots

These show relationships between clip characteristics and efficiency:
- `baseline_sweep_duration_vmaf_per_bpp_frames.webp` - Efficiency vs clip length (frames)
- `baseline_sweep_duration_vmaf_per_bpp_pixels.webp` - Efficiency vs resolution (pixels)
- `baseline_sweep_duration_p5_vmaf_per_bpp_frames.webp` - P5-VMAF efficiency vs frames
- `baseline_sweep_duration_p5_vmaf_per_bpp_pixels.webp` - P5-VMAF efficiency vs pixels

**Insights:**
- How clip duration affects encoding efficiency
- Resolution impact on compression
- Whether results generalize across different video lengths

### Data Export (CSV)

Two CSV files are generated for detailed analysis:

**1. `baseline_sweep_raw_data.csv`**

Per-encoding metrics for all individual encodings. Each row represents one encoding of one clip.

Contains:
- Identifiers: output_file, source_clip
- Parameters: preset, crf
- Video properties: width, height, fps, duration, num_frames, total_pixels
- File metrics: file_size_mb, file_size_bytes, bitrate_kbps
- **New normalized metrics**: bytes_per_frame_per_pixel, encoding_time_per_frame_per_pixel
- Performance: encoding_time_s, encoding_fps
- VMAF: mean, harmonic_mean, min, p1, p5, p25, median, p75, p95, std
- Other metrics: psnr_avg, ssim_avg
- **New efficiency metrics**: bytes_per_vmaf_per_frame_per_pixel, bytes_per_p5_vmaf_per_frame_per_pixel, bytes_per_vmaf_per_encoding_time, bytes_per_p5_vmaf_per_encoding_time
- Legacy metrics (for backward compatibility): bpp, vmaf_per_bpp, p5_vmaf_per_bpp

**2. `baseline_sweep_aggregated.csv`**

Metrics averaged across clips for each parameter combination (preset × CRF). Each row represents the typical performance for one configuration.

Same columns as raw data, but aggregated across all clips.

**Example uses:**
```python
import pandas as pd

# Load aggregated data for parameter comparison
agg_df = pd.read_csv('results/baseline_sweep/baseline_sweep_aggregated.csv')

# Find best efficiency configurations (VMAF > 90, efficient compression)
efficient = agg_df[(agg_df['vmaf_mean'] > 90) & (agg_df['bytes_per_frame_per_pixel'] < 0.004)]
print(efficient[['preset', 'crf', 'vmaf_mean', 'bytes_per_frame_per_pixel', 'bytes_per_vmaf_per_frame_per_pixel']])

# Compare presets at same CRF (using normalized encoding time)
p6_crf30 = agg_df[(agg_df['preset'] == 6) & (agg_df['crf'] == 30)]
p10_crf30 = agg_df[(agg_df['preset'] == 10) & (agg_df['crf'] == 30)]
print(f"Preset 6: {p6_crf30['encoding_time_per_frame_per_pixel'].values[0]:.2f} ms/megapixel")
print(f"Preset 10: {p10_crf30['encoding_time_per_frame_per_pixel'].values[0]:.2f} ms/megapixel")

# Load raw data for per-clip analysis
raw_df = pd.read_csv('results/baseline_sweep/baseline_sweep_raw_data.csv')
# Find which clips are hardest to encode efficiently (higher is worse)
clip_efficiency = raw_df.groupby('source_clip')['bytes_per_vmaf_per_frame_per_pixel'].mean().sort_values(ascending=False)
print("Hardest to encode clips:")
print(clip_efficiency.head())
```

### Text Summary Report

**`baseline_sweep_report.txt`**

Human-readable summary including:
- Study metadata (date, VMAF model, number of clips analyzed)
- Parameter ranges (presets and CRF values tested)
- Aggregated statistics (ranges for all key metrics)
- Best configurations:
  - Highest VMAF Mean
  - Best File Size Efficiency (lowest bytes per VMAF per frame per pixel)
  - Best Overall Efficiency (lowest bytes per VMAF per encoding second)
  - Smallest File Size (lowest bytes per frame per pixel)

**Example output:**
```
Analysis Report: baseline_sweep
================================================================================

Study Metadata
----------------------------------------
Analysis Date: 2026-02-04T19:07:26.798343Z
VMAF Model: vmaf_v0.6.1neg
Clips Analyzed: 5
Total Encodings: 225
Metrics: VMAF, SSIM, PSNR

Parameter Ranges
----------------------------------------
Presets: [2, 3, 4, 5, 6, 7, 8, 9, 10]
CRF values: [20, 25, 30, 35, 40]

Aggregated Statistics (Mean Across Clips)
----------------------------------------
Bitrate per Pixel (bpp):
  Range: 0.021 - 0.095
VMAF per bpp (Quality Efficiency):
  Range: 1030.722 - 4315.950
P5-VMAF per bpp (Quality Efficiency):
  Range: 989.442 - 4115.742

Best Configurations (Aggregated)
----------------------------------------
Highest VMAF Mean:
  Preset 2, CRF 20
  VMAF: 96.46
  bpp: 0.0923

Best Quality Efficiency (VMAF per bpp):
  Preset 3, CRF 40
  VMAF per bpp: 4315.95
  VMAF: 89.85, bpp: 0.0212

Lowest Bitrate (bpp):
  Preset 3, CRF 40
  bpp: 0.0212
  VMAF: 89.85
```

## Interpreting Results

### 1. Start with the Text Report

```bash
cat results/baseline_sweep/baseline_sweep_report.txt
```

This gives you:
- Study metadata and scope
- Parameter ranges tested
- Overall metric ranges
- Best configurations for different goals

### 2. Review VMAF Heatmaps

Open the heatmap files to get the big picture:

**`baseline_sweep_heatmap_vmaf_mean.webp`**:
- Full parameter space visualization
- Look for "sweet spots" where quality is high
- Identify where quality starts to drop significantly

**`baseline_sweep_heatmap_vmaf_p5.webp`**:
- Worst-case quality (5th percentile)
- Should be similar to mean (indicates consistency)
- Large gaps indicate unstable quality

**Key questions:**
- Where does quality plateau?
- Which combinations give good quality?
- Are there "dead zones" to avoid?

### 3. Examine Efficiency Metrics

**`baseline_sweep_heatmap_vmaf_per_bpp.webp`**:
- Quality per compression
- Higher values = more efficient
- Find the "Goldilocks zone" of good quality and low bpp

**`baseline_sweep_heatmap_vmaf_per_bpp_per_time.webp`**:
- Combined efficiency metric
- Balances quality, compression, and encoding speed
- Best for finding practical sweet spots

**Key questions:**
- Which configurations give best "bang for the buck"?
- Where's the efficiency sweet spot?
- What's the trade-off between efficiency and absolute quality?

### 4. Check Parameter Trends with Line Plots

**vs CRF plots** (e.g., `baseline_sweep_vs_crf_vmaf_combined.webp`):
- How does quality change with CRF?
- Compare preset curves
- Identify where diminishing returns start

**vs Preset plots** (e.g., `baseline_sweep_vs_preset_vmaf_per_bpp.webp`):
- How do presets compare at same quality target?
- Speed vs efficiency trade-offs
- Which presets behave consistently?

**Key questions:**
- At what CRF does quality drop noticeably?
- Which preset offers best efficiency at my target quality?
- Is faster preset worth the efficiency loss?

### 5. Review Per-Clip Comparison

**`baseline_sweep_clip_vmaf_mean.webp`** and **`baseline_sweep_clip_vmaf_per_bpp.webp`**:
- Content-dependent encoding behavior
- Which clips are "harder" to encode efficiently
- Consistency across different content types

**Key questions:**
- Do all clips respond similarly to parameter changes?
- Are some clips consistently harder to encode?
- Do results generalize across my content?
- Should I use different settings for different content types?

### 6. Check Duration Analysis

**`baseline_sweep_duration_vmaf_per_bpp_frames.webp`** and **`baseline_sweep_duration_vmaf_per_bpp_pixels.webp`**:
- Relationship between clip characteristics and efficiency
- Whether results scale to different durations/resolutions

**Key questions:**
- Does efficiency change with video length?
- Do shorter clips encode differently than longer ones?
- Is resolution a major factor in compression efficiency?

### 7. Use CSV for Detailed Analysis

For custom questions, load the CSV data:

```python
import pandas as pd

# Load aggregated data
df = pd.read_csv('results/baseline_sweep/baseline_sweep_aggregated.csv')

# Find sweet spot: high quality, good efficiency, reasonable time
sweet_spot = df[
    (df['vmaf_mean'] > 92) &
    (df['vmaf_p5'] > 88) &  # Good worst-case quality
    (df['encoding_time_s'] < 15) &
    (df['vmaf_per_bpp'] > 2000)
].sort_values('vmaf_per_bpp_per_time', ascending=False)

print("Sweet spot configurations:")
print(sweet_spot[['preset', 'crf', 'vmaf_mean', 'bpp', 'encoding_time_s']].head())
```

**Key questions:**
- What's the optimal configuration for my specific requirements?
- How much do I gain/lose by changing one parameter?
- Which metric should I prioritize?

## Customization Examples

### Generate Only Specific Metrics

```bash
# Only VMAF and efficiency plots
python scripts/visualize_study.py baseline_sweep --metrics vmaf_combined vmaf_per_bpp

# Only encoding time analysis
python scripts/visualize_study.py baseline_sweep --metrics encoding_time_s vmaf_per_time
```

### Minimal Output (Fast)

```bash
# Skip per-clip and duration analysis for faster processing
python scripts/visualize_study.py baseline_sweep --no-clip-plots --no-duration-analysis
```

### Custom Analysis Script

Create `custom_analysis.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load aggregated data
df = pd.read_csv('results/baseline_sweep/baseline_sweep_aggregated.csv')

# Custom analysis: Find best preset for streaming
# Target: VMAF > 93, minimize bpp
streaming_configs = df[df['vmaf_mean'] > 93].sort_values('bpp')

print("Best configs for streaming (VMAF > 93):")
print(streaming_configs[['preset', 'crf', 'vmaf_mean', 'bpp', 'encoding_time_s']].head(5))

# Custom plot: Efficiency frontier
plt.figure(figsize=(10, 6))
for preset in sorted(df['preset'].unique()):
    subset = df[df['preset'] == preset].sort_values('bpp')
    plt.plot(subset['bpp'], subset['vmaf_mean'], marker='o', label=f'Preset {preset}')

plt.xlabel('Bitrate per Pixel (bpp)')
plt.ylabel('VMAF Mean Score')
plt.title('Efficiency Frontier by Preset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('custom_crf_analysis.png', dpi=300)
print("Saved: custom_crf_analysis.png")
```

Run it:
```bash
python custom_analysis.py
```

## Comparing Multiple Studies

```bash
# Run multiple studies
just encode-study baseline_sweep
just encode-study film_grain
just encode-study tune_modes

# Analyze all
just analyze-study baseline_sweep
just analyze-study film_grain
just analyze-study tune_modes

# Visualize all
just visualize-study baseline_sweep
just visualize-study film_grain
just visualize-study tune_modes

# Compare results
ls -lh results/*/
```

Then create comparison analysis:

```python
import pandas as pd

# Load all studies
baseline = pd.read_csv('results/baseline_sweep/baseline_sweep_analysis.csv')
grain = pd.read_csv('results/film_grain/film_grain_analysis.csv')
tune = pd.read_csv('results/tune_modes/tune_modes_analysis.csv')

# Add study identifier
baseline['study'] = 'baseline'
grain['study'] = 'film_grain'
tune['study'] = 'tune_modes'

# Combine
all_data = pd.concat([baseline, grain, tune])

# Compare best configurations across studies
print("Best VMAF by study:")
print(all_data.groupby('study')['vmaf_mean'].max())

print("\nBest efficiency by study:")
print(all_data.groupby('study')['vmaf_per_bpp'].max())
```

## Next Steps

After visualizing the baseline_sweep study:

1. **Identify optimal parameters** for your use case
2. **Design focused studies** to explore interesting regions
3. **Test on diverse content** to validate findings
4. **Document decisions** based on empirical data

Example focused study:

```json
{
  "study_name": "optimal_streaming",
  "description": "Focus on preset 8 with CRF 23-28 for streaming",
  "parameters": {
    "preset": [8],
    "crf": [23, 24, 25, 26, 27, 28]
  }
}
```

Then repeat the workflow!
