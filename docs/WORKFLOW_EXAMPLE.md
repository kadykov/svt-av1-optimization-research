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

### Visual Analysis (PNG files)

1. **`baseline_sweep_rate_distortion.png`** (709 KB)
   - Left plot: VMAF mean vs file size for each preset
   - Right plot: VMAF harmonic mean vs file size (worst-case quality)
   - Shows quality/compression efficiency tradeoffs
   - CRF values annotated on curves

2. **`baseline_sweep_speed_quality.png`** (216 KB)
   - Left plot: Encoding time vs VMAF quality
   - Right plot: Quality per encoding second by preset (bar chart)
   - Shows speed vs quality tradeoffs
   - Identifies fastest presets for target quality

3. **`baseline_sweep_parameter_impact.png`** (392 KB)
   - Four heatmaps showing preset × CRF combinations:
     - VMAF mean score
     - File size (MB)
     - Encoding time (seconds)
     - VMAF per MB (efficiency)
   - Full parameter space visualization at a glance

4. **`baseline_sweep_clip_comparison.png`** (157 KB)
   - Left plot: VMAF by clip and preset
   - Right plot: File size by clip and preset
   - Shows content-dependent encoding behavior
   - Identifies "hard to encode" clips

5. **`baseline_sweep_vmaf_distribution.png`** (258 KB)
   - Left plot: Box plots of VMAF percentiles (P5-P95)
   - Right plot: Mean vs harmonic mean scatter
   - Shows quality consistency and frame-level variation
   - Diagonal line indicates perfect consistency

### Data Export (CSV)

**`baseline_sweep_analysis.csv`**

Contains all metrics in tabular format:
- Identifiers: output_file, source_clip
- Parameters: preset, crf
- File metrics: file_size_mb, bitrate_kbps
- Performance: encoding_time_s, encoding_fps, analysis_time_s
- VMAF: mean, harmonic_mean, min, p1, p5, p25, median, p75, p95, std
- Other metrics: psnr_avg, ssim_avg
- Efficiency: vmaf_per_mb, quality_per_encode_s

**Example uses:**
```python
import pandas as pd

df = pd.read_csv('results/baseline_sweep/baseline_sweep_analysis.csv')

# Find best efficiency configurations (VMAF > 90, small file)
efficient = df[(df['vmaf_mean'] > 90) & (df['file_size_mb'] < 5)]
print(efficient[['preset', 'crf', 'vmaf_mean', 'file_size_mb']])

# Compare presets at same CRF
p6_crf30 = df[(df['preset'] == 6) & (df['crf'] == 30)]
p10_crf30 = df[(df['preset'] == 10) & (df['crf'] == 30)]
print(f"Preset 6: {p6_crf30['encoding_time_s'].mean():.2f}s")
print(f"Preset 10: {p10_crf30['encoding_time_s'].mean():.2f}s")
```

### Text Summary Report

**`baseline_sweep_report.txt`**

Human-readable summary including:
- Study metadata (date, VMAF model, clips analyzed)
- Overall statistics (quality range, file size range, time range)
- Best configurations:
  - Highest quality
  - Best efficiency (VMAF per MB)
  - Smallest file
- Parameter impact tables (by preset and CRF)

**Example output:**
```
Analysis Report: baseline_sweep
================================================================================

Analysis Date: 2026-01-30T23:08:52.000339Z
VMAF Model: vmaf_v0.6.1neg
Clips Analyzed: 2
Total Encodings: 18
Metrics: SSIM, PSNR, VMAF

Overall Statistics
--------------------------------------------------------------------------------
VMAF Mean Range: 1.91 - 92.44
File Size Range: 2.44 - 11.87 MB
Encoding Time Range: 3.26 - 17.76 seconds

Best Configurations
--------------------------------------------------------------------------------
Highest Quality:
  Preset: 6, CRF: 20
  VMAF: 92.44
  File Size: 7.18 MB
  Encoding Time: 16.28s

Best Efficiency (VMAF per MB):
  Preset: 10, CRF: 40
  VMAF: 88.20
  VMAF per MB: 36.18
  File Size: 2.44 MB
```

## Interpreting Results

### 1. Start with the Text Report

```bash
cat results/baseline_sweep/baseline_sweep_report.txt
```

This gives you:
- Overall quality and size ranges
- Best configurations for different goals
- Parameter impact statistics

### 2. Review Rate-Distortion Curves

Open `baseline_sweep_rate_distortion.png`:
- Look for the "sweet spot" where quality plateaus
- Compare preset efficiency (quality per MB)
- Identify diminishing returns points

**Key questions:**
- At what CRF does quality drop noticeably?
- Which preset offers best efficiency at target quality?
- Is the quality drop worth the file size reduction?

### 3. Check Speed vs Quality

Open `baseline_sweep_speed_quality.png`:
- See encoding time differences between presets
- Evaluate if quality gain justifies slower encoding
- Check "quality per encoding second" efficiency

**Key questions:**
- For target quality, how much time do different presets take?
- Is faster preset worth slight quality loss?
- What's the time investment for marginal quality gains?

### 4. Examine Parameter Heatmaps

Open `baseline_sweep_parameter_impact.png`:
- See full parameter space at once
- Identify "safe" parameter ranges
- Find optimal combinations for specific goals

**Key questions:**
- Which preset/CRF combinations cluster together in quality?
- Where are the efficiency hotspots (high VMAF per MB)?
- Are there parameter combinations to avoid?

### 5. Review Clip-Specific Behavior

Open `baseline_sweep_clip_comparison.png`:
- Identify content-dependent patterns
- See which clips are "harder" to encode
- Verify consistency across content types

**Key questions:**
- Do all clips respond similarly to parameter changes?
- Are some clips consistently harder to encode?
- Do results generalize across content?

### 6. Assess Quality Consistency

Open `baseline_sweep_vmaf_distribution.png`:
- Check frame-level quality variation (box plots)
- Compare mean vs harmonic mean (consistency indicator)
- Identify encodings with quality drops

**Key questions:**
- Does quality stay consistent throughout the video?
- Are there problematic frames (low P5)?
- Is average quality representative or misleading?

## Customization Examples

### Generate Only Specific Plots

```bash
# Only rate-distortion and parameter impact
just visualize-plots baseline_sweep rate-distortion parameter-impact

# Only speed comparison
just visualize-plots baseline_sweep speed-quality
```

### Custom Output Directory

```bash
# Save to custom location
just visualize-to baseline_sweep analysis_results/baseline/

# Different study, different location
just visualize-to film_grain analysis_results/grain_study/
```

### Python Analysis Script

Create `custom_analysis.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('results/baseline_sweep/baseline_sweep_analysis.csv')

# Custom analysis: Find best preset for 4K streaming
# Target: VMAF > 93, minimize file size
streaming_configs = df[df['vmaf_mean'] > 93].sort_values('file_size_mb')

print("Best configs for 4K streaming (VMAF > 93):")
print(streaming_configs[['preset', 'crf', 'vmaf_mean', 'file_size_mb', 'encoding_time_s']].head(5))

# Custom plot: CRF vs compression ratio
plt.figure(figsize=(10, 6))
for preset in sorted(df['preset'].unique()):
    subset = df[df['preset'] == preset]
    plt.plot(subset['crf'], subset['file_size_mb'], marker='o', label=f'Preset {preset}')

plt.xlabel('CRF')
plt.ylabel('File Size (MB)')
plt.title('CRF vs File Size by Preset')
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
print(all_data.groupby('study')['vmaf_per_mb'].max())
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
