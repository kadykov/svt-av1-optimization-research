# Quality Measurement System

## Overview

The measurement system calculates quality metrics (VMAF, PSNR, SSIM) for all encodings in a study by comparing them against the original source clips.

## Quick Start

```bash
# Measure a completed study
just measure-study baseline_sweep

# Faster: only VMAF
just measure-vmaf baseline_sweep

# Use more threads
just measure-study baseline_sweep --threads 8

# Continue despite errors
just measure-study baseline_sweep --continue-on-error
```

## Metrics Calculated

### VMAF (NEG Mode) - Primary Metric
- **What**: Netflix's perceptual quality metric
- **NEG Mode**: No Enhancement Gain (ideal for codec evaluation)
- **Scale**: 0-100 (higher is better)
- **Statistics**: mean, harmonic mean, percentiles, min/max, std dev
- **Why**: Best predictor of human perception, industry standard for codec testing

### PSNR - Traditional Metric
- **What**: Peak Signal-to-Noise Ratio
- **Scale**: dB (higher is better, typically 30-50 dB)
- **Statistics**: per-channel means (Y, U, V) + average
- **Why**: Traditional benchmark, good for sanity checks

### SSIM - Structural Metric
- **What**: Structural Similarity Index
- **Scale**: 0-1 (1 is perfect)
- **Statistics**: per-channel means (Y, U, V) + average
- **Why**: Better than PSNR, faster than VMAF

## Output Structure

Results saved to: `data/encoded/{study_name}/measurements.json`

```json
{
  "study_name": "baseline_sweep",
  "encoding_metadata_file": "encoding_metadata.json",
  "start_time": "2026-01-28T...",
  "measurements": {
    "clip1_p6_crf28.mkv": {
      "vmaf": {
        "mean": 95.2,
        "harmonic_mean": 94.8,
        "percentile_5": 91.2,
        "min": 89.3,
        "max": 98.1
      },
      "psnr": {"y": 42.3, "u": 45.6, "v": 46.1},
      "ssim": {"y": 0.982, "u": 0.991, "v": 0.993, "all": 0.988},
      "video_info": {
        "encoded_duration": 10.0,
        "source_duration": 10.0,
        "duration_match": true,
        "frame_count": 300
      },
      "measurement_time_seconds": 15.2,
      "success": true
    }
  },
  "summary": {
    "total_measurements": 280,
    "successful_measurements": 278,
    "failed_measurements": 2,
    "vmaf_range": {"min": 78.5, "max": 98.2, "mean": 91.5}
  }
}
```

## Efficiency Metrics

Efficiency metrics are calculated in the analysis phase (via `analyze_study.py`), not stored in measurements.json. This includes:

1. **VMAF per kbps**: Quality per unit of bitrate
   - Higher is better (more quality per bandwidth)
   - Key metric for finding optimal parameters
   - Uses video-only bitrate (extracted via FFprobe)

2. **VMAF per megabyte**: Quality per unit of file size
   - Higher is better (more quality per storage)
   - Useful for archive scenarios

3. **Quality per encoding second**: Quality divided by encoding time
   - Higher is better (more quality per compute time)
   - Useful for speed/quality tradeoffs

### Bitrate Calculation

The analysis script calculates **video-only bitrate** using FFprobe:
- Extracts actual video stream bitrate (kbps)
- Excludes audio streams and container overhead
- More accurate than file size / duration calculations
- Required for proper bpp (bitrate per pixel) metrics in visualization

**Note**: Encodings are done without audio (`-an` flag) to:
- Focus on video codec performance
- Ensure accurate bitrate measurements
- Reduce file sizes and encoding time
- Eliminate audio encoding as a confounding variable

## Usage Examples

### Basic Measurement
```bash
just measure-study baseline_sweep
```

### Only VMAF (Faster)
```bash
just measure-vmaf baseline_sweep
```

### Specific Metrics
```bash
# VMAF and PSNR only
. venv/bin/activate
python scripts/measure_study.py baseline_sweep --metrics vmaf psnr
```

### Performance Tuning
```bash
# Use more threads for VMAF (default: 4)
just measure-study baseline_sweep --threads 8

# Verbose output to debug
just measure-study baseline_sweep -v
```

### Resilient Measurement
```bash
# Continue even if some encodings fail to measure
just measure-study baseline_sweep --continue-on-error
```

## Workflow

Complete workflow from raw videos to measurement:

```bash
# 1. Download videos
just fetch-videos

# 2. Extract test clips
just extract-clips 10 15 30

# 3. Run encoding study
just encode-study baseline_sweep

# 4. Measure quality
just measure-study baseline_sweep

# 5. Review results
cat data/encoded/baseline_sweep/measurements.json
```

## Performance Notes

### VMAF Calculation Time
- ~1-10x realtime depending on:
  - Video resolution
  - Number of threads
  - CPU performance
- Example: 20-second 1080p clip ≈ 30-60 seconds to analyze with 4 threads

### Optimization Tips
1. **Use more threads**: `--threads 8` (but diminishing returns beyond CPU cores)
2. **VMAF only**: Skip PSNR/SSIM if not needed
3. **Parallel studies**: Measure multiple studies simultaneously if you have CPU cores
4. **SSD storage**: Faster I/O helps with large encoded files

## Next Steps

After measurement, you can:
1. **Analyze and visualize**: `just analyze-study baseline_sweep` - generates plots, CSV, and reports
2. **Compare studies**: Analyze multiple parameter sweeps
3. **Optimize parameters**: Find sweet spots for your use case

## Troubleshooting

### "Source clip not found"
- Ensure clips are in `data/test_clips/`
- Check `clip_metadata.json` for clip names
- Verify clips haven't been cleaned

### "Encoded file not found"
- Run encoding first: `just encode-study baseline_sweep`
- Check encoding succeeded in `encoding_metadata.json`

### VMAF calculation fails
- Verify FFmpeg has libvmaf: `ffmpeg -filters | grep vmaf`
- Check video compatibility (both encoded and source)
- Try verbose mode: `-v`

### Slow measurement
- Increase threads: `--threads 8`
- Use VMAF only: `just measure-vmaf baseline_sweep`
- Check CPU usage (should be near 100% during VMAF)
