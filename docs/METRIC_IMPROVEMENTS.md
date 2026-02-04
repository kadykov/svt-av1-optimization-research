# Metric Improvements: Normalized Per-Frame-Per-Pixel Analysis

## Overview

This document explains the improvements made to metrics in the visualization and analysis system. All metrics are now normalized per-frame-per-pixel for fair comparison across different video characteristics.

## Motivation

### The Problem with Raw Metrics

Previously, we used raw metrics like:
- **Encoding time** (seconds): A 4K 60fps video naturally takes longer than a 720p 30fps video
- **Bitrate per pixel** (bpp): Doesn't account for frame rate differences
- **File size**: Varies with video duration and resolution

These metrics make it difficult to compare:
- Videos with different resolutions (1080p vs 4K)
- Videos with different frame rates (24fps vs 60fps)
- Videos with different durations (5s vs 30s)

### The Solution: Normalization

By normalizing metrics per-frame-per-pixel, we can fairly compare encoding efficiency across all video characteristics.

## New Metrics

### 1. File Size Efficiency

**Old metric:** `bpp` (bits per pixel per second)
- Formula: `bitrate / (width × height × fps)`
- Problem: Name was confusing; actually measures bits per pixel per frame

**New metric:** `bytes_per_frame_per_pixel`
- Formula: `file_size_bytes / (num_frames × width × height)`
- Clearer name and interpretation
- Allows direct comparison across videos with different characteristics
- Lower values = better compression

### 2. Encoding Time Efficiency

**Old metric:** `encoding_time_s` (raw seconds)
- Problem: Not comparable across different video complexities

**New metric:** `encoding_time_per_frame_per_pixel`
- Formula: `(encoding_time_s × 1000) / (total_pixels / 1_000_000)`
- Units: milliseconds per megapixel per frame
- Measures computational cost normalized by video complexity
- Allows fair comparison of encoding speed across different resolutions and durations
- Lower values = faster encoding

### 3. Quality Efficiency (Inverted Form)

**Old metrics:**
- `vmaf_per_bpp` = `vmaf / bpp` (higher is better)
- `p5_vmaf_per_bpp` = `vmaf_p5 / bpp` (higher is better)

**Problem:** These metrics answer "how much quality do I get per bit?", but it's more intuitive to ask "how many bits do I need for this quality?"

**New metrics (inverted):**
- `bytes_per_vmaf_per_frame_per_pixel` = `bytes_per_frame_per_pixel / vmaf_mean`
- `bytes_per_p5_vmaf_per_frame_per_pixel` = `bytes_per_frame_per_pixel / vmaf_p5`

**Interpretation:**
- "How many bytes per pixel per frame do I need to achieve this VMAF score?"
- Lower values = more efficient (fewer bytes needed for same quality)
- More intuitive: represents the cost per quality point
- Directly comparable across all video types

### 4. Combined Efficiency

**Old metrics:**
- `vmaf_per_bpp_per_time` = `vmaf / bpp / encoding_time`
- `p5_vmaf_per_bpp_per_time` = `vmaf_p5 / bpp / encoding_time`

**New metrics:**
- `bytes_per_vmaf_per_encoding_time` = `file_size_bytes / vmaf_mean / encoding_time_s`
- `bytes_per_p5_vmaf_per_encoding_time` = `file_size_bytes / vmaf_p5 / encoding_time_s`

**Key insight:** The per-frame-per-pixel terms cancel out mathematically!
```
(bytes/frame/pixel / vmaf) / (time/frame/pixel)
= bytes/frame/pixel / (vmaf × time/frame/pixel)
= bytes / (vmaf × time)
```

**Interpretation:**
- "What's the file size cost per quality point per second of encoding time?"
- Lower values = better overall efficiency
- Balances file size, quality, and encoding speed
- Directly comparable across all video types

## Benefits of Normalization

### Fair Comparisons

```python
# Example: Compare encoding speeds fairly
# Before (misleading):
video_1080p_5s: encoding_time = 10s
video_4k_30s:   encoding_time = 120s
# Conclusion: 4K is 12× slower (misleading!)

# After (normalized):
video_1080p_5s: encoding_time_per_frame_per_pixel = 45 ms/megapixel
video_4k_30s:   encoding_time_per_frame_per_pixel = 42 ms/megapixel
# Conclusion: Actually very similar computational cost!
```

### Generalization

Findings from short clips (5-15s) can now be confidently applied to longer videos, because metrics are normalized by video characteristics.

### Intuitive Interpretation

Inverted efficiency metrics answer practical questions:
- "How many bytes do I need per pixel to achieve VMAF 95?"
- "What's my file size cost per quality point?"

## Backward Compatibility

Legacy metrics are retained in CSV exports for backward compatibility:
- `bpp` (bits per pixel per frame)
- `vmaf_per_bpp` (VMAF per bpp)
- `p5_vmaf_per_bpp` (P5-VMAF per bpp)

However, **all visualizations and reports use the new normalized metrics**.

## Typical Value Ranges

Based on SVT-AV1 encoding:

### `bytes_per_frame_per_pixel`
- **< 0.003**: Highly compressed (check quality!)
- **0.003-0.006**: Good compression with decent quality
- **0.006-0.010**: High quality
- **> 0.010**: Near-lossless

### `bytes_per_vmaf_per_frame_per_pixel`
- **< 0.00003**: Very efficient encoding
- **0.00003-0.00005**: Good efficiency
- **> 0.00010**: Inefficient (too many bytes for quality achieved)

### `encoding_time_per_frame_per_pixel` (ms/megapixel)
- **< 20**: Fast presets (8-10)
- **20-60**: Balanced presets (5-7)
- **60-100**: Slow presets (3-4)
- **> 100**: Very slow, high-quality presets (2)

### `bytes_per_vmaf_per_encoding_time`
- **< 500**: Very efficient overall (good file size, quality, and speed)
- **500-2000**: Balanced efficiency
- **> 5000**: Inefficient (either large files, low quality, or slow encoding)

## Migration Guide

### For Visualization Users

No action needed! The `just visualize-study` command automatically uses new metrics.

### For Custom Analysis Scripts

Update your scripts to use new metric names:

**Before:**
```python
df['bpp']  # Bitrate per pixel
df['vmaf_per_bpp']  # Quality efficiency
df['encoding_time_s']  # Raw encoding time
```

**After:**
```python
df['bytes_per_frame_per_pixel']  # File size efficiency
df['bytes_per_vmaf_per_frame_per_pixel']  # Quality efficiency (inverted)
df['encoding_time_per_frame_per_pixel']  # Normalized encoding time
```

**Finding best configurations:**
```python
# Before (higher is better):
best = df.nlargest(5, 'vmaf_per_bpp')

# After (lower is better - inverted):
best = df.nsmallest(5, 'bytes_per_vmaf_per_frame_per_pixel')
```

### For Report Interpretation

Look for **lower** values in efficiency metrics (inverted form):
- Lower `bytes_per_vmaf_per_frame_per_pixel` = more efficient
- Lower `bytes_per_vmaf_per_encoding_time` = better overall efficiency

## Technical Details

### Calculation Formulas

```python
# Video properties
num_frames = duration * fps
total_pixels = num_frames * width * height

# Normalized metrics
bytes_per_frame_per_pixel = file_size_bytes / total_pixels

encoding_time_per_frame_per_pixel = (encoding_time_s * 1000) / (total_pixels / 1_000_000)

# Efficiency metrics (inverted)
bytes_per_vmaf_per_frame_per_pixel = bytes_per_frame_per_pixel / vmaf_mean

bytes_per_p5_vmaf_per_frame_per_pixel = bytes_per_frame_per_pixel / vmaf_p5

# Combined efficiency (per-frame-per-pixel cancels)
bytes_per_vmaf_per_encoding_time = file_size_bytes / vmaf_mean / encoding_time_s

bytes_per_p5_vmaf_per_encoding_time = file_size_bytes / vmaf_p5 / encoding_time_s
```

### Why Invert Efficiency Metrics?

The inverted form is more intuitive for several reasons:

1. **Practical question**: "How many bytes do I need per pixel to achieve VMAF 95?" vs "How much VMAF do I get per byte?"

2. **Cost-based thinking**: Treats file size as a "cost" and quality as a "goal"

3. **Lower is better**: Consistent with file size metrics (lower = better)

4. **Avoids infinity**: When VMAF approaches 100, the original form (`vmaf / bytes`) grows without bound, making comparisons difficult

### Why Not Transform VMAF?

We considered using `bytes / (100 - vmaf)` to remove VMAF's upper bound (100), but decided against it:

1. **VMAF is well-known**: Everyone understands VMAF 95 means near-transparent quality
2. **Perceptual meaning**: VMAF values correspond to human perception thresholds
3. **Simplicity**: Keeping VMAF as-is makes results easier to interpret

## References

- [VMAF Documentation](https://github.com/Netflix/vmaf)
- [SVT-AV1 Encoder Documentation](https://gitlab.com/AOMediaCodec/SVT-AV1)
- Video codec research typically normalizes by pixels/frame for fair comparison
