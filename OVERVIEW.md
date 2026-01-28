# Project Overview

## Goal
Find optimal SVT-AV1 encoding parameters that balance:
- Encoding computational power (speed)
- Output file size
- Visual quality

## Approach

### Phase 1: Data Collection ✅
- Download diverse video test files with open licenses
- Start with small dataset (4 videos)
- Expand later with more categories

### Phase 2: Clip Preparation ✅ (Current)
- Extract short test clips from full videos (10-30 seconds)
- Multiple clips per video to test different scenes
- Store clips in consistent format
- Random selection with optional seed for reproducibility
- Filter by category, resolution, FPS
- Duration range testing (codec efficiency analysis)

### Phase 3: Encoding Tests ✅
- Study-based approach for focused parameter sweeps
- Baseline study: preset (4-10) × CRF (20-40)
- Specialized studies:
  - Film grain synthesis levels
  - Screen content modes
  - Tuning modes (VQ/PSNR/SSIM)
  - Tile configurations (future)
- Track encoding time and system resources
- SHA256 checksums for reproducibility

### Phase 4: Quality Metrics ✅ (Current)
- Calculate objective metrics:
  - VMAF NEG (Video Multi-method Assessment Fusion, No Enhancement Gain mode)
    * Industry standard for codec evaluation
    * Perceptual quality metric from Netflix
    * Disables enhancement gain for pure codec testing
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)
- Comprehensive statistics:
  - Mean, harmonic mean (better for worst-case quality)
  - Percentiles (1st, 5th, 25th, 50th, 75th, 95th)
  - Min/max, standard deviation
- Efficiency metrics:
  - Quality per byte (VMAF per megabyte)
  - Quality per bitrate (VMAF per kbps)
  - Quality per encoding second (speed/quality tradeoff)
- FFmpeg libvmaf integration (no additional dependencies)

### Phase 5: Analysis
- Visualizations:
  - Quality vs file size curves
  - Encoding speed vs quality
  - Parameter impact analysis
  - Per-category comparisons
- Find optimal parameter sets for different use cases:
  - Archive (best quality, size/speed less important)
  - Streaming (balance)
  - Fast encoding (speed priority)

## Video Categories

### Current Starter Set
1. **3D Animation** - Blender movies (clean, synthetic)
2. **Mixed content** - VFX with live action

### Planned Additions
3. **2D Animation** - Classic cartoons, hand-drawn
4. **Handheld footage** - Camera shake, unstable
5. **Real film grain** - Classic films, organic grain
6. **Screencast** - Code, terminal, presentations
7. **Sports** - Fast motion, tracking
8. **Low light** - Noise, concerts, night scenes
9. **Talking heads** - Interviews, vlogs (common use case)
10. **Gaming** - Screen recordings with UI + 3D
11. **Timelapse** - Gradual changes

## Hypotheses to Test

1. **Film grain synthesis** may be more efficient than encoding real grain
2. Different content types have optimal presets
3. Sweet spot exists around preset 6-8 for most content
4. CRF 25-35 is likely optimal range
5. Faster presets may be good enough for some categories
6. Tiling helps with parallel encoding but may affect quality

## Reproducibility

- All video sources documented with URLs and licenses
- Configuration files for exact parameters
- Requirements frozen for Python environment
- Scripts for entire pipeline
- Results stored with metadata

## Tools

- **FFmpeg** with SVT-AV1 encoder
- **VMAF** for quality assessment
- **Python** for automation and analysis
- **Pandas** for data handling
- **Matplotlib/Seaborn** for visualization
- **Just** for command orchestration
