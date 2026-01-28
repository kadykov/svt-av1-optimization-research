# VMAF Analysis Notes

## Why VMAF NEG Mode for Codec Evaluation

### Background

VMAF (Video Multi-method Assessment Fusion) is Netflix's perceptual quality metric that predicts subjective video quality using machine learning trained on human perception data.

### Standard VMAF vs NEG Mode

**Standard VMAF:**
- Includes an "enhancement gain" term
- Rewards certain post-processing enhancements (sharpening, detail enhancement)
- Useful for evaluating complete streaming pipelines with post-processing
- Can be "gamed" by artificially boosting certain visual characteristics

**NEG Mode (No Enhancement Gain):**
- Disables the enhancement gain component
- Measures only the codec's compression quality
- Prevents encoders from artificially inflating scores with sharpening
- Industry standard for codec comparison (used in AV1, VP9, HEVC testing)

### Why NEG for This Project

Our goal is to evaluate **codec parameters**, not post-processing pipelines:

1. **Pure codec evaluation** - We want to measure SVT-AV1's compression efficiency
2. **Parameter comparison** - Fair comparison between preset/CRF combinations
3. **Industry alignment** - Matches how AV1 codecs are officially evaluated
4. **Prevents gaming** - Ensures we're not rewarding artificial enhancements

### Implementation

We use FFmpeg's libvmaf filter with the NEG model:

```bash
ffmpeg -i encoded.mkv -i original.mkv \
  -lavfi "[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1neg:log_path=output.json:n_threads=4" \
  -f null -
```

Model specification: `version=vmaf_v0.6.1neg`
- `vmaf_v0.6.1` is the model version
- `neg` suffix enables No Enhancement Gain mode

### Metrics Collected

From VMAF analysis, we extract:
- **Mean**: Average quality across all frames
- **Harmonic mean**: Better indicator of worst-case quality (emphasizes low scores)
- **Percentiles**: Distribution of quality (1st, 5th, 25th, 50th, 75th, 95th)
- **Min/Max**: Quality range
- **Standard deviation**: Consistency of quality

### Interpretation

**VMAF Score Range (0-100):**
- **95-100**: Transparent (indistinguishable from source)
- **90-95**: Excellent quality
- **80-90**: Good quality (typical for streaming)
- **70-80**: Acceptable quality (mobile streaming)
- **Below 70**: Noticeable artifacts

For codec evaluation:
- **Harmonic mean** is more important than arithmetic mean (captures worst frames)
- **5th percentile** shows quality floor (worst 5% of frames)
- **Consistency** (low std dev) is desirable

### Additional Metrics

We also calculate:
- **PSNR**: Traditional pixel-wise difference (dB scale, higher is better)
  - Good for comparing similar content
  - Less correlated with human perception than VMAF
  - Useful for sanity checks and traditional codec benchmarks

- **SSIM**: Structural similarity (0-1 scale, 1 is perfect)
  - Better than PSNR for perceptual quality
  - Still not as good as VMAF for predicting human perception
  - Computationally cheaper than VMAF

### References

- VMAF GitHub: https://github.com/Netflix/vmaf
- AV1 codec evaluation methodology uses VMAF NEG
- Alliance for Open Media testing framework uses VMAF NEG
- FFmpeg libvmaf documentation: https://ffmpeg.org/ffmpeg-filters.html#libvmaf
