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
  -lavfi "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]libvmaf=model=version=vmaf_v0.6.1neg:log_path=output.json:n_threads=4" \
  -f null -
```

**Critical: Timestamp Alignment with `setpts=PTS-STARTPTS`**

When clips are extracted using `-c copy` (stream copy), they often retain the
original video's timestamps. This means a clip extracted from 2 minutes into
a source video might start at PTS=120.0 instead of PTS=0.0. When the encoded
version has different start timestamps, libvmaf compares wrong frames,
resulting in:
- Artificially low VMAF scores (often 10-60 instead of 90+)
- VMAF scores that don't correlate with quality settings (CRF)
- Frame scores of 0.0 scattered throughout

The `setpts=PTS-STARTPTS` filter resets timestamps to start from 0, ensuring
proper frame-to-frame alignment between source and encoded videos.

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

## Troubleshooting VMAF Issues

### Low VMAF Scores That Don't Correlate with Quality Settings

**Symptoms:**
- VMAF scores of 10-60 regardless of CRF/quality settings
- CRF 20 and CRF 40 produce nearly identical VMAF scores
- Many frames have VMAF=0.0
- High variance in frame scores (std_dev > 20)
- Alternating pattern of high and low scores (e.g., 92, 12, 17, 91, 20...)

**Common Causes and Solutions:**

1. **B-Frame Ordering Issues from Stream Copy** (Most Common - CRITICAL)
   - When clips are extracted with `-c copy` (stream copy), B-frame ordering
     in the container may differ from presentation order
   - This causes libvmaf to compare frame N of encoded with frame N±1 of reference
   - Results in alternating high/low VMAF scores and artificially low averages
   - **Diagnosis:** Check per-frame VMAF scores - alternating pattern indicates this issue
   - **Fix:** Re-extract source clips using lossless encoding (FFV1):
     ```bash
     # Instead of stream copy:
     ffmpeg -ss START -i source.mp4 -t DURATION -c copy output.mp4  # BAD

     # Use lossless re-encoding:
     ffmpeg -ss START -i source.mp4 -t DURATION -c:v ffv1 -level 3 -an output.mkv  # GOOD
     ```
   - The `extract_clips.py` script now uses FFV1 by default. Use `--fast` flag
     for stream copy mode (not recommended for VMAF analysis).

2. **Timestamp Misalignment**
   - Clips extracted with `-c copy` retain original timestamps
   - Source starts at different PTS than encoded file
   - **Fix:** Use `setpts=PTS-STARTPTS` in the filter graph (already implemented)

3. **Frame Count Mismatch**
   - Different number of frames in source vs encoded
   - Can happen with variable frame rate sources
   - **Check:** `ffprobe -count_frames -select_streams v:0 ...`

4. **Resolution Mismatch**
   - Source and encoded have different resolutions
   - **Check:** `ffprobe -show_entries stream=width,height ...`

### Diagnosing B-Frame Ordering Issues

To check if you have B-frame ordering problems:

1. **Check source has B-frames:**
   ```bash
   ffprobe -v error -select_streams v:0 -show_entries stream=has_b_frames -of csv source.mp4
   # If output is "stream,1", source has B-frames
   ```

2. **Check per-frame VMAF scores:**
   ```bash
   ffmpeg -i encoded.mkv -i source.mp4 \
     -lavfi "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]libvmaf=log_path=/tmp/vmaf.json:log_fmt=json" \
     -f null -

   # Check for alternating pattern
   cat /tmp/vmaf.json | jq '[.frames[] | .metrics.vmaf] | .[:20]'
   ```

   If you see alternating high/low scores like `[92, 12, 91, 15, 93, 18, ...]`,
   you have B-frame ordering issues.

3. **Verify fix by comparing against lossless reference:**
   ```bash
   # Re-encode source to lossless FFV1
   ffmpeg -i source.mp4 -c:v ffv1 -level 3 -an /tmp/source_lossless.mkv

   # Compare encoded against lossless reference
   ffmpeg -i encoded.mkv -i /tmp/source_lossless.mkv \
     -lavfi "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]libvmaf" \
     -f null -
   ```

   If this gives much higher VMAF than the original comparison, the issue was
   B-frame ordering.

### Validating VMAF Setup

Run this sanity check to compare a source to itself (should be ~100):

```bash
ffmpeg -i source.mp4 -i source.mp4 \
  -lavfi "[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1neg" \
  -f null -
```

If this returns anything below 99.9, there's a problem with the setup.
