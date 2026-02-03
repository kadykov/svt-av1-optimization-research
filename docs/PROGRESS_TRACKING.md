# Progress Tracking

Both `encode_study.py` and `analyze_study.py` now include intelligent progress tracking that provides real-time feedback during long-running operations.

## Features

### Weighted Progress Calculation

Instead of simply counting files (which doesn't account for varying encoding/analysis times), the scripts calculate **work units** based on video complexity:

```
Work Units = Frame Count × Width × Height
```

This metric represents the total amount of pixel data that needs to be processed. Videos with:
- Higher resolution (1080p vs 720p)
- Longer duration (more frames)
- Higher frame rate (60fps vs 30fps)

...will have proportionally more work units, and completing them counts for more progress.

### Progress Display

During encoding or analysis, you'll see progress information like this:

```
[Clip 2/5] bbb_1080p60_clip_005.mp4
  [3/9] [34.2%, ETA: 12m 45s] preset=8, crf=30... ✓ (42.3s, 8.45 MB)
```

The display shows:
- **Current clip**: Which clip is being processed (e.g., "Clip 2/5")
- **Parameter combination**: Which encoding parameters are being used (e.g., "3/9")
- **Progress percentage**: Overall completion based on work units (e.g., "34.2%")
- **ETA**: Estimated time remaining based on average processing speed (e.g., "12m 45s")
- **Status**: Success (✓) or failure (✗) with encoding time and output size

### Accurate Time Estimation

The ETA is calculated dynamically:

1. **Initial phase**: When starting, ETA is not shown (only "0.0%")
2. **Learning phase**: After processing some work units, the script estimates average time per work unit
3. **Continuous updates**: As more clips are processed, the ETA becomes increasingly accurate
4. **Adapts to variation**: If some clips encode faster/slower, the ETA adjusts accordingly

### Workload Calculation Phase

Before starting, the scripts scan all clips to calculate total work:

```
Calculating encoding workload...
  bbb_1080p60_clip_005.mp4: 995,328,000 work units (300 frames, 1920×1728)
  sintel_clip_001.mp4: 829,440,000 work units (300 frames, 1920×1440)
  ...
Total workload: 8,954,880,000 work units across 45 encodings
```

This helps you understand:
- Which clips are most complex (will take longest)
- Total amount of work ahead
- Why progress might seem faster/slower for certain clips

## Time Format

Estimated time remaining is formatted for easy reading:
- Under 1 minute: `45s`
- 1-60 minutes: `12m 30s`
- Over 1 hour: `2h 15m`

## Benefits

### For Users
- **Visibility**: Know how much work is left instead of wondering when it will finish
- **Planning**: Better estimate whether to wait or come back later
- **Confidence**: See steady progress even when individual clips take different amounts of time

### For CI/CD
- **Timeouts**: More accurate timeout settings based on expected workload
- **Resource allocation**: Better understanding of job duration
- **Monitoring**: Easier to detect stuck or slow jobs

## Implementation Details

### Video Information Extraction

The scripts use `ffprobe` to extract:
- Frame count
- Resolution (width × height)
- Frame rate
- Duration

This happens once at the start in a "Calculating workload..." phase.

### Fallback Behavior

If video information cannot be extracted (e.g., corrupted file, unsupported format):
- The script assigns a default work unit value (1,000,000)
- Progress tracking continues but may be less accurate for that clip
- The encoding/analysis still proceeds normally

### Progress Updates

Progress is updated after each encoding/analysis completes:
1. Add completed work units to running total
2. Calculate new progress percentage
3. Update ETA based on average time per work unit
4. Display on next line

## Example Output

### Encoding Study

```bash
$ python scripts/encode_study.py config/studies/baseline_sweep.json

Study: baseline_sweep
Description: Comprehensive sweep of preset and CRF values
Clips directory: data/test_clips
Output directory: data/encoded/baseline_sweep
Found 5 clips
Parameter combinations: 9
Total encodings: 45

Calculating encoding workload...
  bbb_1080p60_clip_005.mp4: 995,328,000 work units (300 frames, 1920×1728)
  elephants_dream_clip_004.mp4: 829,440,000 work units (300 frames, 1920×1440)
  sintel_clip_001.mp4: 829,440,000 work units (300 frames, 1920×1440)
  tears_of_steel_clip_002.mp4: 497,664,000 work units (300 frames, 1920×864)
  cosmos_laundromat_clip_003.mp4: 829,440,000 work units (300 frames, 1920×1440)
Total workload: 35,819,520,000 work units across 45 encodings

Starting encoding...

[Clip 1/5] bbb_1080p60_clip_005.mp4
  [1/9] [0.0%] preset=6, crf=20... ✓ (125.4s, 12.34 MB)
  [2/9] [2.8%, ETA: 1h 15m] preset=6, crf=30... ✓ (118.2s, 6.78 MB)
  [3/9] [5.6%, ETA: 1h 12m] preset=6, crf=40... ✓ (115.8s, 3.45 MB)
  [4/9] [8.3%, ETA: 1h 10m] preset=8, crf=20... ✓ (68.9s, 12.12 MB)
  ...
```

### Analysis Study

```bash
$ python scripts/analyze_study.py baseline_sweep

Analyzing study: baseline_sweep
Metrics: vmaf, psnr, ssim
Source clips: data/test_clips
VMAF threads: 16

Calculating analysis workload...
  bbb_1080p60_clip_005.mp4: 995,328,000 work units (300 frames, 1920×1728)
  elephants_dream_clip_004.mp4: 829,440,000 work units (300 frames, 1920×1440)
  ...
Total workload: 35,819,520,000 work units across 45 encodings

Encodings to analyze: 45
Unique source clips: 5

[1/45] [0.0%] Analyzing: bbb_1080p60_clip_005_p6_crf20.mkv
  Source: bbb_1080p60_clip_005.mp4
  Calculating VMAF (NEG mode)...
    Mean: 98.45, Harmonic: 98.12, Min: 96.34
  Calculating PSNR...
    PSNR Y: 42.34 dB, Avg: 41.89 dB
  Calculating SSIM...
    SSIM Y: 0.9812, Avg: 0.9789
  Analysis time: 45.2s

[2/45] [2.8%, ETA: 32m 15s] Analyzing: bbb_1080p60_clip_005_p6_crf30.mkv
  ...
```

## Tips

### Verbose Mode

Use `-v` or `--verbose` to see detailed workload information:

```bash
python scripts/encode_study.py config/studies/baseline_sweep.json -v
```

This shows work units for each clip during the calculation phase.

### Interpreting Progress

- Progress may not be linear - some clips/presets are faster than others
- ETA stabilizes after processing ~10-20% of total work units
- Very fast/slow clips early on can skew initial ETA estimates
- Progress percentage is more reliable than file count for mixed-content studies

### Performance Considerations

The workload calculation phase adds a few seconds at startup but:
- Only runs once at the beginning
- Provides value throughout the entire run
- Negligible overhead compared to encoding/analysis time

## Future Improvements

Potential enhancements being considered:
- Per-preset encoding speed estimates (preset 13 is much faster than preset 6)
- Historical data to improve initial ETA accuracy
- Progress bars instead of text percentage
- Separate ETAs for each metric during analysis
- Save/resume capability with progress state
