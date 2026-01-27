# SVT-AV1 Optimization Research

Research project to find optimal SVT-AV1 encoding parameters balancing computational power, file size, and quality.

## Project Structure

```
.
├── config/
│   └── video_sources.json    # Human-edited: URLs, categories, licenses
├── data/
│   ├── raw_videos/           # Downloaded videos + download_metadata.json
│   ├── test_clips/           # Short clips cut from raw videos + clip_metadata.json
│   └── encoded/              # Encoded test results
├── results/                   # Analysis results, plots, CSVs
├── scripts/
│   ├── fetch_videos.py       # Video downloader with zip extraction
│   └── extract_clips.py      # Random clip extraction with filtering
├── justfile                   # Command shortcuts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

```bash
just install
```

## Quick Start

```bash
# See available videos
just list-videos

# Download all videos (starter set: 4 videos)
just fetch-videos

# Download one video
just fetch-one elephants_dream

# Download by category
just fetch-category 3d_animation
```

## Video Dataset

**Starter set (4 videos from Blender Foundation):**
- **Big Buck Bunny** (4K 60fps) - Colorful 3D animation, high motion
- **Sintel** (1080p) - Dark scenes, synthetic film grain
- **Tears of Steel** (teaser) - Live action + VFX
- **Elephants Dream** (teaser) - Surreal, high complexity

All CC-BY-3.0 or CC-BY-2.5 licensed.

## Commands

### Video Download
```bash
just install              # Setup environment
just list-videos          # List all videos with status
just fetch-videos         # Download all
just fetch-one <id>       # Download specific video
just fetch-category <cat> # Download by category
```

### Clip Extraction
```bash
# Extract 10 clips, 15-30 seconds each (auto-cleans directory first)
just extract-clips 10 15 30

# Extract from specific category
just extract-category 3d_animation 5 20 30

# Reproducible extraction with seed
just extract-seeded 10 15 30 42

# With additional filters (Full HD max, 60fps+)
just extract-clips 8 20 25 --max-height 1080 --min-fps 60

# Keep existing clips (no auto-clean)
just extract-clips 10 15 30 --no-clean
```

### Cleanup
```bash
just clean-clips          # Remove extracted clips only
just clean-videos         # Remove all video files (raw + clips + encoded)
```

## Features

### Video Download
- ✅ Resume interrupted downloads
- ✅ Automatic zip extraction
- ✅ Fallback (continues on error)
- ✅ SHA256 checksums
- ✅ Metadata tracking in `download_metadata.json`
- ✅ Smart skip (won't re-download)

### Clip Extraction
- ✅ Random fragment selection with optional seed
- ✅ Filter by category, resolution, FPS
- ✅ Duration range specification (test codec efficiency)
- ✅ Proportional extraction (longer videos → more clips)
- ✅ Auto-cleanup before extraction (ensures metadata matches clips)
- ✅ FFprobe for metadata, FFmpeg for extraction
- ✅ Metadata tracking in `clip_metadata.json`

## Metadata Design

**`config/video_sources.json`** - Human-edited, minimal:
- ID, name, URL
- Categories, license
- No redundant technical metadata

**`data/raw_videos/download_metadata.json`** - Machine-generated:
- Actual file size (bytes)
- SHA256 checksum
- Downloaded file path
- Categories, license (for reference)

This file should be committed to track verified checksums.

**`data/test_clips/clip_metadata.json`** - Machine-generated:
- Extraction parameters (filters, duration range, seed)
- Per-clip metadata: source video, timestamps, resolution, FPS
- SHA256 checksums for clips

This file is NOT committed (in .gitignore) because:
- It reflects only the current clip set
- Each extraction is reproducible via seed
- Cleaning before extraction ensures metadata always matches actual clips

## Development Workflow

1. ✅ Small dataset
2. ✅ Clip extraction script (current)
3. Develop encoding pipeline
4. Add quality metrics (VMAF, SSIM, PSNR)
5. Analysis and visualization
6. Expand dataset
7. Full parameter sweep

## Next Steps

- [x] ✅ Extract short clips from videos
- [ ] Implement encoding with parameter sweep
- [ ] Calculate quality metrics
- [ ] Create analysis visualizations
- [ ] Expand video collection