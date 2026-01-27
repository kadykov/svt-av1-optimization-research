# SVT-AV1 Optimization Research

Research project to find optimal SVT-AV1 encoding parameters balancing computational power, file size, and quality.

## Project Structure

```
.
├── config/
│   └── video_sources.json    # Human-edited: URLs, categories, licenses
├── data/
│   ├── raw_videos/           # Downloaded videos + download_metadata.json
│   ├── test_clips/           # Short clips cut from raw videos
│   └── encoded/              # Encoded test results
├── results/                   # Analysis results, plots, CSVs
├── scripts/
│   └── fetch_videos.py       # Video downloader with zip extraction
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

```bash
just install              # Setup environment
just list-videos          # List all videos with status
just fetch-videos         # Download all
just fetch-one <id>       # Download specific video
just fetch-category <cat> # Download by category
just clean-videos         # Remove all video files
```

## Features

- ✅ Resume interrupted downloads
- ✅ Automatic zip extraction
- ✅ Fallback (continues on error)
- ✅ SHA256 checksums
- ✅ Metadata tracking in `download_metadata.json`
- ✅ Smart skip (won't re-download)

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

## Development Workflow

1. ✅ Small dataset (current)
2. Create clip extraction script
3. Develop encoding pipeline
4. Add quality metrics (VMAF, SSIM, PSNR)
5. Analysis and visualization
6. Expand dataset
7. Full parameter sweep

## Next Steps

- [ ] Extract short clips from videos
- [ ] Implement encoding with parameter sweep
- [ ] Calculate quality metrics
- [ ] Create analysis visualizations
- [ ] Expand video collection