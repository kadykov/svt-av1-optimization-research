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

### Encoding Studies
```bash
# List available studies
just list-studies

# Dry run to see what would be encoded
just dry-run-study baseline_sweep

# Run encoding study
just encode-study baseline_sweep

# Continue despite encoding errors
just encode-study film_grain --continue-on-error

# Verbose output with FFmpeg commands
just encode-study baseline_sweep -v
```

### Quality Measurement
```bash
# List encoded studies ready for measurement
just list-encoded

# Measure study (VMAF + PSNR + SSIM)
just measure-study baseline_sweep

# Measure with only VMAF (faster)
just measure-vmaf baseline_sweep

# Continue despite measurement errors
just measure-study baseline_sweep --continue-on-error

# Use more threads for faster VMAF calculation
just measure-study baseline_sweep --threads 8

# Verbose output with FFmpeg commands
just measure-study baseline_sweep -v
```

### Analysis & Visualization
```bash
# Generate all plots and CSV for a study
just analyze-study baseline_sweep

# Generate specific metrics only
python scripts/analyze_study.py baseline_sweep --metrics vmaf_combined vmaf_per_bpp

# Skip optional plots for faster processing
python scripts/analyze_study.py baseline_sweep --no-clip-plots --no-duration-analysis

# Skip CSV or report generation
python scripts/analyze_study.py baseline_sweep --no-csv --no-report

# Clean analysis results (plots, CSVs)
just clean-results
```

**Plot organization:**
- **Metric trios**: For each metric, three views (heatmap, vs CRF, vs preset)
- **Per-clip comparison**: Content-dependent behavior analysis
- **Duration analysis**: Efficiency vs clip characteristics
- **CSV exports**: Raw and aggregated data for custom analysis
- **Text report**: Human-readable summary with best configurations

**Available metrics:**
- `vmaf_combined` - VMAF Mean and P5 (combined plot)
- `bpp` - Bitrate per pixel (compression rate)
- `vmaf_per_bpp` - Quality efficiency
- `p5_vmaf_per_bpp` - Worst-case quality efficiency
- `encoding_time_s` - Encoding time
- `vmaf_per_time` - Quality per encoding second
- `vmaf_per_bpp_per_time` - Combined efficiency metric
- `p5_vmaf_per_bpp_per_time` - P5-VMAF combined efficiency

See [VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md) for details.

### Cleanup
```bash
just clean-clips          # Remove extracted clips only
just clean-encoded        # Remove encoded videos (keeps raw + clips)
just clean-results        # Remove analysis results (plots, CSVs)
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

### Encoding Studies
- ✅ Study-based configuration system (focused parameter sweeps)
- ✅ Automatic parameter combination generation
- ✅ Intelligent progress tracking with ETA based on video complexity
- ✅ Encoding time and resource tracking
- ✅ Video-only encoding (no audio) for accurate bitrate measurements
- ✅ Automatic video bitrate calculation using FFprobe
- ✅ SHA256 checksums for encoded files
- ✅ Detailed metadata with system info
- ✅ Continue-on-error for resilient batch encoding
- ✅ Dry-run mode to preview encodings
- ✅ Support for all key SVT-AV1 parameters

### Quality Analysis
- ✅ VMAF (NEG mode) - Netflix's perceptual quality metric for codec evaluation
- ✅ PSNR - Traditional pixel difference metric
- ✅ SSIM - Structural similarity metric
- ✅ Real-time progress tracking with accurate ETA estimates
- ✅ Efficiency metrics (VMAF per kbps, quality per encoding second)
- ✅ Comprehensive statistics (mean, harmonic mean, percentiles)
- ✅ FFmpeg integration (no additional dependencies)
- ✅ Multi-threaded VMAF calculation
- ✅ Detailed analysis metadata with summary

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

**`data/encoded/{study_name}/encoding_metadata.json`** - Machine-generated:
- Study configuration (parameters tested)
- System information (CPU, memory, encoder versions)
- Per-encoding results: timing, file size, checksums, success/failure
- Summary statistics for the entire study

This file is NOT committed (in .gitignore) because:
- Generated from running studies
- Reproducible from study config + clips
- Results will be published to GitHub Pages instead

**`data/encoded/{study_name}/measurements.json`** - Machine-generated:
- Quality metrics per encoding (VMAF, PSNR, SSIM)
- VMAF statistics: mean, harmonic mean, percentiles, min/max
- Video info: duration validation, frame count
- Measurement timing

This file is NOT committed (in .gitignore) because:
- Generated from encoded videos
- Reproducible from encodings + source clips
- Large file size with per-frame statistics

## Architecture Philosophy

This repo contains **process and methodology**, not raw results:
- ✅ **Commit:** Code, configs, schemas, documentation
- ✅ **Commit:** Download metadata with checksums (for reproducibility)
- ❌ **Don't commit:** Video files, clips, encodings, analysis results

**Reproducibility through:**
- Video sources with URLs + SHA256 checksums
- Clip extraction with `--seed` parameter
- Study configurations
- Complete automation scripts

**Results distribution:**
- Local development: Run full pipeline, results stay local
- Public results: GitHub Actions → GitHub Pages (planned)
- GitHub Actions cache: Store encoded videos between runs

## Encoding Studies

Studies are focused parameter sweeps stored in `config/studies/`:

**`baseline_sweep.json`** - Main study: comprehensive preset (4-10) and CRF (20-40) sweep
- Purpose: Find optimal speed/quality/size tradeoffs
- ~56 parameter combinations per clip

**`film_grain.json`** - Film grain synthesis study
- Purpose: Test film grain synthesis efficiency
- Fixed preset=6, crf=28, sweeps film_grain levels and denoise flag

**`screen_content.json`** - Screen content mode study
- Purpose: Test scm parameter for screencasts
- Fixed preset=6, crf=28, tests scm=[0,1,2]

**`tune_modes.json`** - Tuning mode comparison
- Purpose: Compare VQ, PSNR, and SSIM tuning
- Fixed preset=6, crf=28, tests tune=[0,1,2]

### Workflow

1. Extract clips with appropriate filters: `just extract-category 3d_animation 5 20 30`
2. Preview study: `just dry-run-study film_grain`
3. Run encoding: `just encode-study film_grain`
4. Measure quality: `just measure-study film_grain`
5. Analyze results: `just analyze-study film_grain`

## Documentation

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guide, code style, testing
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Design decisions and data flow architecture
- **[OVERVIEW.md](OVERVIEW.md)** - Research methodology, goals, and hypotheses
- **[docs/WORKFLOW_EXAMPLE.md](docs/WORKFLOW_EXAMPLE.md)** - Complete end-to-end workflow example
- **[docs/MEASUREMENT_GUIDE.md](docs/MEASUREMENT_GUIDE.md)** - Quality metrics system (VMAF, PSNR, SSIM)
- **[docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md)** - Analysis and plotting system
- **[docs/PROGRESS_TRACKING.md](docs/PROGRESS_TRACKING.md)** - Progress tracking and ETA estimation
- **[docs/VMAF_NOTES.md](docs/VMAF_NOTES.md)** - Why we use VMAF NEG mode for codec evaluation
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## Development

Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development environment setup
- Code quality tools (Ruff, Mypy, Pytest)
- Testing guidelines
- Code style conventions
- Common development tasks

## Project Status

- ✅ Video download system with metadata tracking
- ✅ Clip extraction with filtering and reproducibility
- ✅ Study-based encoding framework
- ✅ Quality metrics calculation (VMAF NEG, PSNR, SSIM)
- ✅ Comprehensive test coverage and CI/CD
- ✅ Analysis visualizations (rate-distortion curves, efficiency plots)
- 📋 Dataset expansion (planned)
- 📋 Interactive HTML reports (planned)
