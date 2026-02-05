# Troubleshooting Guide

Common issues and solutions for the SVT-AV1 Optimization Research project.

## Development Environment Issues

### Dev Container: FFmpeg Missing VMAF Support

**Symptoms:**
```
[AVFilterGraph @ 0x...] No such filter: 'libvmaf'
Error parsing global options: Filter not found
```

**Cause:** FFmpeg doesn't have libvmaf filter compiled in.

**Solution:** Rebuild the dev container

#### Option 1: VS Code Command Palette (Easiest)

1. Press `Ctrl+Shift+P` (Linux/Windows) or `Cmd+Shift+P` (Mac)
2. Type: `Dev Containers: Rebuild Container`
3. Press Enter
4. Wait for the container to rebuild (2-3 minutes)

#### Option 2: VS Code UI

1. Click the blue icon in the bottom-left corner (looks like "><")
2. Select "Rebuild Container"
3. Wait for rebuild

#### Option 3: Command Line

```bash
# From the host machine (not inside the container)
docker-compose -f .devcontainer/docker-compose.yml down
docker-compose -f .devcontainer/docker-compose.yml build --no-cache
docker-compose -f .devcontainer/docker-compose.yml up -d
```

#### After Rebuild - Verify FFmpeg

```bash
# Check libvmaf filter exists
ffmpeg -filters 2>&1 | grep libvmaf
```

Expected output:
```
 ... libvmaf           VV->V      Calculate the VMAF Motion score.
```

```bash
# Check libvmaf filter help
ffmpeg -hide_banner -h filter=libvmaf
```

Should show libvmaf filter options.

#### Test Analysis

```bash
# Run with verbose mode to see what's happening
just analyze-study baseline_sweep -v
```

### What's Included in the Static FFmpeg Build

The dev container uses John Van Sickle's static FFmpeg builds which include:
- ✅ libvmaf included
- ✅ Latest FFmpeg version
- ✅ Many codecs: SVT-AV1, AV1, VP9, x264, x265, etc.
- ✅ Static binary (no shared library issues)

### Container Rebuild Takes Too Long

First rebuild downloads ~130MB FFmpeg static build. Subsequent rebuilds are faster due to Docker layer caching.

To speed up:
```bash
# Rebuild without cache only if having issues
# In VS Code: Dev Containers: Rebuild Container Without Cache
```

## Python Environment Issues

### ImportError: No module named 'utils'

**Cause:** The `scripts/` directory is not in Python path.

**Solution:**
```bash
# Add to path temporarily
export PYTHONPATH="${PYTHONPATH}:${PWD}/scripts"

# Or use justfile commands (they handle this automatically)
just encode-study baseline_sweep
```

### Module Not Found Errors After Pulling

**Solution:** Reinstall dependencies
```bash
just install
# or for development
just install-dev
```

### AttributeError: 'datetime.datetime' has no attribute 'UTC'

**Cause:** Old code used Python 3.11+ only syntax.

**Solution:** Update to latest code (fixed to use `timezone.utc`):
```bash
git pull origin main
```

Verify Python version:
```bash
python --version  # Should be 3.9+
```

## Testing Issues

### Pytest Not Found

**Solution:**
```bash
just install-dev
```

### Pre-commit Hooks Not Running

**Solution:**
```bash
just install-dev  # This installs pre-commit hooks
```

Verify hooks are installed:
```bash
pre-commit run --all-files
```

### Tests Fail: No Test Fixtures

**Cause:** Test video fixtures not generated.

**Solution:**
```bash
# Generate test video fixture
ffmpeg -f lavfi -i testsrc=duration=2:size=320x240:rate=24 \
       -f lavfi -i sine=frequency=1000:duration=2 \
       -c:v libx264 -preset ultrafast -crf 30 \
       -c:a aac -b:a 96k tests/fixtures/test_video.mp4 -y
```

### Pytest Unrecognized Arguments

**Symptoms:**
```
ERROR: unrecognized arguments: --cov --timeout
```

**Cause:** Coverage or timeout plugins not installed.

**Solution:**
```bash
# Install dev dependencies
just install-dev

# Or run tests without coverage
pytest tests/

# Or install plugins manually
pip install pytest-cov pytest-timeout
```

## Code Quality Issues

### Ruff or Mypy Not Found

**Solution:**
```bash
just install-dev
```

### Pre-commit Hook Failures

**Symptoms:** Commit blocked by pre-commit hooks.

**Solution:**
```bash
# View what failed
git status

# Fix issues automatically
just lint
just format

# Re-run hooks manually
pre-commit run --all-files

# Commit again
git commit
```

### Large File Commit Blocked

**Symptoms:**
```
Check for added large files..............Failed
- hook id: check-added-large-files
- exit code: 1

large_file.mp4 (10 MB) exceeds 1 MB.
```

**Cause:** Pre-commit prevents files >1MB to catch accidental video commits.

**Solution:** Don't commit large files. Instead:
1. Add download URLs to `config/video_sources.json`
2. Let users download with `just fetch-videos`

**If you really need to commit a large file** (not recommended):
```bash
git add -f path/to/large/file
git commit --no-verify
```

## Video Processing Issues

### FFmpeg Command Not Found

**Solution:** Ensure you're in the dev container or install FFmpeg:

```bash
# Check FFmpeg
which ffmpeg
ffmpeg -version

# In dev container, it should be at /usr/local/bin/ffmpeg
```

### VMAF Analysis Fails: Invalid Syntax

**Symptoms:**
```
Error applying option 'version' to filter 'libvmaf': Option not found
```

**Cause:** Incorrect FFmpeg filter syntax (this was fixed in recent updates).

**Solution:** Update to latest code:
```bash
git pull origin main
```

The correct syntax is:
```bash
libvmaf=model=version=vmaf_v0.6.1neg:log_path=...:log_fmt=json
```

### Encoding Fails: Invalid Preset

**Symptoms:**
```
Unrecognized option 'preset'.
```

**Cause:** FFmpeg doesn't have SVT-AV1 encoder.

**Solution:** Rebuild dev container to get static FFmpeg build with SVT-AV1:
```bash
# In VS Code: Dev Containers: Rebuild Container
```

Verify SVT-AV1 is available:
```bash
ffmpeg -codecs 2>&1 | grep av1
```

Should show `libsvtav1` in the output.

### Download Fails: Connection Error

**Symptoms:**
```
Failed to download: Connection timeout
```

**Solutions:**
1. Check internet connection
2. Try again (temporary network issue)
3. Verify URL in `config/video_sources.json` is still valid
4. Try downloading manually and placing in `data/raw_videos/`

### Clip Extraction: No Clips Generated

**Cause:** Filters too restrictive or no videos downloaded.

**Solution:**
```bash
# Check if videos are downloaded
ls -lh data/raw_videos/

# Download videos if missing
just fetch-videos

# Try extraction without filters
just extract-clips 5 10 20
```

## CI/CD Issues

### GitHub Actions Failing

**Solution:** Run checks locally first:
```bash
just check-all
```

Fix any issues, then push:
```bash
git add .
git commit -m "fix: resolve CI issues"
git push
```

### CI Timeout

**Cause:** Integration tests with large videos can timeout.

**Solution:** Tests should use small fixtures. Check that integration tests use `tests/fixtures/test_video.mp4` (51KB) and not full videos.

## General Debugging

### Enable Verbose Mode

Most scripts support `-v` or `--verbose`:
```bash
just encode-study baseline_sweep -v
just analyze-study baseline_sweep -v
```

### Check Script Help

```bash
python scripts/fetch_videos.py --help
python scripts/extract_clips.py --help
python scripts/encode_study.py --help
python scripts/analyze_study.py --help
```

### Inspect Metadata Files

```bash
# Check downloaded videos
cat data/raw_videos/download_metadata.json | jq

# Check extracted clips
cat data/test_clips/clip_metadata.json | jq

# Check encoding results
cat data/encoded/baseline_sweep/encoding_metadata.json | jq

# Check analysis results
cat data/encoded/baseline_sweep/analysis_metadata.json | jq
```

### Clean and Start Fresh

```bash
# Clean generated data (keeps raw videos)
rm -rf data/test_clips/*
rm -rf data/encoded/*
rm -rf results/*

# Download videos again
just fetch-videos

# Extract clips
just extract-clips 5 15 30

# Run study
just encode-study baseline_sweep
just analyze-study baseline_sweep
```

## Getting Help

If you're still stuck:

1. **Check the documentation**:
   - [README.md](README.md) - Overview and quick start
   - [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
   - [docs/MEASUREMENT_GUIDE.md](docs/MEASUREMENT_GUIDE.md) - Measurement system
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions

2. **Search GitHub Issues**: Someone may have had the same problem

3. **Open a new issue**: Include:
   - What you were trying to do
   - The exact command you ran
   - The error message (full output)
   - Your environment (OS, Python version, inside/outside dev container)

## Useful Commands

```bash
# Check Python environment
python --version
pip list

# Check FFmpeg
ffmpeg -version
ffmpeg -encoders 2>&1 | grep svt
ffmpeg -filters 2>&1 | grep vmaf

# Check disk space (large videos can fill disk)
df -h

# Check running processes
ps aux | grep ffmpeg

# Kill stuck FFmpeg processes
pkill ffmpeg

# View justfile commands
just --list
```
