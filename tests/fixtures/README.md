# Test Fixtures

This directory contains test fixtures for the SVT-AV1 optimization research test suite.

## Included Fixtures

### test_video.mp4
A small 2-second test video suitable for fast integration testing.

**Specifications**:
- **Duration**: 2 seconds
- **Resolution**: 320x240
- **Frame rate**: 24 fps
- **Video codec**: H.264 (x264, CRF 30, ultrafast preset)
- **Audio codec**: AAC (96 kbps, mono, 1kHz sine wave)
- **File size**: ~51KB

**Purpose**: Integration tests for video processing scripts without requiring large files.

## Generating New Test Fixtures

If you need to create additional test fixtures, use FFmpeg to generate small synthetic videos:

### Basic Test Video (Color Bars)

```bash
ffmpeg -f lavfi -i testsrc=duration=2:size=320x240:rate=24 \
       -f lavfi -i sine=frequency=1000:duration=2 \
       -c:v libx264 -preset ultrafast -crf 30 \
       -c:a aac -b:a 96k \
       tests/fixtures/test_video.mp4 -y
```

### Other Useful Patterns

**Solid color**:
```bash
ffmpeg -f lavfi -i color=red:size=320x240:duration=2:rate=24 \
       -c:v libx264 -preset ultrafast -crf 30 \
       tests/fixtures/red_video.mp4 -y
```

**Noise pattern** (tests compression better):
```bash
ffmpeg -f lavfi -i noise=duration=2:size=320x240:rate=24 \
       -c:v libx264 -preset ultrafast -crf 30 \
       tests/fixtures/noise_video.mp4 -y
```

## Size Guidelines

- **Target size**: < 100KB per fixture
- **Maximum size**: 500KB (larger files may slow down CI)
- Use high CRF values (25-35) for smaller files
- Use `ultrafast` or `veryfast` preset
- Keep resolution low (320x240 or 640x480)
- Keep duration short (2-5 seconds)

## Gitignore Policy

Small test fixtures (< 500KB) **should be committed** to the repository.
Larger files and test outputs are ignored:

**Committed** (tracked):
- `test_video.mp4` and other small fixtures
- This README

**Ignored** (not tracked):
- `*.ivf` - Encoded test outputs
- `output_*` - Generated during tests
- Any files > 1MB (blocked by pre-commit hook)
