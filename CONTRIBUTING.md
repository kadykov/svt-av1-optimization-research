# Contributing to SVT-AV1 Optimization Research

Thank you for your interest in contributing! This guide will help you set up your development environment and understand the project's development workflows.

## Quick Start

This project uses a dev container with all tools pre-configured and a `justfile` for common tasks.

### First-Time Setup

The dev container automatically installs all dependencies on first start. If you need to reinstall or are working outside the container:

```bash
# Install production dependencies
just install

# Install development dependencies (linting, testing, type checking, pre-commit hooks)
just install-dev
```

## Development Workflow

### Before You Start

Always work on a feature branch:
```bash
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write your code** with type hints and docstrings
2. **Run linting and formatting**:
   ```bash
   just lint     # Auto-fix linting issues
   just format   # Format code
   ```
3. **Add/update tests** for your changes
4. **Run tests locally**:
   ```bash
   just test-unit         # Fast unit tests
   just test-integration  # Integration tests (requires FFmpeg)
   just test              # All tests
   ```
5. **Type check your code**:
   ```bash
   just type-check
   ```
6. **Run all checks at once**:
   ```bash
   just check-all  # Runs lint + format-check + type-check + tests
   ```

### Committing Changes

Pre-commit hooks automatically run before each commit:
```bash
git add .
git commit -m "feat: add new feature"  # Hooks run automatically
```

If hooks fail, fix the issues and commit again. The hooks will auto-fix most issues.

## Code Quality Tools

### Ruff - Linter and Formatter

Fast Python linter and formatter that replaces black, isort, flake8, and more.

- **Config**: [pyproject.toml](pyproject.toml) `[tool.ruff]`
- **Auto-fixes** most issues
- **Rules enabled**: 150+ rules across 19 categories

```bash
# Check for issues (no changes)
just lint

# Check formatting (no changes)
just format-check
```

### Mypy - Static Type Checker

Catches type-related bugs before runtime.

- **Config**: [pyproject.toml](pyproject.toml) `[tool.mypy]`
- **Mode**: Currently permissive (gradual adoption)

```bash
just type-check
```

### Pytest - Testing Framework

- **Config**: [pyproject.toml](pyproject.toml) `[tool.pytest.ini_options]`
- **Test structure**:
  ```
  tests/
  ├── unit/          # Fast tests, no external dependencies
  └── integration/   # Tests requiring FFmpeg, VMAF, etc.
  ```

```bash
just test              # All tests
just test-unit         # Unit tests only (fast)
just test-integration  # Integration tests (requires FFmpeg)
just test-coverage     # Generate coverage report
```

### Pre-commit Hooks

Automatically run checks before each commit (installed by `just install-dev`):
- Ruff linting and formatting
- Mypy type checking
- File size checks (prevents commits >1MB - catches accidental video commits!)
- JSON/YAML validation
- Trailing whitespace removal

**Manual execution**:
```bash
# Run on all files
pre-commit run --all-files

# Skip hooks for a commit (not recommended)
git commit --no-verify
```

## Code Style Guidelines

### Type Hints

All functions should have type hints for parameters and return values:

```python
from pathlib import Path

def process_video(input_path: Path, duration: float) -> dict[str, Any]:
    """Process a video file."""
    return {"status": "success"}
```

**Note**: Use modern Python 3.9+ syntax:
- `dict[str, Any]` instead of `Dict[str, Any]`
- `list[str]` instead of `List[str]`
- `Path` from `pathlib` instead of `str` for file paths

### Imports

Imports are automatically organized by Ruff:
1. Standard library imports
2. Third-party package imports
3. Local module imports

```python
# Standard library
import json
from pathlib import Path
from datetime import datetime, timezone

# Third-party
import requests
from tqdm import tqdm

# Local
from utils import calculate_sha256
```

### Docstrings

Use Google-style docstrings:

```python
def encode_clip(
    clip_path: Path,
    output_path: Path,
    preset: int,
    crf: int
) -> dict[str, Any]:
    """
    Encode a video clip with specified SVT-AV1 parameters.

    Args:
        clip_path: Path to input video clip
        output_path: Path for encoded output
        preset: SVT-AV1 preset (0-13, higher is faster)
        crf: Constant Rate Factor (0-63, higher is lower quality)

    Returns:
        Dictionary with encoding results including timing and file size

    Raises:
        FileNotFoundError: If clip_path doesn't exist
        subprocess.TimeoutExpired: If encoding takes >1 hour
    """
```

### Error Handling

Be explicit about error handling:

```python
try:
    result = subprocess.run(cmd, capture_output=True, check=True, timeout=3600)
except subprocess.TimeoutExpired:
    print("Encoding timed out after 1 hour")
    return {"error": "timeout"}
except subprocess.CalledProcessError as e:
    print(f"Encoding failed: {e.stderr.decode()}")
    return {"error": "encoding_failed"}
```

## Common Development Tasks

### Adding a New Script

1. Create script in `scripts/` directory
2. Add type hints to all functions
3. Add docstrings
4. Import from `utils` for shared functionality
5. Add unit tests in `tests/unit/`
6. Add integration tests in `tests/integration/` if needed
7. Run `just check-all` to verify everything passes
8. Commit changes

### Refactoring Existing Code

1. Create tests for current behavior (regression tests)
2. Make changes
3. Run `just lint` and `just format`
4. Run `just type-check` to catch type errors
5. Run `just test` to ensure tests still pass
6. Commit changes

### Adding Dependencies

1. Add to `pyproject.toml` `dependencies` section with version constraint
2. Run `just install` to install
3. Document why the dependency is needed in commit message
4. Ensure tests pass with new dependency

### Creating Test Fixtures

Small test video files (< 100KB) can be committed to `tests/fixtures/`:

```bash
# Generate a 2-second test video
ffmpeg -f lavfi -i testsrc=duration=2:size=320x240:rate=24 \
       -f lavfi -i sine=frequency=1000:duration=2 \
       -c:v libx264 -preset ultrafast -crf 30 \
       -c:a aac -b:a 96k tests/fixtures/test_video.mp4 -y
```

## CI/CD Pipeline

GitHub Actions automatically runs on every push and pull request:

1. **Lint** - Ruff linter and formatter checks
2. **Type Check** - Mypy static analysis
3. **Tests** - Unit and integration tests (with FFmpeg)
4. **Security** - Bandit vulnerability scanning

**View results**: GitHub repository → Actions tab

**Note**: Failed CI blocks merging unless overridden.

## Testing

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures and pytest configuration
├── unit/                 # Fast tests without external dependencies
│   └── test_utils.py
├── integration/          # Tests requiring FFmpeg, VMAF, etc.
│   ├── test_encode_study.py
│   └── test_extract_clips.py
└── fixtures/             # Test data (videos, configs)
    └── test_video.mp4    # Small test video (51KB)
```

### Writing Tests

```python
import pytest
from pathlib import Path
from utils import calculate_sha256

@pytest.mark.unit
def test_calculate_sha256():
    """Test SHA256 calculation."""
    # Create temporary file
    test_file = Path("test.txt")
    test_file.write_text("hello world")

    # Calculate hash
    result = calculate_sha256(test_file)

    # Verify
    assert result == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    # Cleanup
    test_file.unlink()

@pytest.mark.integration
@pytest.mark.requires_ffmpeg
def test_encoding():
    """Test video encoding with FFmpeg."""
    # Test implementation
    pass
```

### Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests requiring external tools
- `@pytest.mark.slow` - Tests taking >10 seconds
- `@pytest.mark.requires_ffmpeg` - Requires FFmpeg installation
- `@pytest.mark.requires_vmaf` - Requires VMAF model files

### Running Tests

```bash
# All tests
just test

# Only unit tests (fast)
just test-unit

# Only integration tests
just test-integration

# With coverage report
just test-coverage

# Run specific test file
pytest tests/unit/test_utils.py -v

# Run specific test
pytest tests/unit/test_utils.py::test_calculate_sha256 -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s
```

## Troubleshooting

### ImportError: No module named 'utils'

The `scripts/` directory needs to be in Python path. The justfile commands handle this automatically, but for direct execution:

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/scripts"
```

### Pre-commit Hook Failures

If pre-commit hooks fail:

```bash
# View what failed
git status

# Fix issues
just lint
just format

# Re-run hooks
pre-commit run --all-files

# Commit again
git commit
```

### Large File Commit Blocked

Pre-commit prevents files >1MB to catch accidental video commits. If you need to commit a large file (not recommended):

```bash
git add -f path/to/large/file
```

**Better approach**: Store large files externally and document download URLs in `config/video_sources.json`.

### CI Pipeline Failures

1. Run `just check-all` locally to reproduce the issue
2. Fix the failing checks
3. Commit and push again

### FFmpeg Not Found

If you're not using the dev container:

```bash
# Check FFmpeg installation
ffmpeg -version

# Check if libvmaf is available
ffmpeg -filters 2>&1 | grep libvmaf
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for container rebuild instructions.

## Quick Command Reference

```bash
# Setup
just install              # Install production dependencies
just install-dev          # Install dev dependencies + hooks

# Code quality
just lint                 # Lint and auto-fix
just format               # Format code
just format-check         # Check formatting (no changes)
just type-check           # Type check with mypy
just check-all            # Run all checks + tests

# Testing
just test                 # All tests
just test-unit            # Unit tests only
just test-integration     # Integration tests only
just test-coverage        # Generate coverage report

# Video processing (for testing)
just fetch-videos         # Download test videos
just extract-clips N MIN MAX  # Extract N clips of MIN-MAX seconds
just encode-study STUDY   # Run encoding study
just analyze-study STUDY  # Calculate quality metrics
```

**See all available commands**:
```bash
just --list
```

## Project Documentation

- **[README.md](README.md)** - Project overview and quick start
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Design decisions and data flow
- **[OVERVIEW.md](OVERVIEW.md)** - Research methodology and goals
- **[docs/MEASUREMENT_GUIDE.md](docs/MEASUREMENT_GUIDE.md)** - Quality metrics and measurement system
- **[docs/VMAF_NOTES.md](docs/VMAF_NOTES.md)** - Why we use VMAF NEG mode
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Python Type Hints (PEP 484)](https://peps.python.org/pep-0484/)
- [SVT-AV1 Documentation](https://gitlab.com/AOMediaCodec/SVT-AV1)

## Questions?

If you have questions or need help:
1. Check the documentation in the `docs/` directory
2. Look for similar issues in the GitHub Issues
3. Open a new issue with your question

Thank you for contributing!
