# Architecture Decisions

## Metadata Strategy

### Two-File Approach

**`config/video_sources.json`** (Human-edited, version controlled)
- Purpose: Minimal configuration for fetching videos
- Contains: ID, name, URL, categories, license
- Does NOT contain: Technical specs, file sizes, checksums
- Rationale: Keep source config simple and maintainable

**`data/raw_videos/download_metadata.json`** (Machine-generated, version controlled)
- Purpose: Track actual downloaded files with verification data
- Contains: Actual file size (bytes only), SHA256 checksum, file path
- Auto-generated during download
- Rationale: 
  - Provides checksums for reproducibility
  - Records actual file metadata (not estimates)
  - Should be committed to git for checksum verification

### Why This Split?

1. **Human editing**: Users only edit `video_sources.json` with URLs
2. **No redundancy**: Technical metadata only stored once (after download)
3. **Verification**: Checksums enable reproducibility across systems
4. **Precision**: No "expected" sizes vs "actual" sizes - just reality

## File Organization

### Git Tracking

**Committed to git:**
- `config/video_sources.json` - Source configuration
- `data/raw_videos/download_metadata.json` - Checksums for verification
- All Python scripts
- `requirements.txt`
- Documentation

**NOT committed (in .gitignore):**
- Video files (*.mp4, *.mkv, etc.) - Too large
- Encoded videos - Generated outputs
- `data/test_clips/clip_metadata.json` - Generated per extraction
- Python venv - Reproducible via requirements.txt

### Directory Structure

No `.gitkeep` files needed because:
1. Directories documented in README
2. Scripts create directories as needed
3. `download_metadata.json` keeps `raw_videos/` directory naturally

### Clip Extraction Philosophy

**Clean slate approach:**
- Each extraction cleans the output directory first (default behavior)
- Ensures metadata always matches actual clips
- No orphaned files without metadata
- No confusion about which clips are from which extraction

**Rationale:**
- Simpler than merging metadata from multiple extractions
- Reproducible with `--seed` parameter
- Each extraction is a complete, self-contained set
- Use `--no-clean` flag if you want to keep existing clips

**Benefits:**
- Metadata file always accurate
- No file overwrites (clean slate)
- Clear what parameters created current clip set
- Easy to recreate exact same clips with same seed

## Justfile Philosophy

**Keep only useful commands:**
- `install` - Setup environment
- `list-videos` - See what's available
- `fetch-videos` - Download all
- `fetch-one <id>` - Download specific
- `fetch-category <cat>` - Download by category (generic, not hardcoded)
- `clean-videos` - Cleanup

**Removed bloat:**
- ~~`list-categories`~~ - Can use `--help` on script
- ~~`video-info`~~ - Can use `--help` on script
- ~~`disk-usage`~~ - Use standard `du` command
- ~~`tree`~~ - Use standard `tree` command
- ~~`fetch-3d`~~ - Replaced with generic `fetch-category`

Justfile = Quick reference of *actions*, not a wrapper for every possible query.

## Zip File Handling

Videos in `.zip` archives are automatically extracted during download:
1. Download to temporary location (`{id}_temp.zip`)
2. Extract video file from archive
3. Move to expected location
4. Delete zip file
5. Calculate checksum of extracted video

This keeps the data directory clean - only actual video files, no archives.

## Requirements Management

**No frozen requirements:**
- `pip` doesn't have deterministic lock files like `uv` or `npm`
- `pip freeze` is not cross-platform reproducible
- Keep `requirements.txt` with version ranges (>=)
- Dev container ensures consistent environment

Alternative: Could use `uv` or `poetry` for true lock files, but adds complexity.

## Download Strategy

**Fallback mechanism:**
- One failed download doesn't stop the batch
- `--continue-on-error` flag
- Useful when URLs break over time
- Allows partial dataset reconstruction

**Resume capability:**
- HTTP Range requests for interrupted downloads
- Retry from existing byte position
- Handles servers that don't support resume (restart)

**Checksum verification:**
- SHA256 calculated after download
- Stored in metadata for future verification
- Can detect corrupted/modified files
