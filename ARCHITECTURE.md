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

## Git Tracking Strategy

### What Gets Committed

**Code and Configuration:**
- All Python scripts (`scripts/*.py`)
- Study configurations (`config/studies/*.json`)
- Video source configuration (`config/video_sources.json`)
- Schema files (`.schema.json`)
- Documentation (`README.md`, `ARCHITECTURE.md`, `OVERVIEW.md`)
- `requirements.txt`, `justfile`, `.gitignore`

**Verification Data:**
- `data/raw_videos/download_metadata.json` - SHA256 checksums for reproducibility

### What Stays Local (NOT Committed)

**Generated Data:**
- Video files (raw, clips, encoded) - Too large, reproducible via download/extraction
- `data/test_clips/clip_metadata.json` - Generated per extraction, reproducible via seed
- `data/encoded/*/encoding_metadata.json` - Generated per study run
- Analysis results - Generated from encodings

**Rationale:**
1. **Reproducibility via process, not artifacts:**
   - Video sources documented with URLs + checksums
   - Clip extraction reproducible with `--seed` parameter
   - Encoding reproducible with study configs
   - Analysis reproducible from encoding metadata

2. **Size concerns:**
   - Video files can be GBs
   - Encoded results multiply quickly (100s of GB)
   - Git LFS would be required, adds complexity

3. **Research methodology:**
   - Focus is on the *process* and *code*, not specific results
   - Similar to ML research: commit code/config, not trained models
   - Results are observations from running the process

### GitHub Actions + GitHub Pages Architecture

**Planned workflow:**
1. **Repo contains:** Code, configs, methodology
2. **GitHub Action runs:** Full analysis pipeline on schedule or trigger
3. **Action uses cache:** Store encoded videos between runs (keyed by study config hash)
4. **Results published:** GitHub Pages hosts analysis visualizations
5. **Local development:** Users can run same pipeline locally with own videos

**Benefits:**
- Reproducible public results without committing GBs
- Cache prevents re-encoding on every analysis tweak
- Users can verify/extend with own data
- Clear separation: code (git) vs results (GitHub Pages)

**Cache strategy:**
```yaml
# Pseudocode for GitHub Actions
cache_key = hash(study_config + clip_metadata + encoder_version)
if cache_hit:
  skip_encoding()
else:
  run_encoding()
  save_to_cache()
```

## File Organization

### Git Tracking

**Committed to git:**
- `config/video_sources.json` - Source configuration
- `config/studies/*.json` - Study configurations
- `data/raw_videos/download_metadata.json` - Checksums for verification
- All Python scripts
- `requirements.txt`
- Documentation

**NOT committed (in .gitignore):**
- Video files (*.mp4, *.mkv, etc.) - Too large
- Encoded videos (`data/encoded/*/`) - Generated outputs
- `data/test_clips/clip_metadata.json` - Generated per extraction
- Encoding metadata (`data/encoded/*/encoding_metadata.json`) - Generated per study
- Python venv - Reproducible via requirements.txt
- Analysis results - Published to GitHub Pages instead

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

## Encoding Studies Strategy

**Study-based approach instead of exhaustive sweep:**
- Avoids combinatorial explosion of parameters
- Each study focuses on specific hypothesis
- Reuses existing clip filtering (no duplicate logic)
- Extract different clip sets for different studies

**Study configuration:**
- JSON files in `config/studies/`
- Parameters can be single values or arrays (auto-sweep)
- Cartesian product generates all combinations
- Only preset and CRF required (rest optional)

**Output organization:**
- `data/encoded/{study_name}/` per study
- Filename format: `{clip}_p{preset}_crf{crf}[_fg{grain}]...mkv`
- `encoding_metadata.json` tracks all results
- System info included for reproducibility

**Why separate studies:**
1. Baseline sweep focuses on preset/CRF only
2. Film grain study uses fixed preset/CRF, varies grain
3. Each study answers specific question
4. Avoids re-encoding entire collection for each parameter
5. Analysis scripts can be study-specific

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
