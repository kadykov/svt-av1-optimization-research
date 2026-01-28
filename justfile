# Justfile for managing video dataset preparation, encoding studies, and analysis

# Default recipe: show all available commands
default:
    @just --list

# Create virtual environment and install Python dependencies  
install:
    python3 -m venv venv
    . venv/bin/activate && pip install --upgrade pip
    . venv/bin/activate && pip install -r requirements.txt

# List all available video sources
list-videos:
    . venv/bin/activate && python scripts/fetch_videos.py --list

# Download all configured videos (small starter set)
fetch-videos:
    . venv/bin/activate && python scripts/fetch_videos.py --all --continue-on-error

# Download specific video by ID (usage: just fetch-one bbb_4k)
fetch-one ID:
    . venv/bin/activate && python scripts/fetch_videos.py --ids {{ID}}

# Download videos by category (usage: just fetch-category 3d_animation)
fetch-category CATEGORY:
    . venv/bin/activate && python scripts/fetch_videos.py --category {{CATEGORY}} --continue-on-error

# Extract test clips (usage: just extract-clips 10 15 30)
# Arguments: <num_clips> <min_duration> <max_duration>
extract-clips NUM MIN_DUR MAX_DUR *ARGS:
    . venv/bin/activate && python scripts/extract_clips.py --num-clips {{NUM}} --min-duration {{MIN_DUR}} --max-duration {{MAX_DUR}} {{ARGS}}

# Extract clips from specific category (usage: just extract-category 3d_animation 5 20 30)
extract-category CATEGORY NUM MIN_DUR MAX_DUR *ARGS:
    . venv/bin/activate && python scripts/extract_clips.py --category {{CATEGORY}} --num-clips {{NUM}} --min-duration {{MIN_DUR}} --max-duration {{MAX_DUR}} {{ARGS}}

# Extract clips with reproducible seed (usage: just extract-seeded 10 15 30 42)
extract-seeded NUM MIN_DUR MAX_DUR SEED *ARGS:
    . venv/bin/activate && python scripts/extract_clips.py --num-clips {{NUM}} --min-duration {{MIN_DUR}} --max-duration {{MAX_DUR}} --seed {{SEED}} {{ARGS}}

# Encode clips according to study configuration (usage: just encode-study baseline_sweep)
encode-study STUDY *ARGS:
    . venv/bin/activate && python scripts/encode_study.py config/studies/{{STUDY}}.json {{ARGS}}

# List available encoding studies
list-studies:
    @echo "Available studies:"
    @ls -1 config/studies/*.json | xargs -n1 basename -s .json | sed 's/^/  /'

# Dry run of encoding study (show what would be encoded)
dry-run-study STUDY:
    . venv/bin/activate && python scripts/encode_study.py config/studies/{{STUDY}}.json --dry-run

# Analyze encoded videos (calculate VMAF, PSNR, SSIM)
analyze-study STUDY *ARGS:
    . venv/bin/activate && python scripts/analyze_study.py {{STUDY}} {{ARGS}}

# Analyze with only VMAF (faster)
analyze-vmaf STUDY *ARGS:
    . venv/bin/activate && python scripts/analyze_study.py {{STUDY}} --metrics vmaf {{ARGS}}

# List studies that have been encoded and are ready for analysis
list-encoded:
    @echo "Encoded studies ready for analysis:"
    @find data/encoded -mindepth 1 -maxdepth 1 -type d -exec test -f {}/encoding_metadata.json \; -printf "  %f\n" 2>/dev/null || echo "  (none)"

# Clean extracted clips (removes videos and generated metadata, keeps schemas)
clean-clips:
    find data/test_clips -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.webm' -o -name '*.mov' -o -name '*.avi' \) -delete
    rm -f data/test_clips/clip_metadata.json

# Clean encoded videos (removes study directories, keeps schemas)
clean-encoded:
    find data/encoded -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# Clean all videos (CAREFUL! Deletes raw videos, clips, and encoded results)
clean-videos:
    find data/raw_videos -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.webm' -o -name '*.mov' -o -name '*.avi' -o -name '*.zip' \) -delete
    rm -f data/raw_videos/download_metadata.json
    just clean-clips
    just clean-encoded
