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

# Clean extracted clips (keeps raw videos)
clean-clips:
    rm -rf data/test_clips/*.mp4 data/test_clips/*.mkv data/test_clips/*.webm data/test_clips/*.mov data/test_clips/*.avi
    rm -f data/test_clips/clip_metadata.json

# Clean downloaded videos (CAREFUL! This deletes all video files)
clean-videos:
    rm -rf data/raw_videos/*.mp4 data/raw_videos/*.mkv data/raw_videos/*.webm data/raw_videos/*.mov data/raw_videos/*.avi
    rm -rf data/test_clips/*
    rm -rf data/encoded/*
