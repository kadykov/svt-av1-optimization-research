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

# Clean downloaded videos (CAREFUL! This deletes all video files)
clean-videos:
    rm -rf data/raw_videos/*.mp4 data/raw_videos/*.mkv data/raw_videos/*.webm data/raw_videos/*.mov data/raw_videos/*.avi
    rm -rf data/test_clips/*
    rm -rf data/encoded/*
