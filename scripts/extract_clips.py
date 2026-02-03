#!/usr/bin/env python3
"""
Extract short test clips from downloaded videos for encoding tests.

Features:
- Filter by category, resolution, FPS
- Random fragment selection with optional seed
- Duration range specification
- Proportional extraction (longer videos -> more clips)
- FFprobe for metadata, FFmpeg for extraction
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from utils import calculate_sha256


def get_video_info(video_path: Path) -> dict | None:
    """Get video duration, resolution, and FPS using FFprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Get duration (prefer stream duration, fallback to format duration)
        duration = None
        if "streams" in data and len(data["streams"]) > 0:
            stream = data["streams"][0]
            if "duration" in stream:
                duration = float(stream["duration"])

        if duration is None and "format" in data and "duration" in data["format"]:
            duration = float(data["format"]["duration"])

        if duration is None:
            print(f"  ⚠️  Could not determine duration for {video_path.name}", file=sys.stderr)
            return None

        # Get resolution and FPS
        stream = data["streams"][0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))

        # Parse frame rate (e.g., "60/1" or "24000/1001")
        fps_str = stream.get("r_frame_rate", "0/1")
        num, denom = map(int, fps_str.split("/"))
        fps = num / denom if denom > 0 else 0

        return {"duration": duration, "width": width, "height": height, "fps": fps}
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  ⚠️  Error probing {video_path.name}: {e}", file=sys.stderr)
        return None


def extract_clip(video_path: Path, start_time: float, duration: float, output_path: Path) -> bool:
    """Extract a clip from video using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-i",
        str(video_path),
        "-t",
        str(duration),
        "-c",
        "copy",  # Copy streams without re-encoding
        "-avoid_negative_ts",
        "1",
        "-y",  # Overwrite output
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to extract clip: {e.stderr.decode()}", file=sys.stderr)
        return False


def filter_videos(
    videos: list[dict],
    category: str | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    min_fps: float | None = None,
    max_fps: float | None = None,
) -> list[dict]:
    """Filter videos based on criteria."""
    filtered = videos

    if category:
        filtered = [v for v in filtered if category in v.get("categories", [])]

    if max_width:
        filtered = [v for v in filtered if v["info"]["width"] <= max_width]

    if max_height:
        filtered = [v for v in filtered if v["info"]["height"] <= max_height]

    if min_fps:
        filtered = [v for v in filtered if v["info"]["fps"] >= min_fps]

    if max_fps:
        filtered = [v for v in filtered if v["info"]["fps"] <= max_fps]

    return filtered


def generate_random_clips(
    videos: list[dict],
    num_clips: int,
    min_duration: float,
    max_duration: float,
    seed: int | None = None,
) -> list[dict]:
    """Generate random clip specifications proportional to video duration."""
    if seed is not None:
        random.seed(seed)

    # Calculate total duration and weights (using usable duration)
    total_duration = 0
    for v in videos:
        time_range = v.get("usable_time_range")
        if time_range:
            total_duration += time_range["end"] - time_range["start"]
        else:
            total_duration += v["info"]["duration"]

    clips = []

    # Distribute clips proportionally
    for video in videos:
        # Determine usable time range
        time_range = video.get("usable_time_range")
        if time_range:
            usable_start = time_range["start"]
            usable_end = time_range["end"]
            usable_duration = usable_end - usable_start
        else:
            usable_start = 0.0
            usable_end = video["info"]["duration"]
            usable_duration = video["info"]["duration"]

        proportion = usable_duration / total_duration
        video_clips = max(1, round(num_clips * proportion))  # At least 1 clip per video

        for _ in range(video_clips):
            # Random clip duration within range
            clip_duration = random.uniform(min_duration, max_duration)

            # Ensure clip fits in usable range
            max_start = usable_end - clip_duration
            if max_start < usable_start:
                # Usable range is shorter than clip duration, use entire usable range
                start_time = usable_start
                clip_duration = usable_duration
            else:
                start_time = random.uniform(usable_start, max_start)

            clips.append(
                {
                    "video_id": video["id"],
                    "video_path": video["path"],
                    "start_time": start_time,
                    "duration": clip_duration,
                    "source_width": video["info"]["width"],
                    "source_height": video["info"]["height"],
                    "source_fps": video["info"]["fps"],
                    "categories": video.get("categories", []),
                }
            )

    # Shuffle and limit to requested number
    random.shuffle(clips)
    return clips[:num_clips]


def main():
    parser = argparse.ArgumentParser(
        description="Extract short test clips from downloaded videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 10 clips, 15-30 seconds each
  python extract_clips.py --num-clips 10 --min-duration 15 --max-duration 30

  # Extract from 3D animation only
  python extract_clips.py --num-clips 5 --category 3d_animation --min-duration 20 --max-duration 20

  # Extract only Full HD or lower, 60fps+ videos
  python extract_clips.py --num-clips 8 --max-height 1080 --min-fps 60

  # Reproducible extraction with seed
  python extract_clips.py --num-clips 10 --seed 42 --min-duration 10 --max-duration 30
        """,
    )

    parser.add_argument("--num-clips", type=int, required=True, help="Number of clips to extract")
    parser.add_argument(
        "--min-duration", type=float, required=True, help="Minimum clip duration in seconds"
    )
    parser.add_argument(
        "--max-duration", type=float, required=True, help="Maximum clip duration in seconds"
    )

    # Filtering options
    parser.add_argument(
        "--category", type=str, help="Filter by category (e.g., 3d_animation, mixed_content)"
    )
    parser.add_argument("--max-width", type=int, help="Maximum video width in pixels")
    parser.add_argument("--max-height", type=int, help="Maximum video height in pixels")
    parser.add_argument("--min-fps", type=float, help="Minimum FPS")
    parser.add_argument("--max-fps", type=float, help="Maximum FPS")

    # Other options
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/test_clips",
        help="Output directory for clips (default: data/test_clips)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean output directory before extraction (default: clean first)",
    )

    args = parser.parse_args()

    # Validate duration range
    if args.min_duration > args.max_duration:
        print("❌ Error: min-duration cannot be greater than max-duration", file=sys.stderr)
        return 1

    if args.min_duration <= 0:
        print("❌ Error: min-duration must be positive", file=sys.stderr)
        return 1

    # Load metadata
    metadata_path = Path("data/raw_videos/download_metadata.json")
    sources_path = Path("config/video_sources.json")

    if not metadata_path.exists():
        print(
            "❌ Error: download_metadata.json not found. Run fetch_videos.py first.",
            file=sys.stderr,
        )
        return 1

    if not sources_path.exists():
        print("❌ Error: video_sources.json not found.", file=sys.stderr)
        return 1

    with open(metadata_path) as f:
        download_metadata = json.load(f)

    with open(sources_path) as f:
        video_sources = json.load(f)

    # Build video source lookup
    sources_by_id = {v["id"]: v for v in video_sources["sources"]}

    # Probe all downloaded videos
    print("🔍 Probing downloaded videos...")
    videos = []
    for video_id, metadata in download_metadata.get("downloads", {}).items():
        video_path = Path(metadata["file_path"])
        if not video_path.exists():
            print(f"  ⚠️  Skipping {video_id}: file not found", file=sys.stderr)
            continue

        print(f"  📹 Probing {video_path.name}...")
        info = get_video_info(video_path)
        if info is None:
            continue

        # Get categories and usable time range from sources
        source = sources_by_id.get(video_id, {})
        categories = source.get("categories", [])
        usable_time_range = source.get("usable_time_range")

        videos.append(
            {
                "id": video_id,
                "path": str(video_path),
                "info": info,
                "categories": categories,
                "usable_time_range": usable_time_range,
            }
        )

        print(
            f"     Duration: {info['duration']:.1f}s, "
            f"Resolution: {info['width']}x{info['height']}, "
            f"FPS: {info['fps']:.2f}"
        )

    if not videos:
        print("❌ Error: No valid videos found", file=sys.stderr)
        return 1

    print(f"\n✅ Found {len(videos)} valid videos")

    # Filter videos
    print("\n🔍 Applying filters...")
    filtered_videos = filter_videos(
        videos,
        category=args.category,
        max_width=args.max_width,
        max_height=args.max_height,
        min_fps=args.min_fps,
        max_fps=args.max_fps,
    )

    if not filtered_videos:
        print("❌ Error: No videos match the filter criteria", file=sys.stderr)
        return 1

    print(f"✅ {len(filtered_videos)} videos match criteria:")
    for v in filtered_videos:
        print(f"   - {v['id']} ({v['info']['duration']:.1f}s)")

    # Generate random clip specifications
    print(f"\n🎲 Generating {args.num_clips} random clip specifications...")
    if args.seed is not None:
        print(f"   Using seed: {args.seed}")

    clip_specs = generate_random_clips(
        filtered_videos, args.num_clips, args.min_duration, args.max_duration, args.seed
    )

    print(f"✅ Generated {len(clip_specs)} clip specifications")

    # Clean output directory if requested (default behavior)
    if not args.no_clean and args.output_dir.exists():
        print(f"\n🧹 Cleaning output directory {args.output_dir}...")
        # Remove all video files and metadata
        for pattern in ["*.mp4", "*.mkv", "*.webm", "*.mov", "*.avi"]:
            for file in args.output_dir.glob(pattern):
                file.unlink()
                print(f"   Deleted: {file.name}")

        # Remove clip metadata if exists
        metadata_file = args.output_dir / "clip_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
            print("   Deleted: clip_metadata.json")

        print("✅ Directory cleaned")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract clips
    print(f"\n✂️  Extracting clips to {args.output_dir}...")
    clip_metadata = []

    for i, spec in enumerate(clip_specs, 1):
        # Generate clip filename
        clip_name = f"{spec['video_id']}_clip_{i:03d}.mp4"
        output_path = args.output_dir / clip_name

        print(f"  [{i}/{len(clip_specs)}] {clip_name}")
        print(f"     Source: {Path(spec['video_path']).name}")
        print(
            f"     Time: {spec['start_time']:.2f}s - {spec['start_time'] + spec['duration']:.2f}s "
            f"(duration: {spec['duration']:.2f}s)"
        )

        # Extract clip
        success = extract_clip(
            Path(spec["video_path"]), spec["start_time"], spec["duration"], output_path
        )

        if not success:
            print("     ⚠️  Failed to extract clip", file=sys.stderr)
            continue

        # Calculate checksum
        checksum = calculate_sha256(output_path)
        file_size = output_path.stat().st_size

        print(f"     ✅ Extracted: {file_size / 1024 / 1024:.2f} MB")

        # Get actual duration of extracted clip (FFmpeg may extract slightly different duration)
        actual_info = get_video_info(output_path)
        actual_duration = actual_info["duration"] if actual_info else spec["duration"]

        # Store metadata
        clip_metadata.append(
            {
                "clip_name": clip_name,
                "video_id": spec["video_id"],
                "start_time": spec["start_time"],
                "duration": spec["duration"],
                "actual_duration": actual_duration,
                "source_width": spec["source_width"],
                "source_height": spec["source_height"],
                "source_fps": spec["source_fps"],
                "categories": spec["categories"],
                "file_size_bytes": file_size,
                "sha256": checksum,
            }
        )

    # Save clip metadata
    metadata_output = args.output_dir / "clip_metadata.json"
    with open(metadata_output, "w") as f:
        json.dump(
            {
                "extraction_params": {
                    "num_clips": args.num_clips,
                    "min_duration": args.min_duration,
                    "max_duration": args.max_duration,
                    "category": args.category,
                    "max_width": args.max_width,
                    "max_height": args.max_height,
                    "min_fps": args.min_fps,
                    "max_fps": args.max_fps,
                    "seed": args.seed,
                },
                "clips": clip_metadata,
            },
            f,
            indent=2,
        )

    print("\n✅ Extraction complete!")
    print(f"   Clips saved to: {args.output_dir}")
    print(f"   Metadata saved to: {metadata_output}")
    print(f"   Total clips: {len(clip_metadata)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
