#!/usr/bin/env python3
"""
Video fetcher for SVT-AV1 optimization research project.

This script downloads video files from configured sources with:
- Resume capability for interrupted downloads
- Fallback mechanism (continues on failure)
- Progress tracking
- Checksum verification
- Metadata tracking
- Automatic zip extraction
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


class VideoFetcher:
    """Handles downloading and managing video test files."""
    
    def __init__(self, config_path: str, output_dir: str):
        """
        Initialize the video fetcher.
        
        Args:
            config_path: Path to video_sources.json
            output_dir: Directory to save downloaded videos
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.output_dir / "download_metadata.json"
        self.sources = self._load_config()
        self.metadata = self._load_metadata()
        
    def _load_config(self) -> List[Dict]:
        """Load video sources from config file."""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config.get('sources', [])
    
    def _load_metadata(self) -> Dict:
        """Load or create metadata tracking file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"downloads": {}}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_file_path(self, source: Dict) -> Path:
        """Generate output file path for a source."""
        url = source['url']
        # Remove .zip extension if present - we'll extract it
        if url.endswith('.zip'):
            # Get the filename without .zip
            filename = Path(url).stem
            # If it doesn't have a video extension, add .mp4
            if not any(filename.endswith(ext) for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']):
                filename = f"{source['id']}.mp4"
        else:
            extension = Path(url).suffix
            filename = f"{source['id']}{extension}"
        return self.output_dir / filename
    
    def _calculate_checksum(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum."""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def _is_download_complete(self, source: Dict) -> bool:
        """Check if a video is already downloaded and valid."""
        file_path = self._get_file_path(source)
        
        if not file_path.exists():
            return False
        
        # Check metadata
        source_id = source['id']
        if source_id not in self.metadata['downloads']:
            return False
        
        meta = self.metadata['downloads'][source_id]
        
        # Verify file size matches expected
        actual_size = file_path.stat().st_size
        expected_size = meta.get('size_bytes', 0)
        
        if actual_size != expected_size:
            return False
        
        # If checksum exists, verify it
        if 'checksum' in meta:
            actual_checksum = self._calculate_checksum(file_path)
            if actual_checksum != meta['checksum']:
                print(f"  ⚠️  Checksum mismatch for {source['name']}, will re-download")
                return False
        
        return True
    
    def download_video(self, source: Dict, resume: bool = True) -> bool:
        """
        Download a single video file.
        
        Args:
            source: Video source dictionary
            resume: Whether to resume partial downloads
            
        Returns:
            True if download succeeded, False otherwise
        """
        file_path = self._get_file_path(source)
        url = source['url']
        source_id = source['id']
        is_zip = url.endswith('.zip')
        
        # If it's a zip, download to temp location first
        if is_zip:
            download_path = self.output_dir / f"{source_id}_temp.zip"
        else:
            download_path = file_path
        
        # Check if already downloaded
        if self._is_download_complete(source):
            print(f"✓ {source['name']} already downloaded")
            return True
        
        print(f"⬇ Downloading {source['name']}...")
        print(f"  URL: {url}")
        
        # Determine if we should resume
        existing_size = 0
        headers = {}
        mode = 'wb'
        
        if resume and download_path.exists():
            existing_size = download_path.stat().st_size
            headers['Range'] = f'bytes={existing_size}-'
            mode = 'ab'
            print(f"  Resuming from {existing_size / 1024 / 1024:.1f} MB")
        
        try:
            # Make request
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # Handle resume requests
            if response.status_code == 416:  # Range not satisfiable
                print(f"  ⚠️  Cannot resume, starting fresh")
                existing_size = 0
                response = requests.get(url, stream=True, timeout=30)
                mode = 'wb'
            elif response.status_code not in (200, 206):
                print(f"  ❌ Failed: HTTP {response.status_code}")
                return False
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if response.status_code == 206:  # Partial content
                total_size += existing_size
            
            # Download with progress bar
            with open(download_path, mode) as f, tqdm(
                total=total_size,
                initial=existing_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {source['id']}"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # If it's a zip file, extract it
            if is_zip:
                print(f"  📦 Extracting archive...")
                try:
                    with zipfile.ZipFile(download_path, 'r') as zip_ref:
                        # List contents to find the video file
                        video_files = [f for f in zip_ref.namelist() 
                                     if any(f.endswith(ext) for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm'])]
                        
                        if not video_files:
                            print(f"  ❌ No video file found in archive")
                            download_path.unlink()  # Clean up zip
                            return False
                        
                        if len(video_files) > 1:
                            print(f"  ⚠️  Multiple video files in archive, using first one: {video_files[0]}")
                        
                        # Extract the video file
                        video_file = video_files[0]
                        print(f"  Extracting: {video_file}")
                        zip_ref.extract(video_file, self.output_dir)
                        
                        # Move to expected location
                        extracted_path = self.output_dir / video_file
                        extracted_path.rename(file_path)
                        
                    # Remove the zip file
                    download_path.unlink()
                    print(f"  ✓ Extracted successfully")
                    
                except zipfile.BadZipFile:
                    print(f"  ❌ Invalid zip file")
                    download_path.unlink()
                    return False
            
            # Calculate checksum
            print(f"  🔍 Verifying download...")
            checksum = self._calculate_checksum(file_path)
            
            # Update metadata (only essential info, no redundant MB field)
            self.metadata['downloads'][source_id] = {
                'name': source['name'],
                'url': url,
                'size_bytes': file_path.stat().st_size,
                'checksum': checksum,
                'categories': source.get('categories', []),
                'license': source.get('license', 'unknown'),
                'file_path': str(file_path)
            }
            self._save_metadata()
            
            print(f"  ✅ Success! ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Network error: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def fetch_all(self, categories: Optional[List[str]] = None, 
                  ids: Optional[List[str]] = None,
                  continue_on_error: bool = True) -> Dict[str, int]:
        """
        Download all configured videos (or filtered subset).
        
        Args:
            categories: Only download videos with these categories
            ids: Only download videos with these IDs
            continue_on_error: Continue downloading even if some fail
            
        Returns:
            Dictionary with counts: {'success': N, 'failed': M, 'skipped': K}
        """
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Filter sources
        sources_to_fetch = self.sources
        
        if ids:
            sources_to_fetch = [s for s in sources_to_fetch if s['id'] in ids]
        
        if categories:
            sources_to_fetch = [
                s for s in sources_to_fetch 
                if any(cat in s.get('categories', []) for cat in categories)
            ]
        
        if not sources_to_fetch:
            print("No videos match the specified filters.")
            return results
        
        print(f"\n📦 Fetching {len(sources_to_fetch)} video(s)...\n")
        
        for i, source in enumerate(sources_to_fetch, 1):
            print(f"\n[{i}/{len(sources_to_fetch)}] {source['name']}")
            print("─" * 60)
            
            if self._is_download_complete(source):
                print(f"✓ Already downloaded")
                results['skipped'] += 1
                continue
            
            success = self.download_video(source)
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
                if not continue_on_error:
                    print("\n❌ Stopping due to error (use --continue-on-error to override)")
                    break
        
        return results
    
    def list_sources(self, categories: Optional[List[str]] = None):
        """List all configured video sources."""
        sources = self.sources
        
        if categories:
            sources = [
                s for s in sources 
                if any(cat in s.get('categories', []) for cat in categories)
            ]
        
        print(f"\n📹 Available video sources ({len(sources)}):\n")
        
        for source in sources:
            status = "✓" if self._is_download_complete(source) else "○"
            print(f"{status} {source['id']}")
            print(f"   Name: {source['name']}")
            print(f"   Categories: {', '.join(source.get('categories', []))}")
            print(f"   License: {source.get('license', 'unknown')}")
            
            # Show actual file info if downloaded
            if self._is_download_complete(source):
                meta = self.metadata['downloads'][source['id']]
                print(f"   Downloaded: {meta['size_bytes'] / 1024 / 1024:.1f} MB")
            print()
    
    def show_info(self, video_id: str):
        """Show detailed information about a specific video."""
        source = next((s for s in self.sources if s['id'] == video_id), None)
        
        if not source:
            print(f"❌ Video '{video_id}' not found in config")
            return
        
        print(f"\n📹 {source['name']}")
        print("=" * 60)
        print(f"ID: {source['id']}")
        print(f"URL: {source['url']}")
        print(f"Categories: {', '.join(source.get('categories', []))}")
        print(f"License: {source.get('license', 'unknown')}")
        
        # Check download status
        if self._is_download_complete(source):
            file_path = self._get_file_path(source)
            meta = self.metadata['downloads'][video_id]
            print(f"\n✓ Downloaded")
            print(f"  Path: {file_path}")
            print(f"  Size: {meta['size_bytes'] / 1024 / 1024:.1f} MB ({meta['size_bytes']:,} bytes)")
            print(f"  Checksum (SHA256): {meta['checksum']}")
        else:
            print(f"\n○ Not downloaded")
    
    def get_categories(self) -> List[str]:
        """Get all unique categories from sources."""
        categories = set()
        for source in self.sources:
            categories.update(source.get('categories', []))
        return sorted(categories)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Fetch video files for SVT-AV1 optimization research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available videos
  python fetch_videos.py --list
  
  # Download all videos
  python fetch_videos.py --all
  
  # Download specific videos by ID
  python fetch_videos.py --ids bbb_trailer sintel_trailer
  
  # Download videos by category
  python fetch_videos.py --category 3d_animation
  
  # Show info about a specific video
  python fetch_videos.py --info bbb_trailer
  
  # List all categories
  python fetch_videos.py --list-categories
        """
    )
    
    parser.add_argument('--config', default='config/video_sources.json',
                       help='Path to video sources config (default: config/video_sources.json)')
    parser.add_argument('--output-dir', default='data/raw_videos',
                       help='Output directory for videos (default: data/raw_videos)')
    
    # Actions
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--list', action='store_true',
                             help='List all available videos')
    action_group.add_argument('--list-categories', action='store_true',
                             help='List all available categories')
    action_group.add_argument('--info', metavar='ID',
                             help='Show detailed info about a video')
    action_group.add_argument('--all', action='store_true',
                             help='Download all videos')
    
    # Filters
    parser.add_argument('--category', action='append', dest='categories',
                       help='Filter by category (can be used multiple times)')
    parser.add_argument('--ids', nargs='+',
                       help='Download specific videos by ID')
    
    # Options
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume partial downloads')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue downloading even if some videos fail')
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = VideoFetcher(args.config, args.output_dir)
    
    # Handle actions
    if args.list:
        fetcher.list_sources(categories=args.categories)
    elif args.list_categories:
        categories = fetcher.get_categories()
        print(f"\n📂 Available categories ({len(categories)}):\n")
        for cat in categories:
            count = sum(1 for s in fetcher.sources if cat in s.get('categories', []))
            print(f"  • {cat} ({count} videos)")
        print()
    elif args.info:
        fetcher.show_info(args.info)
    elif args.all or args.ids or args.categories:
        results = fetcher.fetch_all(
            categories=args.categories,
            ids=args.ids,
            continue_on_error=args.continue_on_error
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Download Summary")
        print("=" * 60)
        print(f"✅ Successful: {results['success']}")
        print(f"⏭️  Skipped (already downloaded): {results['skipped']}")
        print(f"❌ Failed: {results['failed']}")
        print()
        
        if results['failed'] > 0:
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
