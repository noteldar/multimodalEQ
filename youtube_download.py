#!/usr/bin/env python3
"""
YouTube Video Downloader

This script downloads YouTube videos to local MP4 files using yt-dlp.
Features:
- Download single videos or playlists
- Choose video quality/resolution
- Extract audio only
- Custom output directory and filename
- Progress tracking
- Error handling and logging

Requirements:
- yt-dlp
- ffmpeg (for format conversion)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List
import re

try:
    import yt_dlp
except ImportError:
    print("yt-dlp not installed. Install with: pip install yt-dlp")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("youtube_download.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """
    YouTube video downloader using yt-dlp.
    """

    def __init__(self, output_dir: str = "downloads", quality: str = "best"):
        """
        Initialize the YouTube downloader.

        Args:
            output_dir: Directory to save downloaded videos
            quality: Video quality preference ('best', 'worst', specific resolution)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality = quality

        # Default yt-dlp options
        self.ydl_opts = {
            "outtmpl": str(self.output_dir / "%(title)s.%(ext)s"),
            "format": self._get_format_selector(),
            "writeinfojson": False,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "ignoreerrors": True,
            "no_warnings": False,
        }

        logger.info(f"YouTube downloader initialized - Output: {self.output_dir}")

    def _get_format_selector(self) -> str:
        """Get format selector based on quality preference."""
        if self.quality == "best":
            return "best[ext=mp4]/best"
        elif self.quality == "worst":
            return "worst[ext=mp4]/worst"
        elif self.quality.endswith("p"):  # e.g., "720p", "1080p"
            height = self.quality[:-1]
            return f"best[height<={height}][ext=mp4]/best[height<={height}]/best[ext=mp4]/best"
        else:
            return "best[ext=mp4]/best"

    def download_video(self, url: str, custom_filename: Optional[str] = None) -> Dict:
        """
        Download a single YouTube video.

        Args:
            url: YouTube video URL
            custom_filename: Optional custom filename (without extension)

        Returns:
            Dictionary with download results
        """
        logger.info(f"Starting download: {url}")

        # Create custom options for this download
        opts = self.ydl_opts.copy()

        if custom_filename:
            # Sanitize filename
            safe_filename = self._sanitize_filename(custom_filename)
            opts["outtmpl"] = str(self.output_dir / f"{safe_filename}.%(ext)s")

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)

                result = {
                    "url": url,
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "view_count": info.get("view_count", 0),
                    "upload_date": info.get("upload_date", "Unknown"),
                    "success": False,
                    "filepath": None,
                    "error": None,
                }

                logger.info(
                    f"Video info - Title: {result['title']}, Duration: {result['duration']}s"
                )

                # Download the video
                ydl.download([url])

                # Find the downloaded file
                expected_filename = ydl.prepare_filename(info)
                if os.path.exists(expected_filename):
                    result["filepath"] = expected_filename
                    result["success"] = True
                    logger.info(f"Download successful: {expected_filename}")
                else:
                    # Try to find the file with different extension
                    base_path = Path(expected_filename).with_suffix("")
                    for ext in [".mp4", ".webm", ".mkv", ".avi"]:
                        potential_file = str(base_path) + ext
                        if os.path.exists(potential_file):
                            result["filepath"] = potential_file
                            result["success"] = True
                            logger.info(f"Download successful: {potential_file}")
                            break

                if not result["success"]:
                    result["error"] = "Downloaded file not found"
                    logger.error(f"Download failed: {result['error']}")

                return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Download error: {error_msg}")
            return {"url": url, "success": False, "error": error_msg, "filepath": None}

    def download_playlist(
        self, playlist_url: str, max_videos: Optional[int] = None
    ) -> List[Dict]:
        """
        Download a YouTube playlist.

        Args:
            playlist_url: YouTube playlist URL
            max_videos: Maximum number of videos to download (None for all)

        Returns:
            List of download results for each video
        """
        logger.info(f"Starting playlist download: {playlist_url}")

        opts = self.ydl_opts.copy()
        opts["outtmpl"] = str(
            self.output_dir / "%(playlist_index)s - %(title)s.%(ext)s"
        )

        if max_videos:
            opts["playlistend"] = max_videos

        results = []

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Get playlist info
                playlist_info = ydl.extract_info(playlist_url, download=False)
                playlist_title = playlist_info.get("title", "Unknown Playlist")
                entries = playlist_info.get("entries", [])

                logger.info(f"Playlist: {playlist_title}, Videos: {len(entries)}")

                # Download each video
                for i, entry in enumerate(
                    entries[:max_videos] if max_videos else entries
                ):
                    if entry is None:
                        continue

                    video_url = entry.get("webpage_url") or entry.get("url")
                    if not video_url:
                        continue

                    logger.info(
                        f"Downloading video {i+1}/{len(entries)}: {entry.get('title', 'Unknown')}"
                    )

                    try:
                        ydl.download([video_url])
                        results.append(
                            {
                                "url": video_url,
                                "title": entry.get("title", "Unknown"),
                                "success": True,
                                "error": None,
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to download video {i+1}: {str(e)}")
                        results.append(
                            {
                                "url": video_url,
                                "title": entry.get("title", "Unknown"),
                                "success": False,
                                "error": str(e),
                            }
                        )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Playlist download error: {error_msg}")
            results.append({"url": playlist_url, "success": False, "error": error_msg})

        return results

    def download_audio_only(
        self, url: str, custom_filename: Optional[str] = None
    ) -> Dict:
        """
        Download only the audio from a YouTube video.

        Args:
            url: YouTube video URL
            custom_filename: Optional custom filename (without extension)

        Returns:
            Dictionary with download results
        """
        logger.info(f"Starting audio-only download: {url}")

        opts = self.ydl_opts.copy()
        opts.update(
            {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }
        )

        if custom_filename:
            safe_filename = self._sanitize_filename(custom_filename)
            opts["outtmpl"] = str(self.output_dir / f"{safe_filename}.%(ext)s")

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

                result = {
                    "url": url,
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "success": False,
                    "filepath": None,
                    "error": None,
                }

                ydl.download([url])

                # Find the downloaded audio file
                expected_filename = ydl.prepare_filename(info)
                base_path = Path(expected_filename).with_suffix(".mp3")

                if base_path.exists():
                    result["filepath"] = str(base_path)
                    result["success"] = True
                    logger.info(f"Audio download successful: {base_path}")
                else:
                    result["error"] = "Downloaded audio file not found"
                    logger.error(f"Audio download failed: {result['error']}")

                return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Audio download error: {error_msg}")
            return {"url": url, "success": False, "error": error_msg, "filepath": None}

    def get_video_info(self, url: str) -> Dict:
        """
        Get information about a YouTube video without downloading.

        Args:
            url: YouTube video URL

        Returns:
            Dictionary with video information
        """
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)

                return {
                    "title": info.get("title", "Unknown"),
                    "description": info.get("description", ""),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "view_count": info.get("view_count", 0),
                    "like_count": info.get("like_count", 0),
                    "upload_date": info.get("upload_date", "Unknown"),
                    "tags": info.get("tags", []),
                    "thumbnail": info.get("thumbnail", ""),
                    "formats": [
                        {
                            "format_id": f.get("format_id", ""),
                            "ext": f.get("ext", ""),
                            "resolution": f.get("resolution", ""),
                            "filesize": f.get("filesize", 0),
                        }
                        for f in info.get("formats", [])
                    ],
                }
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {"error": str(e)}

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        # Remove multiple spaces and trim
        filename = re.sub(r"\s+", " ", filename).strip()
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        return filename


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Download YouTube videos to MP4 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single video
  python youtube_download.py "https://www.youtube.com/watch?v=VIDEO_ID"
  
  # Download with custom quality
  python youtube_download.py "VIDEO_URL" --quality 720p
  
  # Download playlist
  python youtube_download.py "PLAYLIST_URL" --playlist --max-videos 5
  
  # Download audio only
  python youtube_download.py "VIDEO_URL" --audio-only
  
  # Custom output directory and filename
  python youtube_download.py "VIDEO_URL" --output-dir "./videos" --filename "my_video"
        """,
    )

    parser.add_argument("url", help="YouTube video or playlist URL")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="downloads",
        help="Output directory (default: downloads)",
    )
    parser.add_argument(
        "--quality",
        "-q",
        default="best",
        choices=["best", "worst", "720p", "1080p", "480p"],
        help="Video quality (default: best)",
    )
    parser.add_argument("--filename", "-f", help="Custom filename (without extension)")
    parser.add_argument(
        "--playlist", "-p", action="store_true", help="Download as playlist"
    )
    parser.add_argument(
        "--max-videos", "-m", type=int, help="Maximum videos to download from playlist"
    )
    parser.add_argument(
        "--audio-only", "-a", action="store_true", help="Download audio only (MP3)"
    )
    parser.add_argument(
        "--info", "-i", action="store_true", help="Show video info without downloading"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize downloader
    downloader = YouTubeDownloader(output_dir=args.output_dir, quality=args.quality)

    try:
        if args.info:
            # Just show video information
            info = downloader.get_video_info(args.url)
            if "error" in info:
                print(f"❌ Error getting video info: {info['error']}")
                return

            print(f"\n📹 Video Information:")
            print(f"Title: {info['title']}")
            print(f"Uploader: {info['uploader']}")
            print(f"Duration: {info['duration']//60}:{info['duration']%60:02d}")
            print(f"Views: {info['view_count']:,}")
            print(f"Upload Date: {info['upload_date']}")
            print(f"Available Formats: {len(info['formats'])}")

        elif args.playlist:
            # Download playlist
            results = downloader.download_playlist(args.url, args.max_videos)

            successful = sum(1 for r in results if r["success"])
            failed = len(results) - successful

            print(f"\n📊 Playlist Download Complete!")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print(f"📁 Saved to: {downloader.output_dir}")

        elif args.audio_only:
            # Download audio only
            result = downloader.download_audio_only(args.url, args.filename)

            if result["success"]:
                print(f"\n🎵 Audio download successful!")
                print(f"Title: {result['title']}")
                print(f"File: {result['filepath']}")
            else:
                print(f"❌ Audio download failed: {result['error']}")

        else:
            # Download single video
            result = downloader.download_video(args.url, args.filename)

            if result["success"]:
                print(f"\n✅ Download successful!")
                print(f"Title: {result['title']}")
                print(f"Duration: {result['duration']//60}:{result['duration']%60:02d}")
                print(f"Uploader: {result['uploader']}")
                print(f"File: {result['filepath']}")
            else:
                print(f"❌ Download failed: {result['error']}")

    except KeyboardInterrupt:
        print("\n⏹️  Download interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")


# Utility functions for easy importing
def download_youtube_video(
    url: str,
    output_dir: str = "downloads",
    quality: str = "best",
    filename: Optional[str] = None,
) -> Dict:
    """
    Simple function to download a YouTube video.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        quality: Video quality preference
        filename: Custom filename (optional)

    Returns:
        Dictionary with download results
    """
    downloader = YouTubeDownloader(output_dir=output_dir, quality=quality)
    return downloader.download_video(url, filename)


def download_youtube_audio(
    url: str, output_dir: str = "downloads", filename: Optional[str] = None
) -> Dict:
    """
    Simple function to download audio from a YouTube video.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio
        filename: Custom filename (optional)

    Returns:
        Dictionary with download results
    """
    downloader = YouTubeDownloader(output_dir=output_dir)
    return downloader.download_audio_only(url, filename)


if __name__ == "__main__":
    main()
