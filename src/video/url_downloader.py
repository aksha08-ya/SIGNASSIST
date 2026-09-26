"""Video URL Ingestion & Platform Detection for SIGNASSIST (Phase 14).

Provides:
- URLVideoMetadata dataclass
- URLVideoDownloader class for URL validation, platform detection,
  yt-dlp video retrieval, temporary file management, and graceful fallback.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Optional, Union
from urllib.parse import urlparse


import yt_dlp

from src.utils.exceptions import URLExtractionError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


@dataclass
class URLVideoMetadata:
    """Structured container holding retrieved web video metadata."""

    original_url: str
    platform_name: str
    video_title: str
    retrieved_file_path: str
    duration_sec: Optional[float]
    is_temporary: bool = True
    temp_dir: Optional[str] = None

    def cleanup(self) -> None:
        """Cleans up the underlying temporary directory if owned."""
        if self.is_temporary and self.temp_dir:
            try:
                p = Path(self.temp_dir)
                if p.exists() and p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    logger.info(f"Cleaned up temporary download directory '{self.temp_dir}'")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary directory '{self.temp_dir}': {e}")


class URLVideoDownloader:
    """Validates, detects, and downloads web videos using yt-dlp."""

    def __init__(self, timeout_sec: int = 15) -> None:
        self.timeout_sec = timeout_sec

    def validate_url(self, url: str) -> str:
        """Validates URL structure, scheme (HTTP/HTTPS), and host.

        Raises:
            URLExtractionError: If URL is empty, malformed, or uses an unsupported scheme/path.
        """
        if not url or not isinstance(url, str):
            raise URLExtractionError(
                "Provided URL is empty or invalid. Please supply a valid HTTP or HTTPS video URL."
            )

        clean_url = url.strip()
        try:
            parsed = urlparse(clean_url)
        except Exception as e:
            raise URLExtractionError(
                f"Malformed URL structure: '{clean_url}'. "
                "Please verify the web link or download the video file manually to use Phase 13 local upload."
            ) from e

        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            raise URLExtractionError(
                f"Unsupported URL scheme '{scheme}://'. Only 'http://' and 'https://' are supported. "
                "Please download the video file manually and process it using Phase 13 local upload."
            )

        if not parsed.netloc:
            raise URLExtractionError(
                f"URL lacks a valid network host: '{clean_url}'. "
                "Please check the URL or use Phase 13 local file upload."
            )

        return clean_url

    def detect_platform(self, url: str) -> str:
        """Determines target platform identifier deterministically without downloading."""
        valid_url = self.validate_url(url)
        parsed = urlparse(valid_url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()

        # Strip standard subdomains
        for prefix in ("www.", "m.", "mobile.", "web."):
            if host.startswith(prefix):
                host = host[len(prefix) :]

        # 1. YouTube
        if host in ("youtube.com", "youtu.be", "youtube-nocookie.com") or "youtube" in host:
            if host == "youtu.be" or "/watch" in path or "/shorts" in path or "/embed" in path or "/v/" in path:
                return "youtube"
            return "youtube"

        # 2. Instagram
        if host in ("instagram.com", "instagr.am") or "instagram" in host:
            if "/reel" in path or "/reels" in path or "/p/" in path or "/tv/" in path:
                return "instagram"
            return "instagram"

        # 3. Facebook
        if host in ("facebook.com", "fb.watch", "fb.com") or "facebook" in host:
            return "facebook"

        # 4. Google Drive
        if host in ("drive.google.com", "docs.google.com"):
            return "gdrive"

        # 5. Direct Video File URL
        ext = Path(path).suffix.lower()
        if ext in SUPPORTED_VIDEO_EXTENSIONS:
            return "direct"

        return "unknown"

    def download_video(
        self,
        url: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> URLVideoMetadata:
        """Downloads a web video to a local directory using yt-dlp."""
        valid_url = self.validate_url(url)
        platform_name = self.detect_platform(valid_url)

        if output_dir is not None:
            target_dir = Path(output_dir).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            is_temp = False
            temp_dir_str = None
        else:
            temp_dir_str = tempfile.mkdtemp(prefix="signassist_url_")
            target_dir = Path(temp_dir_str)
            is_temp = True

        out_template = str(target_dir / "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": self.timeout_sec,
            "max_filesize": 150 * 1024 * 1024,  # Cap max video size 150MB
            "merge_output_format": "mp4",
        }

        try:
            logger.info(f"Initiating yt-dlp video retrieval for URL '{valid_url}' ({platform_name})...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(valid_url, download=True)
        except Exception as e:
            if is_temp and temp_dir_str:
                shutil.rmtree(temp_dir_str, ignore_errors=True)

            err_msg = str(e).lower()
            if "private" in err_msg or "login" in err_msg or "auth" in err_msg:
                diag = f"The requested video on {platform_name.upper()} is private or requires authentication."
            elif "copyright" in err_msg or "unavailable" in err_msg or "not found" in err_msg or "404" in err_msg:
                diag = f"The requested video on {platform_name.upper()} is unavailable, restricted, or deleted."
            elif "timeout" in err_msg or "timed out" in err_msg:
                diag = f"Connection timed out while retrieving video from {platform_name.upper()}."
            else:
                diag = f"Failed to retrieve video content from {platform_name.upper()}."

            raise URLExtractionError(
                f"{diag} Please download the video file manually and process it using the Phase 13 local upload feature.",
                details={"url": valid_url, "platform": platform_name, "error": str(e)},
            ) from e

        # Locate downloaded video file in target_dir
        downloaded_files = [
            p for p in target_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]

        if not downloaded_files:
            # Fallback check any non-empty file
            downloaded_files = [p for p in target_dir.glob("*") if p.is_file() and p.stat().st_size > 0]

        if not downloaded_files:
            if is_temp and temp_dir_str:
                shutil.rmtree(temp_dir_str, ignore_errors=True)
            raise URLExtractionError(
                f"Video retrieval completed but no valid video file was saved for platform {platform_name.upper()}. "
                "Please download the file manually and process it using Phase 13 local upload."
            )

        downloaded_file = downloaded_files[0]
        if downloaded_file.stat().st_size == 0:
            if is_temp and temp_dir_str:
                shutil.rmtree(temp_dir_str, ignore_errors=True)
            raise URLExtractionError(
                f"Retrieved video file is empty (0 bytes) from {platform_name.upper()}. "
                "Please use Phase 13 local video upload."
            )

        title = info_dict.get("title", downloaded_file.stem) if isinstance(info_dict, dict) else downloaded_file.stem
        duration = info_dict.get("duration") if isinstance(info_dict, dict) else None
        if duration is not None:
            duration = float(duration)

        return URLVideoMetadata(
            original_url=valid_url,
            platform_name=platform_name,
            video_title=str(title),
            retrieved_file_path=str(downloaded_file.resolve()),
            duration_sec=duration,
            is_temporary=is_temp,
            temp_dir=temp_dir_str,
        )

    def process_url(
        self,
        url: str,
        processor: Optional[Union["VideoProcessor", Any]] = None,
        *,
        output_dir: Optional[Union[str, Path]] = None,
        manual_signer_id: Optional[int] = None,
        enable_tts: bool = False,
        chunk_size: int = 60,
    ) -> Any:
        """Downloads a web video and processes it through Phase 13 VideoProcessor.

        Args:
            url: Web video URL (YouTube, Instagram, Facebook, Google Drive, or Direct link).
            processor: VideoProcessor instance (creates default instance if None).
            output_dir: Optional directory for downloaded video. If None, uses a temporary directory.
            manual_signer_id: Optional manual signer track ID override.
            enable_tts: Whether to synthesize speech for reconstructed transcript.
            chunk_size: Frame chunk size for video processing.

        Returns:
            VideoProcessingResult containing video metadata, quality report, timeline, and transcript.
        """
        from src.video.processor import VideoProcessor

        if processor is None:
            processor = VideoProcessor()

        metadata = self.download_video(url, output_dir=output_dir)

        try:
            result = processor.process_file(
                file_path=metadata.retrieved_file_path,
                manual_signer_id=manual_signer_id,
                enable_tts=enable_tts,
                chunk_size=chunk_size,
            )

            result.additional_metadata["source_type"] = "url"
            result.additional_metadata["original_url"] = metadata.original_url
            result.additional_metadata["platform_name"] = metadata.platform_name
            result.additional_metadata["video_title"] = metadata.video_title
            result.additional_metadata["is_temporary"] = metadata.is_temporary

            return result
        finally:
            metadata.cleanup()
