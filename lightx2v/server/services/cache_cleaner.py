import threading
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger


class CacheCleaner:
    def __init__(
        self,
        cache_dir: Path,
        retention_hours: float = 1.0,  # 文件保留时间（小时）
        check_interval: int = 300,  # 检查间隔（秒）
    ):
        self.cache_dir = cache_dir
        self.input_image_dir = cache_dir / "inputs" / "imgs"
        self.input_audio_dir = cache_dir / "inputs" / "audios"
        self.output_video_dir = cache_dir / "outputs"

        self.retention_hours = retention_hours
        self.check_interval = check_interval

        self._cleanup_thread = None
        self._stop_event = threading.Event()

    def _get_file_age_seconds(self, file_path: Path) -> float:
        try:
            return time.time() - file_path.stat().st_mtime
        except Exception as e:
            logger.warning(f"Failed to get file age for {file_path}: {e}")
            return 0

    def _get_files_sorted_by_age(self, directory: Path) -> List[Path]:
        try:
            if not directory.exists():
                return []

            files = []
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    files.append(file_path)

            files.sort(key=lambda f: f.stat().st_mtime)
            return files
        except Exception as e:
            logger.error(f"Failed to list files in {directory}: {e}")
            return []

    def _delete_file_safe(self, file_path: Path) -> tuple[bool, int]:
        try:
            file_size = file_path.stat().st_size
            file_path.unlink()
            logger.info(f"Deleted: {file_path.name} ({file_size / 1024 / 1024:.2f} MB)")
            return True, file_size
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False, 0

    def _cleanup_directory(self, directory: Path, dir_name: str) -> None:
        if not directory.exists():
            return

        retention_seconds = self.retention_hours * 3600
        files = self._get_files_sorted_by_age(directory)
        
        deleted_count = 0
        total_size_freed = 0

        for file_path in files:
            age_seconds = self._get_file_age_seconds(file_path)
            if age_seconds > retention_seconds:
                success, file_size = self._delete_file_safe(file_path)
                if success:
                    deleted_count += 1
                    total_size_freed += file_size

        if deleted_count > 0:
            logger.info(
                f"{dir_name} cleanup: deleted {deleted_count} files, "
                f"freed {total_size_freed / 1024 / 1024:.2f} MB"
            )

    def _cleanup_all(self) -> None:
        self._cleanup_directory(self.input_image_dir, "Inputs/images")
        self._cleanup_directory(self.input_audio_dir, "Inputs/audios")
        self._cleanup_directory(self.output_video_dir, "Outputs")

    def _cleanup_loop(self) -> None:
        logger.info("Cache cleaner started in background thread")

        while not self._stop_event.is_set():
            try:
                self._cleanup_all()
                self._stop_event.wait(timeout=self.check_interval)

            except Exception as e:
                logger.exception(f"Error in cleanup loop: {e}")
                time.sleep(60)

        logger.info("Cache cleaner stopped")

    def start(self) -> None:
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            logger.warning("Cache cleaner is already running")
            return

        self._stop_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CacheCleaner"
        )
        self._cleanup_thread.start()
        logger.info(
            f"Cache cleaner started: retention={self.retention_hours}h, "
            f"check_interval={self.check_interval}s"
        )

    def stop(self) -> None:
        if self._cleanup_thread is None:
            return

        logger.info("Stopping cache cleaner")
        self._stop_event.set()

        self._cleanup_thread.join(timeout=10)
        if self._cleanup_thread.is_alive():
            logger.warning("Cache cleaner thread did not stop gracefully")

        self._cleanup_thread = None
        logger.info("Cache cleaner stopped")

    def cleanup_now(self) -> None:
        logger.info("Running manual cleanup")
        self._cleanup_all()
        logger.info("Manual cleanup completed")

    def get_stats(self) -> dict:
        def get_dir_size(directory: Path) -> float:
            total_size = 0
            try:
                if directory.exists():
                    for file_path in directory.glob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
            except Exception as e:
                logger.warning(f"Failed to calculate size for {directory}: {e}")
            return total_size / (1024**3)

        return {
            "cache_dir": str(self.cache_dir),
            "inputs": {
                "image_dir": str(self.input_image_dir),
                "audio_dir": str(self.input_audio_dir),
                "image_count": len(list(self.input_image_dir.glob("*"))) if self.input_image_dir.exists() else 0,
                "audio_count": len(list(self.input_audio_dir.glob("*"))) if self.input_audio_dir.exists() else 0,
                "total_size_gb": get_dir_size(self.input_image_dir) + get_dir_size(self.input_audio_dir),
            },
            "outputs": {
                "video_dir": str(self.output_video_dir),
                "video_count": len(list(self.output_video_dir.glob("*"))) if self.output_video_dir.exists() else 0,
                "total_size_gb": get_dir_size(self.output_video_dir),
            },
            "config": {
                "retention_hours": self.retention_hours,
                "check_interval": self.check_interval,
            },
        }
