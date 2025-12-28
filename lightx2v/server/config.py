import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_queue_size: int = 10

    task_timeout: int = 300
    task_history_limit: int = 1000

    http_timeout: int = 30
    http_max_retries: int = 3

    cache_dir: str = str(Path(__file__).parent.parent / "server_cache")
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

    # 缓存清理配置
    enable_cache_cleaner: bool = False  # 是否启用缓存清理
    cache_retention_hours: float = 1.0  # 缓存文件保留时间（小时）
    cache_check_interval: int = 300  # 检查间隔（秒）

    @classmethod
    def from_env(cls) -> "ServerConfig":
        config = cls()

        if env_host := os.environ.get("LIGHTX2V_HOST"):
            config.host = env_host

        if env_port := os.environ.get("LIGHTX2V_PORT"):
            try:
                config.port = int(env_port)
            except ValueError:
                logger.warning(f"Invalid port in environment: {env_port}")

        if env_queue_size := os.environ.get("LIGHTX2V_MAX_QUEUE_SIZE"):
            try:
                config.max_queue_size = int(env_queue_size)
            except ValueError:
                logger.warning(f"Invalid max queue size: {env_queue_size}")

        # MASTER_ADDR is now managed by torchrun, no need to set manually

        if env_cache_dir := os.environ.get("LIGHTX2V_CACHE_DIR"):
            config.cache_dir = env_cache_dir

        # 缓存清理配置
        if env_enable_cleaner := os.environ.get("LIGHTX2V_ENABLE_CACHE_CLEANER"):
            config.enable_cache_cleaner = env_enable_cleaner.lower() in ("true", "1", "yes")

        if env_retention := os.environ.get("LIGHTX2V_CACHE_RETENTION_HOURS"):
            try:
                config.cache_retention_hours = float(env_retention)
            except ValueError:
                logger.warning(f"Invalid cache retention hours: {env_retention}")

        if env_interval := os.environ.get("LIGHTX2V_CACHE_CHECK_INTERVAL"):
            try:
                config.cache_check_interval = int(env_interval)
            except ValueError:
                logger.warning(f"Invalid cache check interval: {env_interval}")

        return config

    def validate(self) -> bool:
        valid = True

        if self.max_queue_size <= 0:
            logger.error("max_queue_size must be positive")
            valid = False

        if self.task_timeout <= 0:
            logger.error("task_timeout must be positive")
            valid = False

        return valid


server_config = ServerConfig.from_env()
