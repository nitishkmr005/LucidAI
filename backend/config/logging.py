import sys
from pathlib import Path

from loguru import logger

from config.settings import get_settings

_LOGS_DIR = Path("logs")


def _prune_old_logs(keep: int = 5) -> None:
    """Keep only the latest `keep` JSON log files."""
    files = sorted(_LOGS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink(missing_ok=True)


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()

    # Colorful terminal output
    logger.add(
        sys.stdout,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # Rotating JSON file sink — keep latest 5
    _LOGS_DIR.mkdir(exist_ok=True)
    logger.add(
        str(_LOGS_DIR / "neurotalk_{time:YYYY-MM-DD_HH-mm-ss}.json"),
        level="DEBUG",
        serialize=True,      # writes structured JSON per line
        rotation="10 MB",    # new file at 10 MB
        retention=5,         # keep 5 files max
        compression=None,
    )

    _prune_old_logs(keep=5)
