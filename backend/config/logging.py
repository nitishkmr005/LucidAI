import sys
from pathlib import Path

from loguru import logger

from config.settings import get_settings

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_LOGS_DIR = _BACKEND_DIR / "logs"


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

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
