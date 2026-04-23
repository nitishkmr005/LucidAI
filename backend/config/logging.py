import sys

from loguru import logger

from config.settings import get_settings


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()

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
