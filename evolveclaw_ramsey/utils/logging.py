"""Logging configuration for evolveclaw_ramsey."""
from __future__ import annotations
import logging
from pathlib import Path

LOGGER_NAME = "evolveclaw_ramsey"

def setup_logging(level: str = "INFO", run_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if run_dir:
        log_path = Path(run_dir) / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger

def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
