"""Tests for logging setup."""
import logging
import tempfile
from pathlib import Path
from evolveclaw_ramsey.utils.logging import setup_logging, get_logger, LOGGER_NAME


def test_setup_logging_returns_logger():
    logger = setup_logging(level="DEBUG")
    assert logger.name == LOGGER_NAME
    assert logger.level == logging.DEBUG


def test_setup_logging_with_file_handler():
    run_dir = tempfile.mkdtemp()
    logger = setup_logging(level="INFO", run_dir=run_dir)
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    log_path = Path(run_dir) / "run.log"
    assert log_path.exists()


def test_setup_logging_clears_previous_handlers():
    setup_logging(level="INFO")
    setup_logging(level="DEBUG")
    logger = logging.getLogger(LOGGER_NAME)
    # Should have only 1 console handler, not accumulated
    assert len(logger.handlers) == 1


def test_get_logger_returns_named_logger():
    logger = get_logger()
    assert logger.name == LOGGER_NAME
