"""Minimal logging helper used across the codebase."""

import logging
import sys

_LOGGER_NAME = "bonescanad"
_CONFIGURED = False


def get_logger(level: int = logging.INFO) -> logging.Logger:
    """Return the shared logger, configuring handlers on first use."""
    global _CONFIGURED

    logger = logging.getLogger(_LOGGER_NAME)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _CONFIGURED = True

    logger.setLevel(level)
    return logger


def configure_log_file(path) -> None:
    """Append a file handler so a run can be inspected after the fact."""
    logger = get_logger()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(path):
            return

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
