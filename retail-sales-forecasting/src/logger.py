"""
logger.py

Centralised logging setup.  Import `logger` from this module to get
a pre-configured Logger that writes to both the console and a log file.

Usage
-----
    from src.logger import logger
    logger.info("Training started")
    logger.error("Something went wrong", exc_info=True)
"""

import io
import logging
import sys
from pathlib import Path

from src.config import LOG_FILE, LOG_LEVEL

# ─── Ensure log directory exists ──────────────────────────────────────────────
LOG_FILE = Path(LOG_FILE)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ─── Formatter ────────────────────────────────────────────────────────────────
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

# ─── Handlers ─────────────────────────────────────────────────────────────────
_console_handler = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_console_handler.setFormatter(formatter)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(formatter)

# ─── Root logger for this project ─────────────────────────────────────────────
logger = logging.getLogger("retail_forecast")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger.addHandler(_console_handler)
logger.addHandler(_file_handler)
logger.propagate = False  # don't double-log to root logger
