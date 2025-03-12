"""Implements a centralized logging system. Configures and provides a logger
that writes to both files and console. Logs system events, errors and results
with timestamps, facilitating debugging and execution tracking. Follows good
logging practices with severity levels."""

import logging
from datetime import datetime
from pathlib import Path

from config.settings import CONFIG


def setup_logger():
    """Configure and return a logger for the application."""
    logs_dir = Path(CONFIG["output"]["logs_dir"])
    logs_dir.mkdir(exist_ok=True, parents=True)

    log_file = (
        logs_dir / f"fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Configure logging
    logger = logging.getLogger("fraud_detection")
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
