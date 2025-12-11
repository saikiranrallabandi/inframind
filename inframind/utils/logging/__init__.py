"""
Production-Grade Logging System for InfraMind.

Enhanced logging with structured JSON output, correlation tracking,
performance monitoring, and training metrics.

Usage:
    from inframind.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting training", epoch=1, loss=0.5)

    # With timing
    with logger.timer("training_step"):
        # training code
        pass

    # With correlation (for distributed training)
    with logger.correlation(run_id="run-123"):
        logger.info("Training started")
"""

import logging
import os
from typing import Optional

from .core import InfraMindLogger, set_training_context
from .formatters import JSONFormatter, HumanReadableFormatter, CompactFormatter
from .correlation import (
    set_correlation_id,
    get_correlation_id,
    set_run_id,
    get_run_id,
    set_experiment_id,
    get_experiment_id,
    correlation_context,
    generate_correlation_id,
)
from .metrics import get_global_metrics


# Global configuration
_config = None
_configured = False


def _get_environment() -> str:
    """Get current environment from env var."""
    return os.getenv('INFRAMIND_ENV', 'development')


def _configure_root_logger():
    """Configure root logger based on environment."""
    global _configured

    if _configured:
        return
    _configured = True

    env = _get_environment()

    # Get root logger
    root_logger = logging.getLogger()

    # Set level based on environment
    if env == 'production':
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Use JSON formatter in production, human-readable in development
    if env == 'production':
        formatter = JSONFormatter()
    else:
        formatter = HumanReadableFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress noisy third-party library DEBUG logs
    library_levels = {
        'httpcore': logging.INFO,
        'httpx': logging.INFO,
        'urllib3': logging.INFO,
        'transformers': logging.WARNING,
        'accelerate': logging.WARNING,
        'datasets': logging.WARNING,
        'bitsandbytes': logging.WARNING,
    }

    for logger_name, level in library_levels.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str, level: Optional[int] = None) -> InfraMindLogger:
    """
    Get or create an enhanced logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Optional logging level (overrides config)

    Returns:
        InfraMindLogger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Hello world")

        # Enhanced features:
        with logger.correlation(run_id="run-123"):
            with logger.timer("training_step"):
                logger.info("Processing", epoch=1, loss=0.5)
    """
    # Ensure root logger is configured
    _configure_root_logger()

    # Get Python logger
    python_logger = logging.getLogger(name)

    if level is not None:
        python_logger.setLevel(level)

    # Wrap in enhanced logger
    return InfraMindLogger(name, python_logger)


__all__ = [
    # Main API
    'get_logger',
    'set_training_context',

    # Correlation
    'set_correlation_id',
    'get_correlation_id',
    'set_run_id',
    'get_run_id',
    'set_experiment_id',
    'get_experiment_id',
    'correlation_context',
    'generate_correlation_id',

    # Metrics
    'get_global_metrics',

    # Advanced (for custom setup)
    'InfraMindLogger',
    'JSONFormatter',
    'HumanReadableFormatter',
    'CompactFormatter',
]
