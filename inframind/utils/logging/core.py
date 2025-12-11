"""
Enhanced Production Logger for InfraMind.

Production-grade logger with structured logging, correlation,
performance timing, and training metrics tracking.
"""

import logging
import time
import functools
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable

from .correlation import (
    get_all_context,
    correlation_context,
    set_correlation_id,
    set_run_id,
    set_experiment_id,
    generate_correlation_id,
)
from .metrics import get_global_metrics


class InfraMindLogger:
    """
    Enhanced production logger with structured logging.

    Features:
    - Structured JSON logging
    - Training run correlation tracking
    - Performance timing (context manager & decorator)
    - Training metrics collection
    - Backward compatible with simple logger
    """

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize enhanced logger.

        Args:
            name: Logger name (usually __name__)
            logger: Optional pre-configured Python logger
        """
        self.name = name
        self.logger = logger or logging.getLogger(name)
        self.metrics = get_global_metrics()

    def _log_with_context(
        self,
        level: int,
        message: str,
        exc_info: Optional[Any] = None,
        **extra_fields
    ):
        """Internal method to log with context enrichment."""
        start_time = time.time()

        try:
            # Create a new LogRecord with extra fields
            record = self.logger.makeRecord(
                name=self.name,
                level=level,
                fn="",
                lno=0,
                msg=message,
                args=(),
                exc_info=exc_info,
            )

            # Add extra fields as attribute
            if extra_fields:
                record.extra_fields = extra_fields

            # Handle the record
            self.logger.handle(record)

            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_log(logging.getLevelName(level), duration_ms)

        except Exception as e:
            # Fallback: log to stderr
            import sys
            print(f"Logging error: {e}", file=sys.stderr)

    def debug(self, message: str, **context):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, **context)

    def info(self, message: str, **context):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, **context)

    def warning(self, message: str, **context):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, **context)

    def error(self, message: str, exc_info: Optional[Any] = None, **context):
        """Log error message with optional exception info."""
        self._log_with_context(logging.ERROR, message, exc_info=exc_info, **context)

    def critical(self, message: str, exc_info: Optional[Any] = None, **context):
        """Log critical message with optional exception info."""
        self._log_with_context(logging.CRITICAL, message, exc_info=exc_info, **context)

    @contextmanager
    def correlation(
        self,
        correlation_id: Optional[str] = None,
        run_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        **extra
    ):
        """
        Context manager for setting correlation context.

        Args:
            correlation_id: Optional correlation ID (auto-generated if None)
            run_id: Optional training run ID
            experiment_id: Optional experiment ID
            **extra: Additional context values (e.g., epoch, step)

        Example:
            with logger.correlation(run_id="run-123", epoch=1):
                logger.info("Processing")
        """
        with correlation_context(
            correlation_id=correlation_id,
            run_id=run_id,
            experiment_id=experiment_id,
            **extra
        ) as corr_id:
            yield corr_id

    @contextmanager
    def timer(self, operation_name: str, log_start: bool = False):
        """
        Context manager for timing operations.

        Args:
            operation_name: Name of the operation to time
            log_start: Log when operation starts (default False)

        Example:
            with logger.timer("training_step"):
                result = train_batch(...)
            # Logs: "training_step completed in 234ms"
        """
        if log_start:
            self.debug(f"{operation_name} starting")

        start_time = time.time()

        try:
            yield

        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Record operation metrics
            self.metrics.record_operation(operation_name, duration_ms)

            # Log completion with timing
            if self.metrics.is_slow_operation(duration_ms):
                self.warning(
                    f"{operation_name} completed (SLOW)",
                    duration_ms=round(duration_ms, 2),
                    slow=True,
                )
            else:
                self.debug(
                    f"{operation_name} completed",
                    duration_ms=round(duration_ms, 2),
                )

    def timed(self, operation_name: Optional[str] = None):
        """
        Decorator for timing functions.

        Args:
            operation_name: Optional operation name (uses function name if None)

        Example:
            @logger.timed("process_batch")
            def process_batch(batch):
                pass
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation_name or func.__name__

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.timer(op_name):
                    return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.timer(op_name):
                    return func(*args, **kwargs)

            # Return appropriate wrapper based on function type
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def log_training_step(self, step: int, loss: float, **extra):
        """
        Log a training step with metrics.

        Args:
            step: Training step number
            loss: Loss value for this step
            **extra: Additional metrics (e.g., learning_rate, grad_norm)
        """
        self.metrics.record_training_step(loss)
        self.info(
            f"Step {step}",
            step=step,
            loss=round(loss, 6),
            **extra
        )

    def log_epoch(self, epoch: int, **extra):
        """
        Log epoch completion with metrics.

        Args:
            epoch: Epoch number
            **extra: Additional metrics (e.g., val_loss, accuracy)
        """
        self.metrics.record_epoch()
        self.info(
            f"Epoch {epoch} completed",
            epoch=epoch,
            **extra
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return self.metrics.get_metrics()

    def set_level(self, level: int):
        """Set logging level."""
        self.logger.setLevel(level)

    def add_handler(self, handler: logging.Handler):
        """Add a handler to the logger."""
        self.logger.addHandler(handler)

    def remove_handler(self, handler: logging.Handler):
        """Remove a handler from the logger."""
        self.logger.removeHandler(handler)


# Convenience methods for setting context directly
def set_training_context(
    run_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
):
    """
    Set training context for logging.

    Args:
        run_id: Training run ID
        experiment_id: Experiment ID
        correlation_id: Correlation ID
    """
    if correlation_id:
        set_correlation_id(correlation_id)
    if run_id:
        set_run_id(run_id)
    if experiment_id:
        set_experiment_id(experiment_id)
