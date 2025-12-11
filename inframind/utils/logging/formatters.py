"""
Custom Log Formatters for InfraMind.

Provides JSON and human-readable formatters for structured training logs.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict
from .correlation import get_all_context


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs as JSON objects with timestamp, level, logger name, message,
    and any training context variables (run_id, epoch, loss, etc.).
    """

    def __init__(self, include_exc_info: bool = True):
        super().__init__()
        self.include_exc_info = include_exc_info

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data: Dict[str, Any] = {
            'timestamp': self._get_timestamp(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add correlation context
        context = get_all_context()
        log_data.update(context)

        # Add custom fields from record
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if self.include_exc_info and record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None,
            }

        # Add source location
        log_data['source'] = {
            'file': record.pathname,
            'line': record.lineno,
            'function': record.funcName,
        }

        return json.dumps(log_data, default=str)

    def _get_timestamp(self, record: logging.LogRecord) -> str:
        """Get ISO 8601 timestamp from record."""
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.isoformat()


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development.

    Outputs logs in a clean, colored format suitable for console viewing.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }

    def __init__(self, use_colors: bool = True, include_context: bool = True):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.use_colors = use_colors
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        # Get base format
        formatted = super().format(record)

        # Add correlation context if enabled
        if self.include_context:
            context = get_all_context()
            if context:
                context_str = ' | '.join([f"{k}={v}" for k, v in context.items()])
                formatted += f" | {context_str}"

        # Add custom fields
        if hasattr(record, 'extra_fields'):
            extra_str = ' | '.join([f"{k}={v}" for k, v in record.extra_fields.items()])
            formatted += f" | {extra_str}"

        # Apply colors if enabled
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            formatted = f"{color}{formatted}{reset}"

        return formatted


class CompactFormatter(logging.Formatter):
    """
    Compact formatter for production console output.

    Outputs minimal information, suitable for production environments.
    """

    def __init__(self):
        super().__init__(
            fmt='%(levelname)s | %(name)s | %(message)s',
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format log record compactly."""
        formatted = super().format(record)

        # Add run ID if present (compact format)
        context = get_all_context()
        if 'run_id' in context:
            formatted = f"[{context['run_id']}] {formatted}"

        return formatted
