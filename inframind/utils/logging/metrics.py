"""
Training Metrics and Performance Tracking.

Tracks training performance, operation timing, and slow operations.
"""

import time
from typing import Dict, Any
from collections import defaultdict
from threading import Lock


class TrainingMetrics:
    """
    Tracks training performance and statistics.

    Thread-safe metrics collector for training operations.
    """

    def __init__(self, slow_threshold_ms: float = 5000.0):
        """
        Initialize training metrics.

        Args:
            slow_threshold_ms: Threshold for slow operation warning (default 5000ms)
        """
        self.slow_threshold_ms = slow_threshold_ms
        self._lock = Lock()

        # Initialize metrics
        self._metrics = {
            'total_logs': 0,
            'logs_by_level': defaultdict(int),
            'errors': 0,
            'warnings': 0,
            'slow_operations': 0,
            'total_log_time_ms': 0.0,
            'avg_log_time_ms': 0.0,
            'operation_times': {},
            # Training-specific metrics
            'training_steps': 0,
            'epochs_completed': 0,
            'total_loss': 0.0,
            'avg_loss': 0.0,
        }

    def record_log(self, level: str, duration_ms: float = 0.0):
        """Record a log event."""
        with self._lock:
            self._metrics['total_logs'] += 1
            self._metrics['logs_by_level'][level] += 1

            if level == 'ERROR' or level == 'CRITICAL':
                self._metrics['errors'] += 1
            elif level == 'WARNING':
                self._metrics['warnings'] += 1

            if duration_ms > 0:
                self._metrics['total_log_time_ms'] += duration_ms
                total = self._metrics['total_logs']
                self._metrics['avg_log_time_ms'] = (
                    self._metrics['total_log_time_ms'] / total
                )

    def record_operation(self, operation_name: str, duration_ms: float):
        """Record an operation timing."""
        with self._lock:
            if operation_name not in self._metrics['operation_times']:
                self._metrics['operation_times'][operation_name] = {
                    'count': 0,
                    'total_ms': 0.0,
                    'avg_ms': 0.0,
                    'min_ms': float('inf'),
                    'max_ms': 0.0,
                }

            op_stats = self._metrics['operation_times'][operation_name]
            op_stats['count'] += 1
            op_stats['total_ms'] += duration_ms
            op_stats['avg_ms'] = op_stats['total_ms'] / op_stats['count']
            op_stats['min_ms'] = min(op_stats['min_ms'], duration_ms)
            op_stats['max_ms'] = max(op_stats['max_ms'], duration_ms)

            # Check if slow
            if duration_ms > self.slow_threshold_ms:
                self._metrics['slow_operations'] += 1

    def record_training_step(self, loss: float):
        """Record a training step with loss."""
        with self._lock:
            self._metrics['training_steps'] += 1
            self._metrics['total_loss'] += loss
            self._metrics['avg_loss'] = (
                self._metrics['total_loss'] / self._metrics['training_steps']
            )

    def record_epoch(self):
        """Record epoch completion."""
        with self._lock:
            self._metrics['epochs_completed'] += 1

    def is_slow_operation(self, duration_ms: float) -> bool:
        """Check if operation exceeds slow threshold."""
        return duration_ms > self.slow_threshold_ms

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return {
                'total_logs': self._metrics['total_logs'],
                'logs_by_level': dict(self._metrics['logs_by_level']),
                'errors': self._metrics['errors'],
                'warnings': self._metrics['warnings'],
                'slow_operations': self._metrics['slow_operations'],
                'avg_log_time_ms': round(self._metrics['avg_log_time_ms'], 2),
                'training_steps': self._metrics['training_steps'],
                'epochs_completed': self._metrics['epochs_completed'],
                'avg_loss': round(self._metrics['avg_loss'], 6),
                'operation_times': {
                    k: {
                        'count': v['count'],
                        'avg_ms': round(v['avg_ms'], 2),
                        'min_ms': round(v['min_ms'], 2),
                        'max_ms': round(v['max_ms'], 2),
                    }
                    for k, v in self._metrics['operation_times'].items()
                },
            }

    def reset(self):
        """Reset all metrics to zero."""
        with self._lock:
            self._metrics = {
                'total_logs': 0,
                'logs_by_level': defaultdict(int),
                'errors': 0,
                'warnings': 0,
                'slow_operations': 0,
                'total_log_time_ms': 0.0,
                'avg_log_time_ms': 0.0,
                'operation_times': {},
                'training_steps': 0,
                'epochs_completed': 0,
                'total_loss': 0.0,
                'avg_loss': 0.0,
            }


# Global metrics instance
_global_metrics = TrainingMetrics()


def get_global_metrics() -> TrainingMetrics:
    """Get global metrics instance."""
    return _global_metrics
