"""
Request Correlation for Training Logging.

Provides context-aware training tracking using Python's contextvars.
Allows correlation IDs and training metadata to persist across async operations.
"""

import contextvars
from typing import Optional, Dict, Any
from contextlib import contextmanager
import uuid


# Context variables for training tracking
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)
run_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'run_id', default=None
)
experiment_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'experiment_id', default=None
)
extra_context_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    'extra_context', default={}
)


def generate_correlation_id() -> str:
    """
    Generate a unique correlation ID.

    Returns:
        12-character hex correlation ID (e.g., 'train-a1b2c3d4e5f6')
    """
    return f"train-{uuid.uuid4().hex[:12]}"


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get correlation ID from current context."""
    return correlation_id_var.get()


def set_run_id(run_id: str) -> None:
    """Set run ID for current training run."""
    run_id_var.set(run_id)


def get_run_id() -> Optional[str]:
    """Get run ID from current context."""
    return run_id_var.get()


def set_experiment_id(experiment_id: str) -> None:
    """Set experiment ID for current experiment."""
    experiment_id_var.set(experiment_id)


def get_experiment_id() -> Optional[str]:
    """Get experiment ID from current context."""
    return experiment_id_var.get()


def set_extra_context(key: str, value: Any) -> None:
    """Set extra context value for current context."""
    context = extra_context_var.get().copy()
    context[key] = value
    extra_context_var.set(context)


def get_extra_context() -> Dict[str, Any]:
    """Get all extra context from current context."""
    return extra_context_var.get().copy()


def get_all_context() -> Dict[str, Any]:
    """
    Get all context variables as a dictionary.

    Returns:
        Dictionary with all context values
    """
    context = {}

    correlation_id = get_correlation_id()
    if correlation_id:
        context['correlation_id'] = correlation_id

    run_id = get_run_id()
    if run_id:
        context['run_id'] = run_id

    experiment_id = get_experiment_id()
    if experiment_id:
        context['experiment_id'] = experiment_id

    # Merge extra context
    context.update(get_extra_context())

    return context


def clear_context() -> None:
    """Clear all context variables."""
    correlation_id_var.set(None)
    run_id_var.set(None)
    experiment_id_var.set(None)
    extra_context_var.set({})


@contextmanager
def correlation_context(
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
        with correlation_context(run_id="run-123", epoch=1):
            logger.info("Processing")  # Will include run_id and epoch
    """
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = generate_correlation_id()

    # Save current context
    old_correlation_id = get_correlation_id()
    old_run_id = get_run_id()
    old_experiment_id = get_experiment_id()
    old_extra = get_extra_context()

    try:
        # Set new context
        set_correlation_id(correlation_id)
        if run_id:
            set_run_id(run_id)
        if experiment_id:
            set_experiment_id(experiment_id)
        for key, value in extra.items():
            set_extra_context(key, value)

        yield correlation_id

    finally:
        # Restore previous context
        correlation_id_var.set(old_correlation_id)
        run_id_var.set(old_run_id)
        experiment_id_var.set(old_experiment_id)
        extra_context_var.set(old_extra)
