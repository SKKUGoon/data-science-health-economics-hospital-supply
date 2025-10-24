"""
Enhanced logging utilities with MLflow integration and consistent formatting.

This module provides centralized logging configuration that integrates with MLflow
and provides consistent log formatting and context management across the pipeline.
"""

import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Union
from pathlib import Path

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class PipelineLogger:
    """
    Enhanced logger with MLflow integration and consistent formatting.

    Provides centralized logging configuration with:
    - MLflow integration for experiment tracking
    - Consistent log formatting across components
    - Context management for structured logging
    - Performance metrics logging
    """

    _loggers: Dict[str, logging.Logger] = {}
    _configured = False

    def __init__(
        self,
        name: str,
        level: Union[str, int] = logging.INFO,
        enable_mlflow: bool = True,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pipeline logger.

        Args:
            name: Logger name (typically component/module name)
            level: Logging level
            enable_mlflow: Whether to enable MLflow logging integration
            context: Additional context to include in log messages
        """
        # Some defaults
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # INIT
        self.name = name
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.context = context or {}

        # Configure logging if not already done
        if not PipelineLogger._configured:
            self._configure_logging()
            PipelineLogger._configured = True

        # Get or create logger
        if name not in PipelineLogger._loggers:
            logger = logging.getLogger(f"pipeline.{name}")
            logger.setLevel(level)
            PipelineLogger._loggers[name] = logger

        self.logger = PipelineLogger._loggers[name]

    @classmethod
    def _configure_logging(cls):
        """Configure global logging settings."""
        # Create custom formatter
        formatter = PipelineFormatter()

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Suppress noisy third-party loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        if MLFLOW_AVAILABLE:
            logging.getLogger("mlflow").setLevel(logging.WARNING)

    def _format_message(self, message: str, extra_context: Optional[Dict[str, Any]] = None) -> str:
        """Format message with context information."""
        context = {**self.context}
        if extra_context:
            context.update(extra_context)

        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            return f"{message} [{context_str}]"
        return message

    def debug(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.debug(formatted_msg)

    def info(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """Log info message."""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.info(formatted_msg)

        # Log to MLflow if enabled and active run exists
        if self.enable_mlflow and self._is_mlflow_active():
            try:
                mlflow.log_text(formatted_msg, f"logs/{self.name}_info.txt")
            except Exception:
                # Don't fail if MLflow logging fails
                pass

    def warning(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.warning(formatted_msg)

        if self.enable_mlflow and self._is_mlflow_active():
            try:
                mlflow.log_text(formatted_msg, f"logs/{self.name}_warnings.txt")
            except Exception:
                pass

    def error(self, message: str, extra_context: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message."""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.error(formatted_msg, exc_info=exc_info)

        if self.enable_mlflow and self._is_mlflow_active():
            try:
                mlflow.log_text(formatted_msg, f"logs/{self.name}_errors.txt")
            except Exception:
                pass

    def critical(self, message: str, extra_context: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message."""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.critical(formatted_msg, exc_info=exc_info)

        if self.enable_mlflow and self._is_mlflow_active():
            try:
                mlflow.log_text(formatted_msg, f"logs/{self.name}_critical.txt")
            except Exception:
                pass

    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """
        Log metrics to both standard logging and MLflow.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time series metrics
        """
        # Log to standard logger
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items()])
        self.info(f"Metrics: {metrics_str}")

        # Log to MLflow if enabled
        if self.enable_mlflow and self._is_mlflow_active():
            try:
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(name, value, step=step)
            except Exception as e:
                self.warning(f"Failed to log metrics to MLflow: {e}")

    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters to both standard logging and MLflow.

        Args:
            params: Dictionary of parameter names and values
        """
        # Log to standard logger
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.info(f"Parameters: {params_str}")

        # Log to MLflow if enabled
        if self.enable_mlflow and self._is_mlflow_active():
            try:
                # Convert complex objects to strings for MLflow
                mlflow_params = {}
                for k, v in params.items():
                    if isinstance(v, (str, int, float, bool)):
                        mlflow_params[k] = v
                    else:
                        mlflow_params[k] = str(v)
                mlflow.log_params(mlflow_params)
            except Exception as e:
                self.warning(f"Failed to log parameters to MLflow: {e}")

    def log_artifact(self, artifact_path: Union[str, Path], artifact_name: Optional[str] = None):
        """
        Log artifact to MLflow.

        Args:
            artifact_path: Path to the artifact file
            artifact_name: Optional name for the artifact in MLflow
        """
        if not self.enable_mlflow or not self._is_mlflow_active():
            self.debug(f"Skipping artifact logging: {artifact_path}")
            return

        try:
            if artifact_name:
                mlflow.log_artifact(str(artifact_path), artifact_name)
            else:
                mlflow.log_artifact(str(artifact_path))
            self.debug(f"Logged artifact: {artifact_path}")
        except Exception as e:
            self.warning(f"Failed to log artifact {artifact_path}: {e}")

    def time_operation(self, operation_name: str):
        """
        Context manager for timing operations.

        Args:
            operation_name: Name of the operation being timed

        Returns:
            Context manager that logs operation duration
        """
        return TimedOperation(self, operation_name)

    def update_context(self, new_context: Dict[str, Any]):
        """Update logger context."""
        self.context.update(new_context)

    def clear_context(self):
        """Clear logger context."""
        self.context.clear()

    def _is_mlflow_active(self) -> bool:
        """Check if MLflow has an active run."""
        if not MLFLOW_AVAILABLE:
            return False
        try:
            return mlflow.active_run() is not None
        except Exception:
            return False


class PipelineFormatter(logging.Formatter):
    """Custom formatter for pipeline logs."""

    def __init__(self):
        super().__init__()
        self.format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def format(self, record):
        # Use custom format
        formatter = logging.Formatter(self.format_string, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class TimedOperation:
    """Context manager for timing operations with logging."""

    def __init__(self, logger: PipelineLogger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {duration:.2f}s")
            # Log timing metric to MLflow
            self.logger.log_metrics({f"{self.operation_name}_duration_seconds": duration})
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.2f}s: {exc_val}")

    @property
    def duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def get_logger(name: str, **kwargs) -> PipelineLogger:
    """
    Convenience function to get a pipeline logger.

    Args:
        name: Logger name
        **kwargs: Additional arguments passed to PipelineLogger

    Returns:
        PipelineLogger instance
    """
    return PipelineLogger(name, **kwargs)