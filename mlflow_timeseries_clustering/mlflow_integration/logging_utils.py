"""
Logging utilities for the MLflow Time Series Clustering Pipeline.

This module provides structured logging utilities with plain text formatting
and MLflow integration for comprehensive pipeline monitoring.
"""

import logging
import mlflow
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json

from ..core.exceptions import MLflowIntegrationError


class PipelineLogger:
    """
    Centralized logging utility for the Time Series Clustering Pipeline.

    This class provides structured logging with MLflow integration,
    ensuring consistent log formatting without emojis or special characters.
    """

    def __init__(self, component_name: str, log_level: int = logging.INFO):
        """
        Initialize the pipeline logger.

        Args:
            component_name: Name of the component using this logger
            log_level: Logging level (default: INFO)
        """
        self.component_name = component_name
        self.logger = logging.getLogger(f"pipeline.{component_name}")
        self.logger.setLevel(log_level)

        # Set up plain text formatter
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_parameters(self, params: Dict[str, Any], prefix: str = "") -> None:
        """
        Log parameters to both local logger and MLflow.

        Args:
            params: Dictionary of parameters to log
            prefix: Optional prefix for parameter names
        """
        try:
            for param_name, param_value in params.items():
                full_param_name = f"{prefix}.{param_name}" if prefix else param_name

                # Log to MLflow
                mlflow.log_param(full_param_name, param_value)

                # Log locally
                self.logger.info(f"Parameter {full_param_name}: {param_value}")

        except Exception as e:
            self.logger.error(f"Failed to log parameters: {str(e)}")
            raise MLflowIntegrationError(
                f"Failed to log parameters for {self.component_name}: {str(e)}",
                operation="log_parameters",
                details={"component": self.component_name}
            )

    def log_metrics(self, metrics: Dict[str, Union[int, float]],
                   step: Optional[int] = None, prefix: str = "") -> None:
        """
        Log metrics to both local logger and MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series metrics
            prefix: Optional prefix for metric names
        """
        try:
            for metric_name, metric_value in metrics.items():
                full_metric_name = f"{prefix}.{metric_name}" if prefix else metric_name

                # Log to MLflow
                if step is not None:
                    mlflow.log_metric(full_metric_name, metric_value, step=step)
                else:
                    mlflow.log_metric(full_metric_name, metric_value)

                # Log locally
                step_info = f" (step {step})" if step is not None else ""
                self.logger.info(f"Metric {full_metric_name}: {metric_value}{step_info}")

        except Exception as e:
            self.logger.error(f"Failed to log metrics: {str(e)}")
            raise MLflowIntegrationError(
                f"Failed to log metrics for {self.component_name}: {str(e)}",
                operation="log_metrics",
                details={"component": self.component_name}
            )

    def log_model_info(self, model_type: str, model_params: Dict[str, Any],
                      training_metrics: Dict[str, float],
                      cluster_id: Optional[int] = None) -> None:
        """
        Log comprehensive model information.

        Args:
            model_type: Type of model (e.g., 'HDBSCAN', 'LightGBM')
            model_params: Model parameters
            training_metrics: Training performance metrics
            cluster_id: Optional cluster ID for cluster-specific models
        """
        cluster_info = f" for cluster {cluster_id}" if cluster_id is not None else ""
        self.logger.info(f"Training {model_type} model{cluster_info}")

        # Log parameters with model type prefix
        self.log_parameters(model_params, prefix=f"{model_type.lower()}_params")

        # Log metrics with model type prefix
        self.log_metrics(training_metrics, prefix=f"{model_type.lower()}_metrics")

        self.logger.info(f"Completed {model_type} model training{cluster_info}")

    def log_batch_processing(self, batch_size: int, processing_time: float,
                           noise_ratio: float, predictions_count: int) -> None:
        """
        Log batch processing information.

        Args:
            batch_size: Size of the processed batch
            processing_time: Time taken to process the batch
            noise_ratio: Ratio of noise points in the batch
            predictions_count: Number of predictions made
        """
        self.logger.info(f"Processing batch of size {batch_size}")

        batch_metrics = {
            "batch_size": batch_size,
            "processing_time_seconds": processing_time,
            "noise_ratio": noise_ratio,
            "predictions_count": predictions_count,
            "throughput_per_second": batch_size / processing_time if processing_time > 0 else 0
        }

        self.log_metrics(batch_metrics, prefix="batch")

        self.logger.info(
            f"Batch processing completed: {batch_size} samples in {processing_time:.2f}s "
            f"(noise ratio: {noise_ratio:.3f})"
        )

    def log_retraining_event(self, trigger_reason: str, noise_ratio: float,
                           old_model_version: str, new_model_version: str) -> None:
        """
        Log model retraining event.

        Args:
            trigger_reason: Reason that triggered retraining
            noise_ratio: Current noise ratio
            old_model_version: Previous model version
            new_model_version: New model version after retraining
        """
        self.logger.info(f"Model retraining triggered: {trigger_reason}")

        retraining_info = {
            "trigger_reason": trigger_reason,
            "noise_ratio": noise_ratio,
            "old_model_version": old_model_version,
            "new_model_version": new_model_version,
            "retraining_timestamp": datetime.now().isoformat()
        }

        # Log as parameters for tracking
        self.log_parameters(retraining_info, prefix="retraining")

        self.logger.info(
            f"Retraining completed: {old_model_version} -> {new_model_version} "
            f"(noise ratio: {noise_ratio:.3f})"
        )

    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Log error information with context.

        Args:
            error: Exception that occurred
            context: Additional context information
        """
        error_msg = f"Error in {self.component_name}: {str(error)}"
        self.logger.error(error_msg)

        # Log error details to MLflow as tags
        try:
            error_tags = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "component": self.component_name,
                "timestamp": datetime.now().isoformat()
            }

            if context:
                error_tags.update({f"context_{k}": str(v) for k, v in context.items()})

            mlflow.set_tags(error_tags)

        except Exception as mlflow_error:
            self.logger.error(f"Failed to log error to MLflow: {str(mlflow_error)}")

    def log_performance_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log performance summary information.

        Args:
            summary: Dictionary containing performance summary data
        """
        self.logger.info(f"Performance summary for {self.component_name}")

        # Separate metrics and parameters
        metrics = {}
        parameters = {}

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
            else:
                parameters[key] = value

        # Log metrics and parameters
        if metrics:
            self.log_metrics(metrics, prefix="summary")
        if parameters:
            self.log_parameters(parameters, prefix="summary")

        self.logger.info("Performance summary logging completed")

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(f"[{self.component_name}] {message}")

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(f"[{self.component_name}] {message}")

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(f"[{self.component_name}] {message}")

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(f"[{self.component_name}] {message}")


def setup_pipeline_logging(log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration for the entire pipeline.

    Args:
        log_level: Logging level for the pipeline
    """
    # Configure root logger for the pipeline
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in pipeline_logger.handlers[:]:
        pipeline_logger.removeHandler(handler)

    # Create console handler with plain text formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    pipeline_logger.addHandler(console_handler)

    # Prevent propagation to root logger
    pipeline_logger.propagate = False

    pipeline_logger.info("Pipeline logging configured successfully")