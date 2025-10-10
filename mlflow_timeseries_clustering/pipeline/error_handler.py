"""
Comprehensive error handling and recovery mechanisms for the MLflow Time Series Clustering Pipeline.

This module provides centralized error handling, graceful degradation strategies,
and detailed error logging for all pipeline components.
"""

import logging
import traceback
import functools
from typing import Dict, Any, Optional, Callable, Union, Type, List
from datetime import datetime
import numpy as np

from ..core.exceptions import (
    PipelineError, DataValidationError, ModelTrainingError,
    PredictionError, MLflowIntegrationError, RetrainingError
)
from ..core.data_models import BatchResult
from ..mlflow_integration.logging_utils import PipelineLogger


logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Centralized error handler for the pipeline with recovery strategies.

    This class provides comprehensive error handling including:
    - Error classification and routing
    - Graceful degradation strategies
    - Detailed error logging and debugging
    - Recovery mechanisms for different error types
    """

    def __init__(self, pipeline_name: str = "TimeSeriesClusteringPipeline"):
        """
        Initialize the error handler.

        Args:
            pipeline_name: Name of the pipeline for logging context
        """
        self.pipeline_name = pipeline_name
        self.logger = PipelineLogger(f"ErrorHandler.{pipeline_name}")

        # Error tracking
        self.error_history: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_attempts: Dict[str, int] = {}

        # Recovery strategies
        self.recovery_strategies: Dict[Type[Exception], Callable] = {
            DataValidationError: self._handle_data_validation_error,
            ModelTrainingError: self._handle_model_training_error,
            PredictionError: self._handle_prediction_error,
            MLflowIntegrationError: self._handle_mlflow_error,
            RetrainingError: self._handle_retraining_error
        }

        # Fallback configurations
        self.fallback_config = {
            'max_recovery_attempts': 3,
            'enable_graceful_degradation': True,
            'log_full_traceback': True,
            'enable_error_notifications': False
        }

        self.logger.info(f"ErrorHandler initialized for pipeline: {pipeline_name}")

    def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                    enable_recovery: bool = True) -> Dict[str, Any]:
        """
        Handle an error with appropriate recovery strategy.

        Args:
            error: The exception that occurred
            context: Additional context information
            enable_recovery: Whether to attempt recovery

        Returns:
            Dictionary containing error handling results
        """
        error_id = self._generate_error_id()
        error_type = type(error).__name__

        self.logger.error(f"Handling error {error_id}: {error_type} - {str(error)}")

        # Record error
        error_record = self._record_error(error, error_id, context)

        # Attempt recovery if enabled
        recovery_result = None
        if enable_recovery:
            recovery_result = self._attempt_recovery(error, context)

        # Log detailed error information
        self._log_error_details(error, error_record, recovery_result)

        # Update error statistics
        self._update_error_statistics(error_type)

        return {
            'error_id': error_id,
            'error_type': error_type,
            'error_message': str(error),
            'recovery_attempted': enable_recovery,
            'recovery_result': recovery_result,
            'error_record': error_record,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f"ERR_{self.pipeline_name}_{timestamp}"

    def _record_error(self, error: Exception, error_id: str,
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Record error details for tracking and analysis."""
        error_record = {
            'error_id': error_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'traceback': traceback.format_exc() if self.fallback_config['log_full_traceback'] else None
        }

        # Add component-specific details if available
        if hasattr(error, 'component'):
            error_record['component'] = error.component
        if hasattr(error, 'details'):
            error_record['details'] = error.details

        # Store in error history
        self.error_history.append(error_record)

        # Keep only recent errors to prevent memory issues
        max_history_size = 1000
        if len(self.error_history) > max_history_size:
            self.error_history = self.error_history[-max_history_size:]

        return error_record

    def _attempt_recovery(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Attempt to recover from the error using appropriate strategy."""
        error_type = type(error)
        recovery_key = f"{error_type.__name__}_{id(error)}"

        # Check recovery attempt limits
        if self.recovery_attempts.get(recovery_key, 0) >= self.fallback_config['max_recovery_attempts']:
            self.logger.warning(f"Maximum recovery attempts reached for error type: {error_type.__name__}")
            return {
                'recovery_attempted': False,
                'recovery_successful': False,
                'reason': 'max_attempts_exceeded',
                'attempts': self.recovery_attempts[recovery_key]
            }

        # Increment recovery attempt counter
        self.recovery_attempts[recovery_key] = self.recovery_attempts.get(recovery_key, 0) + 1

        # Find appropriate recovery strategy
        recovery_strategy = None
        for exception_type, strategy in self.recovery_strategies.items():
            if isinstance(error, exception_type):
                recovery_strategy = strategy
                break

        if recovery_strategy is None:
            self.logger.warning(f"No recovery strategy found for error type: {error_type.__name__}")
            return {
                'recovery_attempted': False,
                'recovery_successful': False,
                'reason': 'no_strategy_available'
            }

        # Attempt recovery
        try:
            self.logger.info(f"Attempting recovery for error type: {error_type.__name__} (attempt {self.recovery_attempts[recovery_key]})")

            recovery_result = recovery_strategy(error, context)

            if recovery_result.get('recovery_successful', False):
                self.logger.info(f"Recovery successful for error type: {error_type.__name__}")
                # Reset recovery counter on success
                self.recovery_attempts[recovery_key] = 0
            else:
                self.logger.warning(f"Recovery failed for error type: {error_type.__name__}")

            return recovery_result

        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy failed: {str(recovery_error)}")
            return {
                'recovery_attempted': True,
                'recovery_successful': False,
                'reason': 'recovery_strategy_failed',
                'recovery_error': str(recovery_error)
            }

    def _handle_data_validation_error(self, error: DataValidationError,
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle data validation errors with data cleaning strategies."""
        self.logger.info("Attempting data validation error recovery")

        recovery_result = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'strategy': 'data_cleaning',
            'actions_taken': []
        }

        try:
            # Extract data from context if available
            if context and 'data' in context:
                data = context['data']

                if isinstance(data, dict) and 'X' in data:
                    X = data['X']
                    y = data.get('y')

                    # Attempt data cleaning
                    cleaned_data = self._clean_data(X, y)

                    if cleaned_data['success']:
                        recovery_result['recovery_successful'] = True
                        recovery_result['cleaned_data'] = cleaned_data
                        recovery_result['actions_taken'] = cleaned_data['actions_taken']
                        self.logger.info("Data validation error recovered through data cleaning")
                    else:
                        recovery_result['reason'] = 'data_cleaning_failed'
                        recovery_result['cleaning_errors'] = cleaned_data.get('errors', [])
                else:
                    recovery_result['reason'] = 'no_data_in_context'
            else:
                recovery_result['reason'] = 'no_context_data'

        except Exception as e:
            recovery_result['reason'] = 'recovery_exception'
            recovery_result['recovery_error'] = str(e)

        return recovery_result

    def _handle_model_training_error(self, error: ModelTrainingError,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle model training errors with fallback strategies."""
        self.logger.info("Attempting model training error recovery")

        recovery_result = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'strategy': 'parameter_adjustment',
            'actions_taken': []
        }

        try:
            # Determine model type and adjust parameters
            model_type = getattr(error, 'details', {}).get('model_type', 'unknown')

            if model_type == 'HDBSCAN':
                recovery_result.update(self._recover_hdbscan_training(error, context))
            elif model_type in ['LightGBM_TimeSeries', 'LightGBM_Resource']:
                recovery_result.update(self._recover_lightgbm_training(error, context))
            elif model_type == 'kNN_fallback':
                recovery_result.update(self._recover_knn_training(error, context))
            else:
                recovery_result['reason'] = f'unsupported_model_type_{model_type}'

        except Exception as e:
            recovery_result['reason'] = 'recovery_exception'
            recovery_result['recovery_error'] = str(e)

        return recovery_result

    def _handle_prediction_error(self, error: PredictionError,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle prediction errors with fallback predictions."""
        self.logger.info("Attempting prediction error recovery")

        recovery_result = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'strategy': 'fallback_prediction',
            'actions_taken': []
        }

        try:
            prediction_type = getattr(error, 'details', {}).get('prediction_type', 'unknown')
            batch_size = getattr(error, 'details', {}).get('batch_size', 0)

            if prediction_type == 'cluster_assignment':
                fallback_result = self._create_fallback_cluster_assignments(batch_size, context)
            elif prediction_type == 'timeseries_prediction':
                fallback_result = self._create_fallback_timeseries_predictions(batch_size, context)
            elif prediction_type == 'resource_prediction':
                fallback_result = self._create_fallback_resource_predictions(batch_size, context)
            else:
                fallback_result = {'success': False, 'reason': f'unsupported_prediction_type_{prediction_type}'}

            if fallback_result.get('success', False):
                recovery_result['recovery_successful'] = True
                recovery_result['fallback_predictions'] = fallback_result['predictions']
                recovery_result['actions_taken'] = fallback_result.get('actions_taken', [])
            else:
                recovery_result['reason'] = fallback_result.get('reason', 'fallback_creation_failed')

        except Exception as e:
            recovery_result['reason'] = 'recovery_exception'
            recovery_result['recovery_error'] = str(e)

        return recovery_result

    def _handle_mlflow_error(self, error: MLflowIntegrationError,
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle MLflow integration errors with local fallbacks."""
        self.logger.info("Attempting MLflow error recovery")

        recovery_result = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'strategy': 'local_fallback',
            'actions_taken': []
        }

        try:
            operation = getattr(error, 'details', {}).get('operation', 'unknown')

            if operation in ['log_metrics', 'log_parameters', 'log_artifacts']:
                # Enable local logging fallback
                recovery_result['recovery_successful'] = True
                recovery_result['actions_taken'].append('enabled_local_logging')
                recovery_result['fallback_mode'] = 'local_logging'
                self.logger.info("MLflow error recovered with local logging fallback")
            elif operation in ['model_registry', 'register_model']:
                # Skip model registry operations
                recovery_result['recovery_successful'] = True
                recovery_result['actions_taken'].append('skipped_model_registry')
                recovery_result['fallback_mode'] = 'skip_registry'
                self.logger.info("MLflow error recovered by skipping model registry")
            else:
                recovery_result['reason'] = f'unsupported_operation_{operation}'

        except Exception as e:
            recovery_result['reason'] = 'recovery_exception'
            recovery_result['recovery_error'] = str(e)

        return recovery_result

    def _handle_retraining_error(self, error: RetrainingError,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle retraining errors with rollback strategies."""
        self.logger.info("Attempting retraining error recovery")

        recovery_result = {
            'recovery_attempted': True,
            'recovery_successful': False,
            'strategy': 'rollback_to_previous',
            'actions_taken': []
        }

        try:
            # Attempt to rollback to previous model version
            if context and 'previous_models' in context:
                recovery_result['recovery_successful'] = True
                recovery_result['actions_taken'].append('rolled_back_to_previous_models')
                recovery_result['rollback_info'] = context['previous_models']
                self.logger.info("Retraining error recovered by rolling back to previous models")
            else:
                # Disable retraining temporarily
                recovery_result['recovery_successful'] = True
                recovery_result['actions_taken'].append('disabled_automatic_retraining')
                recovery_result['fallback_mode'] = 'manual_retraining_only'
                self.logger.info("Retraining error recovered by disabling automatic retraining")

        except Exception as e:
            recovery_result['reason'] = 'recovery_exception'
            recovery_result['recovery_error'] = str(e)

        return recovery_result

    def _clean_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Attempt to clean data for validation errors."""
        cleaning_result = {
            'success': False,
            'actions_taken': [],
            'errors': []
        }

        try:
            cleaned_X = X.copy()
            cleaned_y = y.copy() if y is not None else None

            # Handle NaN values
            if np.any(np.isnan(cleaned_X)):
                # Replace NaN with column means
                col_means = np.nanmean(cleaned_X, axis=0)
                nan_mask = np.isnan(cleaned_X)
                cleaned_X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
                cleaning_result['actions_taken'].append('replaced_nan_with_column_means')

            # Handle infinite values
            if np.any(np.isinf(cleaned_X)):
                # Replace inf with column max/min
                inf_mask = np.isinf(cleaned_X)
                pos_inf_mask = cleaned_X == np.inf
                neg_inf_mask = cleaned_X == -np.inf

                for col in range(cleaned_X.shape[1]):
                    col_data = cleaned_X[:, col]
                    finite_data = col_data[np.isfinite(col_data)]

                    if len(finite_data) > 0:
                        col_max = np.max(finite_data)
                        col_min = np.min(finite_data)

                        cleaned_X[pos_inf_mask[:, col], col] = col_max
                        cleaned_X[neg_inf_mask[:, col], col] = col_min

                cleaning_result['actions_taken'].append('replaced_inf_with_column_extremes')

            # Handle y if provided
            if cleaned_y is not None:
                if np.any(np.isnan(cleaned_y)):
                    y_mean = np.nanmean(cleaned_y)
                    cleaned_y[np.isnan(cleaned_y)] = y_mean
                    cleaning_result['actions_taken'].append('replaced_y_nan_with_mean')

                if np.any(np.isinf(cleaned_y)):
                    finite_y = cleaned_y[np.isfinite(cleaned_y)]
                    if len(finite_y) > 0:
                        y_max = np.max(finite_y)
                        y_min = np.min(finite_y)
                        cleaned_y[cleaned_y == np.inf] = y_max
                        cleaned_y[cleaned_y == -np.inf] = y_min
                    cleaning_result['actions_taken'].append('replaced_y_inf_with_extremes')

            # Validate cleaned data
            if cleaned_X.shape[0] > 0 and cleaned_X.shape[1] > 0:
                if not np.any(np.isnan(cleaned_X)) and not np.any(np.isinf(cleaned_X)):
                    if cleaned_y is None or (not np.any(np.isnan(cleaned_y)) and not np.any(np.isinf(cleaned_y))):
                        cleaning_result['success'] = True
                        cleaning_result['cleaned_X'] = cleaned_X
                        if cleaned_y is not None:
                            cleaning_result['cleaned_y'] = cleaned_y

        except Exception as e:
            cleaning_result['errors'].append(str(e))

        return cleaning_result

    def _recover_hdbscan_training(self, error: ModelTrainingError,
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recover HDBSCAN training with adjusted parameters."""
        recovery_info = {'recovery_successful': False}

        try:
            # Suggest more conservative HDBSCAN parameters
            suggested_params = {
                'min_cluster_size': max(10, int(error.details.get('training_data_size', 100) * 0.05)),
                'min_samples': 5,
                'cluster_selection_epsilon': 0.1,
                'cluster_selection_method': 'leaf'
            }

            recovery_info['recovery_successful'] = True
            recovery_info['suggested_parameters'] = suggested_params
            recovery_info['actions_taken'] = ['adjusted_hdbscan_parameters']

        except Exception as e:
            recovery_info['recovery_error'] = str(e)

        return recovery_info

    def _recover_lightgbm_training(self, error: ModelTrainingError,
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recover LightGBM training with adjusted parameters."""
        recovery_info = {'recovery_successful': False}

        try:
            # Suggest more conservative LightGBM parameters
            suggested_params = {
                'num_leaves': 15,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'min_data_in_leaf': 10,
                'verbose': -1
            }

            recovery_info['recovery_successful'] = True
            recovery_info['suggested_parameters'] = suggested_params
            recovery_info['actions_taken'] = ['adjusted_lightgbm_parameters']

        except Exception as e:
            recovery_info['recovery_error'] = str(e)

        return recovery_info

    def _recover_knn_training(self, error: ModelTrainingError,
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recover kNN training with adjusted parameters."""
        recovery_info = {'recovery_successful': False}

        try:
            # Suggest more conservative kNN parameters
            suggested_params = {
                'n_neighbors': 3,
                'weights': 'uniform',
                'algorithm': 'auto',
                'metric': 'euclidean'
            }

            recovery_info['recovery_successful'] = True
            recovery_info['suggested_parameters'] = suggested_params
            recovery_info['actions_taken'] = ['adjusted_knn_parameters']

        except Exception as e:
            recovery_info['recovery_error'] = str(e)

        return recovery_info

    def _create_fallback_cluster_assignments(self, batch_size: int,
                                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create fallback cluster assignments."""
        try:
            # Assign all points to cluster 0 as fallback
            fallback_assignments = np.zeros(batch_size, dtype=int)

            return {
                'success': True,
                'predictions': fallback_assignments,
                'actions_taken': ['assigned_all_to_cluster_0'],
                'fallback_type': 'uniform_cluster_assignment'
            }
        except Exception as e:
            return {'success': False, 'reason': str(e)}

    def _create_fallback_timeseries_predictions(self, batch_size: int,
                                              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create fallback time series predictions."""
        try:
            # Use zero predictions as fallback
            fallback_predictions = np.zeros(batch_size)

            # If historical data is available, use mean
            if context and 'historical_predictions' in context:
                historical_mean = np.mean(context['historical_predictions'])
                fallback_predictions.fill(historical_mean)
                action = 'used_historical_mean'
            else:
                action = 'used_zero_predictions'

            return {
                'success': True,
                'predictions': fallback_predictions,
                'actions_taken': [action],
                'fallback_type': 'mean_or_zero_prediction'
            }
        except Exception as e:
            return {'success': False, 'reason': str(e)}

    def _create_fallback_resource_predictions(self, batch_size: int,
                                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create fallback resource predictions."""
        try:
            # Default resource predictions
            fallback_resources = {
                0: {  # Default cluster
                    'processing_time': 1.0,
                    'memory_usage': 100.0,
                    'computational_complexity': 1.0
                }
            }

            return {
                'success': True,
                'predictions': fallback_resources,
                'actions_taken': ['used_default_resource_estimates'],
                'fallback_type': 'default_resource_prediction'
            }
        except Exception as e:
            return {'success': False, 'reason': str(e)}

    def _log_error_details(self, error: Exception, error_record: Dict[str, Any],
                          recovery_result: Optional[Dict[str, Any]] = None) -> None:
        """Log detailed error information."""
        try:
            # Log basic error information
            self.logger.error(f"Error Details - ID: {error_record['error_id']}, Type: {error_record['error_type']}, Message: {error_record['error_message']}")

            # Log context if available
            if error_record.get('context'):
                logger.debug(f"Error Context: {error_record['context']}")

            # Log component and details if available
            if error_record.get('component'):
                self.logger.error(f"Error Component: {error_record['component']}")

            if error_record.get('details'):
                logger.debug(f"Error Details: {error_record['details']}")

            # Log recovery information
            if recovery_result:
                if recovery_result.get('recovery_successful'):
                    self.logger.info("Recovery Successful - Actions: %s",
                                   recovery_result.get('actions_taken', []))
                else:
                    self.logger.warning(f"Recovery Failed - Reason: {recovery_result.get('reason', 'unknown')}")

            # Log full traceback if enabled
            if self.fallback_config['log_full_traceback'] and error_record.get('traceback'):
                logger.debug(f"Full Traceback:\n{error_record['traceback']}")

        except Exception as logging_error:
            # Fallback logging if detailed logging fails
            print(f"Error logging failed: {logging_error}")
            print(f"Original error: {error}")

    def _update_error_statistics(self, error_type: str) -> None:
        """Update error statistics for monitoring."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = sum(self.error_counts.values())

        statistics = {
            'total_errors': total_errors,
            'error_counts_by_type': self.error_counts.copy(),
            'error_rates_by_type': {
                error_type: count / total_errors if total_errors > 0 else 0
                for error_type, count in self.error_counts.items()
            },
            'recent_errors': len([
                error for error in self.error_history
                if (datetime.now() - datetime.fromisoformat(error['timestamp'])).total_seconds() < 3600
            ]),
            'recovery_attempts': self.recovery_attempts.copy(),
            'total_error_records': len(self.error_history)
        }

        return statistics

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error records."""
        return self.error_history[-limit:] if self.error_history else []

    def clear_error_history(self) -> None:
        """Clear error history (useful for testing or maintenance)."""
        self.error_history.clear()
        self.error_counts.clear()
        self.recovery_attempts.clear()
        self.logger.info("Error history cleared")

    def update_fallback_config(self, config_updates: Dict[str, Any]) -> None:
        """Update fallback configuration."""
        self.fallback_config.update(config_updates)
        self.logger.info(f"Fallback configuration updated: {config_updates}")


def error_handler_decorator(error_handler: ErrorHandler,
                          enable_recovery: bool = True,
                          reraise_on_failure: bool = True):
    """
    Decorator for automatic error handling in pipeline methods.

    Args:
        error_handler: ErrorHandler instance to use
        enable_recovery: Whether to attempt error recovery
        reraise_on_failure: Whether to reraise the error if recovery fails

    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Extract context from function arguments
                context = {
                    'function_name': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }

                # Add data context if available
                if args and hasattr(args[0], '__class__'):
                    context['class_name'] = args[0].__class__.__name__

                # Handle the error
                error_result = error_handler.handle_error(e, context, enable_recovery)

                # Check if recovery was successful
                if error_result.get('recovery_result', {}).get('recovery_successful', False):
                    # Return fallback result if available
                    recovery_result = error_result['recovery_result']
                    if 'fallback_predictions' in recovery_result:
                        return recovery_result['fallback_predictions']
                    elif 'cleaned_data' in recovery_result:
                        # Retry with cleaned data
                        cleaned_data = recovery_result['cleaned_data']
                        if 'cleaned_X' in cleaned_data and 'cleaned_y' in cleaned_data:
                            # Update kwargs with cleaned data
                            if 'X' in kwargs:
                                kwargs['X'] = cleaned_data['cleaned_X']
                            if 'y' in kwargs:
                                kwargs['y'] = cleaned_data['cleaned_y']
                            # Retry the function call
                            return func(*args, **kwargs)

                # Reraise the error if recovery failed and reraise is enabled
                if reraise_on_failure:
                    raise e
                else:
                    # Return error information instead of raising
                    return error_result

        return wrapper
    return decorator


def create_graceful_batch_result(error: Exception, batch_size: int) -> BatchResult:
    """
    Create a graceful BatchResult for error scenarios.

    Args:
        error: The error that occurred
        batch_size: Size of the batch being processed

    Returns:
        BatchResult with fallback values
    """
    try:
        return BatchResult(
            cluster_assignments=np.zeros(batch_size, dtype=int),  # Assign all to cluster 0
            timeseries_predictions=np.zeros(batch_size),  # Zero predictions
            resource_predictions={0: {'processing_time': 1.0, 'memory_usage': 100.0}},
            noise_ratio=1.0,  # Assume all noise for safety
            processing_time=0.0
        )
    except Exception as fallback_error:
        # If even the fallback fails, create minimal result
        return BatchResult(
            cluster_assignments=np.array([]),
            timeseries_predictions=np.array([]),
            resource_predictions={},
            noise_ratio=1.0,
            processing_time=0.0
        )


class PipelineErrorContext:
    """
    Context manager for pipeline operations with automatic error handling.

    Usage:
        with PipelineErrorContext(error_handler, "training_phase") as ctx:
            # Pipeline operations
            result = some_pipeline_operation()
            ctx.add_context("operation_result", result)
    """

    def __init__(self, error_handler: ErrorHandler, operation_name: str):
        self.error_handler = error_handler
        self.operation_name = operation_name
        self.context = {'operation_name': operation_name}
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.context['start_time'] = self.start_time.isoformat()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Add timing information
            if self.start_time:
                duration = (datetime.now() - self.start_time).total_seconds()
                self.context['operation_duration'] = duration

            # Handle the error
            self.error_handler.handle_error(exc_val, self.context)

        return False  # Don't suppress the exception

    def add_context(self, key: str, value: Any) -> None:
        """Add additional context information."""
        self.context[key] = value