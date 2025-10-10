"""
Custom exception classes for the MLflow Time Series Clustering Pipeline.

This module defines pipeline-specific exceptions for better error handling
and debugging throughout the system.
"""


class PipelineError(Exception):
    """
    Base exception class for all pipeline-related errors.

    This is the parent class for all custom exceptions in the pipeline,
    allowing for broad exception handling when needed.
    """

    def __init__(self, message: str, component: str = None, details: dict = None):
        """
        Initialize pipeline error.

        Args:
            message: Human-readable error message
            component: Name of the component where error occurred
            details: Additional error details for debugging
        """
        self.message = message
        self.component = component
        self.details = details or {}

        # Format the full error message
        full_message = message
        if component:
            full_message = f"[{component}] {message}"

        super().__init__(full_message)

    def __str__(self):
        """Return string representation of the error."""
        base_str = super().__str__()
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base_str} (Details: {details_str})"
        return base_str


class DataValidationError(PipelineError):
    """
    Exception raised when input data validation fails.

    This exception is raised when data doesn't meet the expected format,
    contains invalid values, or fails other validation checks.
    """

    def __init__(self, message: str, data_type: str = None, expected_shape: tuple = None,
                 actual_shape: tuple = None, **kwargs):
        """
        Initialize data validation error.

        Args:
            message: Human-readable error message
            data_type: Type of data that failed validation
            expected_shape: Expected data shape
            actual_shape: Actual data shape
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if data_type:
            details['data_type'] = data_type
        if expected_shape:
            details['expected_shape'] = expected_shape
        if actual_shape:
            details['actual_shape'] = actual_shape

        kwargs['details'] = details
        super().__init__(message, component="DataValidation", **kwargs)


class ModelTrainingError(PipelineError):
    """
    Exception raised when model training fails.

    This exception is raised when HDBSCAN clustering, LightGBM training,
    or kNN model fitting encounters errors.
    """

    def __init__(self, message: str, model_type: str = None, cluster_id: int = None,
                 training_data_size: int = None, **kwargs):
        """
        Initialize model training error.

        Args:
            message: Human-readable error message
            model_type: Type of model that failed training
            cluster_id: Cluster ID if error is cluster-specific
            training_data_size: Size of training data
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if model_type:
            details['model_type'] = model_type
        if cluster_id is not None:
            details['cluster_id'] = cluster_id
        if training_data_size is not None:
            details['training_data_size'] = training_data_size

        kwargs['details'] = details
        super().__init__(message, component="ModelTraining", **kwargs)


class PredictionError(PipelineError):
    """
    Exception raised when model prediction fails.

    This exception is raised when cluster assignment, time series prediction,
    or resource usage prediction encounters errors.
    """

    def __init__(self, message: str, prediction_type: str = None, cluster_id: int = None,
                 batch_size: int = None, **kwargs):
        """
        Initialize prediction error.

        Args:
            message: Human-readable error message
            prediction_type: Type of prediction that failed
            cluster_id: Cluster ID if error is cluster-specific
            batch_size: Size of batch being predicted
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if prediction_type:
            details['prediction_type'] = prediction_type
        if cluster_id is not None:
            details['cluster_id'] = cluster_id
        if batch_size is not None:
            details['batch_size'] = batch_size

        kwargs['details'] = details
        super().__init__(message, component="Prediction", **kwargs)


class MLflowIntegrationError(PipelineError):
    """
    Exception raised when MLflow integration fails.

    This exception is raised when MLflow logging, artifact storage,
    or model registry operations encounter errors.
    """

    def __init__(self, message: str, operation: str = None, experiment_name: str = None,
                 run_id: str = None, **kwargs):
        """
        Initialize MLflow integration error.

        Args:
            message: Human-readable error message
            operation: MLflow operation that failed
            experiment_name: Name of the experiment
            run_id: MLflow run ID if applicable
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        if experiment_name:
            details['experiment_name'] = experiment_name
        if run_id:
            details['run_id'] = run_id

        kwargs['details'] = details
        super().__init__(message, component="MLflowIntegration", **kwargs)


class RetrainingError(PipelineError):
    """
    Exception raised when model retraining fails.

    This exception is raised when automatic or manual retraining
    encounters errors during execution.
    """

    def __init__(self, message: str, trigger_reason: str = None, noise_ratio: float = None,
                 **kwargs):
        """
        Initialize retraining error.

        Args:
            message: Human-readable error message
            trigger_reason: Reason that triggered retraining
            noise_ratio: Current noise ratio that triggered retraining
            **kwargs: Additional arguments passed to parent class
        """
        details = kwargs.get('details', {})
        if trigger_reason:
            details['trigger_reason'] = trigger_reason
        if noise_ratio is not None:
            details['noise_ratio'] = noise_ratio

        kwargs['details'] = details
        super().__init__(message, component="Retraining", **kwargs)