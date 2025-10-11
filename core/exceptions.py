"""
Shared exception hierarchy for the project.

This module defines a unified exception hierarchy that can be used
across all modules to provide consistent error handling.
"""


class PipelineError(Exception):
    """Base exception for all pipeline-related errors.

    This is the root exception class that all other pipeline exceptions
    should inherit from. It provides a consistent interface for error
    handling across the entire project.
    """

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ConfigurationError(PipelineError):
    """Exception raised for configuration-related errors.

    This exception is raised when there are issues with:
    - Invalid configuration parameters
    - Missing required configuration
    - Configuration validation failures
    """
    pass


class ValidationError(PipelineError):
    """Exception raised for data validation errors.

    This exception is raised when:
    - Input data fails validation
    - Pydantic model validation fails
    - Data quality checks fail
    """
    pass


class ClusteringError(PipelineError):
    """Exception raised for clustering algorithm errors.

    This exception is raised when:
    - Clustering algorithms fail to converge
    - Invalid clustering parameters
    - Insufficient data for clustering
    """
    pass


class MLflowIntegrationError(PipelineError):
    """Exception raised for MLflow integration errors.

    This exception is raised when:
    - MLflow tracking server is unavailable
    - Artifact logging fails
    - Experiment management errors
    - Model registry operations fail
    """
    pass


class ModelTrainingError(PipelineError):
    """Exception raised for model training errors.

    This exception is raised when:
    - Model training fails
    - Invalid training parameters
    - Insufficient training data
    """
    pass


class PredictionError(PipelineError):
    """Exception raised for prediction errors.

    This exception is raised when:
    - Model prediction fails
    - Invalid input data for prediction
    - Model not found or not loaded
    """
    pass


class DataProcessingError(PipelineError):
    """Exception raised for data processing errors.

    This exception is raised when:
    - Data transformation fails
    - Invalid data format
    - Data loading errors
    """
    pass


class RetrainingError(PipelineError):
    """Exception raised for model retraining errors.

    This exception is raised when:
    - Model retraining fails
    - Retraining trigger conditions are not met
    - Retraining process encounters errors
    """
    pass