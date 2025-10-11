"""
Core data models for the MLflow Time Series Clustering Pipeline using Pydantic BaseModels.

This module defines the primary data structures used throughout the pipeline with
comprehensive validation, serialization, and type safety. All models use Pydantic
BaseModel for automatic validation, JSON serialization, and runtime type checking.

**Model Categories:**

1. **Clustering Models:**
   - `ClusteringResult`: HDBSCAN clustering output with validation
   - Handles cluster labels, centers, noise ratios, and quality metrics

2. **Batch Processing Models:**
   - `BatchResult`: Complete batch processing output
   - Includes cluster assignments, predictions, and performance metrics

3. **Performance Monitoring Models:**
   - `PerformanceReport`: Comprehensive performance analysis
   - `PerformanceAlert`: Performance degradation alerts
   - `DriftDetectionResult`: Data drift detection results

4. **Retraining Models:**
   - `RetrainingEvent`: Retraining trigger events and metadata
   - `RetrainingResult`: Retraining operation outcomes

5. **Machine Learning Models:**
   - `ModelTrainingResult`: Model training outcomes and metrics
   - `PredictionResult`: Model prediction results with metadata

6. **Embedding Models:**
   - `EmbeddingResult`: Document embedding operation results
   - `EmbeddingMetrics`: Performance metrics for embedding operations

**Key Features:**
- **Automatic Validation**: All fields validated at runtime with meaningful errors
- **Type Safety**: Full type checking with Python type hints
- **JSON Serialization**: Automatic conversion to/from JSON with datetime handling
- **Field Documentation**: Comprehensive field descriptions for API documentation
- **Custom Validators**: Specialized validation for healthcare and ML data
- **Enum Support**: Structured enums for categorical data (AlertLevel, TriggerReason)

**Validation Features:**
- Range validation for numeric fields (0.0 ≤ noise_ratio ≤ 1.0)
- Array shape validation for numpy arrays
- Datetime handling with automatic timezone support
- Custom validators for domain-specific data patterns
- Comprehensive error messages for debugging

**Usage Examples:**

```python
from core.data_models import ClusteringResult, BatchResult, PerformanceAlert

# Create validated clustering result
result = ClusteringResult(
    labels=[0, 1, 0, -1, 1],
    cluster_centers=[[1.0, 2.0], [3.0, 4.0]],
    noise_ratio=0.2,
    metrics={"silhouette_score": 0.75, "n_clusters": 2}
)

# Automatic validation catches errors
try:
    invalid_result = ClusteringResult(
        labels=[0, 1],
        cluster_centers=[[1.0, 2.0]],
        noise_ratio=1.5  # Invalid: > 1.0
    )
except ValidationError as e:
    print(f"Validation failed: {e}")

# JSON serialization
json_data = result.json()
restored_result = ClusteringResult.parse_raw(json_data)

# Performance monitoring
alert = PerformanceAlert(
    level=AlertLevel.WARNING,
    component="ModelPredictor",
    metric="rmse",
    current_value=0.85,
    threshold_value=0.80,
    message="RMSE threshold exceeded"
)
```

**Integration with Pipeline:**
- Used by all pipeline components for data exchange
- Provides consistent data structures across modules
- Enables automatic validation at module boundaries
- Supports MLflow logging with proper serialization
- Facilitates debugging with comprehensive error messages

**Numpy Array Handling:**
- Automatic conversion between numpy arrays and Python lists
- Validation of array shapes and data types
- Proper serialization for MLflow artifact storage
- Memory-efficient handling of large datasets
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import numpy as np


class TriggerReason(str, Enum):
    """Enumeration of possible retraining trigger reasons."""
    NOISE_THRESHOLD_EXCEEDED = "noise_threshold_exceeded"
    MANUAL_COMMAND = "manual_command"
    SCHEDULED_RETRAINING = "scheduled_retraining"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ClusteringResult(BaseModel):
    """
    Result of HDBSCAN clustering operation.

    This model represents the output of clustering algorithms with
    automatic validation and serialization capabilities.
    """
    labels: List[int] = Field(
        ...,
        description="Cluster labels for each data point (-1 for noise points)"
    )
    noise_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of noise points to total points"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing clustering quality metrics"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }

    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v):
        """Validate cluster labels."""
        if not isinstance(v, (list, np.ndarray)):
            raise ValueError("labels must be a list or numpy array")
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return v


class BatchResult(BaseModel):
    """
    Result of processing a single data batch.

    This model represents the output of batch processing with
    validation for all components.
    """
    cluster_assignments: List[int] = Field(
        ...,
        description="Cluster assignments for batch data points"
    )
    timeseries_predictions: List[List[float]] = Field(
        ...,
        description="Time series predictions for each data point"
    )
    resource_predictions: Dict[int, Dict[str, float]] = Field(
        default_factory=dict,
        description="Resource usage predictions per cluster"
    )
    noise_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of noise points in this batch"
    )
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Time taken to process this batch in seconds"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }

    @field_validator('cluster_assignments')
    @classmethod
    def validate_cluster_assignments(cls, v):
        """Validate cluster assignments."""
        if not isinstance(v, (list, np.ndarray)):
            raise ValueError("cluster_assignments must be a list or numpy array")
        if isinstance(v, np.ndarray):
            v = v.tolist()
        return v

    @field_validator('timeseries_predictions')
    @classmethod
    def validate_timeseries_predictions(cls, v):
        """Validate timeseries predictions."""
        if not isinstance(v, (list, np.ndarray)):
            raise ValueError("timeseries_predictions must be a list or numpy array")
        if isinstance(v, np.ndarray):
            if v.ndim == 2:
                v = v.tolist()
            else:
                raise ValueError("timeseries_predictions must be a 2D array")
        return v


class PerformanceReport(BaseModel):
    """
    Comprehensive performance report for the pipeline.

    This model represents performance monitoring data with
    validation and automatic timestamp handling.
    """
    clustering_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metrics related to clustering performance"
    )
    model_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metrics for time series and resource usage models"
    )
    visualizations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of visualization file paths or data"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this report was generated"
    )
    model_version: str = Field(
        ...,
        min_length=1,
        description="Version identifier for the models used"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


class RetrainingEvent(BaseModel):
    """
    Data model representing a retraining event.

    This model captures information about when and why
    retraining was triggered with full validation.
    """
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the retraining was triggered"
    )
    reason: TriggerReason = Field(
        ...,
        description="Why the retraining was triggered"
    )
    trigger_value: Optional[float] = Field(
        default=None,
        description="The value that triggered retraining (e.g., noise ratio)"
    )
    threshold_value: Optional[float] = Field(
        default=None,
        description="The threshold that was exceeded (if applicable)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the trigger event"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }

    @field_validator('trigger_value', 'threshold_value')
    @classmethod
    def validate_numeric_values(cls, v):
        """Validate numeric trigger and threshold values."""
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError("trigger_value and threshold_value must be numeric or None")
        return v


class PerformanceAlert(BaseModel):
    """Performance alert data structure with validation."""
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the alert was generated"
    )
    level: AlertLevel = Field(
        ...,
        description="Alert severity level"
    )
    component: str = Field(
        ...,
        min_length=1,
        description="Component that generated the alert"
    )
    metric: str = Field(
        ...,
        min_length=1,
        description="Metric that triggered the alert"
    )
    current_value: float = Field(
        ...,
        description="Current value of the metric"
    )
    threshold_value: float = Field(
        ...,
        description="Threshold value that was exceeded"
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable alert message"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the alert"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


class DriftDetectionResult(BaseModel):
    """Drift detection result data structure with validation."""
    drift_detected: bool = Field(
        ...,
        description="Whether drift was detected"
    )
    drift_score: float = Field(
        ...,
        ge=0.0,
        description="Numerical score indicating drift magnitude"
    )
    drift_type: str = Field(
        ...,
        min_length=1,
        description="Type of drift detected"
    )
    affected_metrics: List[str] = Field(
        default_factory=list,
        description="List of metrics affected by drift"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When drift detection was performed"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the drift detection"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


class RetrainingResult(BaseModel):
    """
    Result of a retraining operation.

    This model represents the outcome of model retraining
    with validation and performance tracking.
    """
    success: bool = Field(
        ...,
        description="Whether retraining was successful"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When retraining was completed"
    )
    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Duration of retraining in seconds"
    )
    models_retrained: List[str] = Field(
        default_factory=list,
        description="List of models that were retrained"
    )
    performance_improvement: Optional[Dict[str, float]] = Field(
        default=None,
        description="Performance improvement metrics"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if retraining failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the retraining"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


class ModelTrainingResult(BaseModel):
    """
    Result of model training operation.

    This model represents the outcome of training machine learning models
    with comprehensive validation and metrics tracking.
    """
    model_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the trained model"
    )
    model_type: str = Field(
        ...,
        min_length=1,
        description="Type of model (e.g., 'timeseries', 'resource')"
    )
    cluster_id: Optional[int] = Field(
        default=None,
        description="Cluster ID if model is cluster-specific"
    )
    training_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Training performance metrics"
    )
    validation_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Validation performance metrics"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature importance scores"
    )
    training_duration: float = Field(
        ...,
        ge=0.0,
        description="Training duration in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When training was completed"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


class PredictionResult(BaseModel):
    """
    Result of model prediction operation.

    This model represents the output of model predictions
    with validation and metadata tracking.
    """
    predictions: List[Union[float, List[float]]] = Field(
        ...,
        description="Model predictions"
    )
    prediction_intervals: Optional[List[List[float]]] = Field(
        default=None,
        description="Prediction intervals if available"
    )
    confidence_scores: Optional[List[float]] = Field(
        default=None,
        description="Confidence scores for predictions"
    )
    model_id: str = Field(
        ...,
        min_length=1,
        description="ID of the model used for prediction"
    )
    cluster_assignments: Optional[List[int]] = Field(
        default=None,
        description="Cluster assignments for input data"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When predictions were made"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }

class EmbeddingMetrics(BaseModel):
    """
    Metrics for embedding operations.

    This model tracks performance and quality metrics
    for embedding generation and storage operations.
    """
    total_documents: int = Field(
        ...,
        ge=0,
        description="Total number of documents processed"
    )
    total_embeddings: int = Field(
        ...,
        ge=0,
        description="Total number of embeddings generated"
    )
    avg_embedding_time: float = Field(
        ...,
        ge=0.0,
        description="Average time to generate embeddings in seconds"
    )
    total_processing_time: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in seconds"
    )
    documents_per_second: float = Field(
        ...,
        ge=0.0,
        description="Processing rate in documents per second"
    )
    batch_count: int = Field(
        ...,
        ge=0,
        description="Number of batches processed"
    )
    avg_batch_time: float = Field(
        ...,
        ge=0.0,
        description="Average batch processing time in seconds"
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of errors encountered"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


class EmbeddingResult(BaseModel):
    """
    Result of embedding operation.

    This model represents the outcome of document embedding
    with comprehensive metrics and validation.
    """
    success: bool = Field(
        ...,
        description="Whether embedding operation was successful"
    )
    collection_name: str = Field(
        ...,
        min_length=1,
        description="Name of the vector collection"
    )
    metrics: EmbeddingMetrics = Field(
        ...,
        description="Performance metrics for the embedding operation"
    )
    dataset_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistics about the processed dataset"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When embedding operation was completed"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed"
    )
    hospital_profile_id: Optional[str] = Field(
        default=None,
        description="Hospital profile used for embedding"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }