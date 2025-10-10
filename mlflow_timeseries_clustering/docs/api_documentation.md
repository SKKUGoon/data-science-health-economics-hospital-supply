# MLflow Time Series Clustering Pipeline - API Documentation

This document provides comprehensive API documentation for all classes, methods, and functions in the MLflow Time Series Clustering Pipeline.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Configuration Management](#configuration-management)
3. [Data Models](#data-models)
4. [Pipeline Controller](#pipeline-controller)
5. [Clustering Components](#clustering-components)
6. [Model Management](#model-management)
7. [MLflow Integration](#mlflow-integration)
8. [Monitoring and Retraining](#monitoring-and-retraining)
9. [Error Handling](#error-handling)
10. [Utilities and Helpers](#utilities-and-helpers)

## Core Classes

### TimeSeriesClusteringPipeline

The main pipeline controller that orchestrates the entire clustering and prediction workflow.

```python
class TimeSeriesClusteringPipeline:
    """
    Main pipeline controller for MLflow Time Series Clustering.
    
    This class orchestrates the complete pipeline lifecycle:
    1. Initial training phase with HDBSCAN clustering and LightGBM models
    2. Streaming processing phase with batch processing and drift monitoring
    3. Adaptive retraining phase with automatic model updates
    """
```

#### Constructor

```python
def __init__(self, config: PipelineConfig) -> None:
    """
    Initialize the Time Series Clustering Pipeline.
    
    Args:
        config: Pipeline configuration containing all parameters
        
    Raises:
        PipelineError: If initialization fails
        
    Example:
        >>> config = PipelineConfig(noise_threshold=0.25)
        >>> pipeline = TimeSeriesClusteringPipeline(config)
    """
```

#### Core Methods

##### fit_initial

```python
def fit_initial(self, X: np.ndarray, y: np.ndarray) -> ClusteringResult:
    """
    Perform initial training of the complete pipeline.
    
    This method:
    1. Validates input data
    2. Trains HDBSCAN clustering model
    3. Fits cluster-specific LightGBM models
    4. Sets up kNN fallback mechanism
    5. Registers models in MLflow
    
    Args:
        X: Training features, shape (n_samples, n_features)
        y: Training targets, shape (n_samples,)
        
    Returns:
        ClusteringResult containing cluster labels, metrics, and metadata
        
    Raises:
        DataValidationError: If input data is invalid
        ModelTrainingError: If model training fails
        MLflowIntegrationError: If MLflow operations fail
        
    Example:
        >>> X = np.random.randn(1000, 10)
        >>> y = np.random.randn(1000)
        >>> result = pipeline.fit_initial(X, y)
        >>> print(f"Found {result.metrics['n_clusters']} clusters")
    """
```

##### process_batch

```python
def process_batch(self, X_batch: np.ndarray, 
                 y_batch: Optional[np.ndarray] = None) -> BatchResult:
    """
    Process a new data batch using the trained pipeline.
    
    This method:
    1. Scales input features
    2. Assigns cluster labels using HDBSCAN prediction
    3. Falls back to kNN for noise points
    4. Makes time series predictions
    5. Predicts resource usage
    6. Monitors for drift and triggers retraining if needed
    
    Args:
        X_batch: Batch features, shape (batch_size, n_features)
        y_batch: Optional batch targets, shape (batch_size,)
        
    Returns:
        BatchResult containing predictions, assignments, and metrics
        
    Raises:
        PipelineError: If pipeline is not fitted
        DataValidationError: If batch data is invalid
        PredictionError: If prediction fails
        
    Example:
        >>> X_batch = np.random.randn(100, 10)
        >>> result = pipeline.process_batch(X_batch)
        >>> print(f"Noise ratio: {result.noise_ratio:.3f}")
    """
```

##### retrain

```python
def retrain(self, trigger_reason: str = "Manual request") -> None:
    """
    Manually trigger pipeline retraining.
    
    This method:
    1. Collects historical data from expanding window
    2. Refits HDBSCAN clustering model
    3. Retrains all cluster-specific models
    4. Updates model registry with new versions
    5. Generates comparison reports
    
    Args:
        trigger_reason: Reason for triggering retraining
        
    Raises:
        PipelineError: If pipeline is not fitted
        RetrainingError: If retraining fails
        MLflowIntegrationError: If MLflow operations fail
        
    Example:
        >>> pipeline.retrain("High noise ratio detected")
    """
```

##### get_performance_report

```python
def get_performance_report(self) -> PerformanceReport:
    """
    Generate comprehensive performance report for the pipeline.
    
    Returns:
        PerformanceReport containing:
        - Clustering metrics (silhouette score, cluster count, etc.)
        - Model performance metrics per cluster
        - Visualizations and plots
        - Timestamp and model version information
        
    Raises:
        PipelineError: If pipeline is not fitted
        
    Example:
        >>> report = pipeline.get_performance_report()
        >>> print(f"Model version: {report.model_version}")
        >>> print(f"Clusters: {report.clustering_metrics['n_clusters']}")
    """
```

#### Properties

```python
@property
def is_fitted(self) -> bool:
    """Check if the pipeline is fitted and ready for use."""

@property
def is_batch_processing_enabled(self) -> bool:
    """Check if batch processing is currently enabled."""

@property
def current_model_version(self) -> str:
    """Get the current model version."""

@property
def last_performance_report(self) -> Optional[PerformanceReport]:
    """Get the last generated performance report."""

@property
def current_health_status(self) -> str:
    """Get current pipeline health status."""

@property
def monitoring_statistics(self) -> Dict[str, Any]:
    """Get performance monitoring statistics."""
```

#### Utility Methods

```python
def set_callbacks(self,
                 pre_training: Optional[Callable[[], None]] = None,
                 post_training: Optional[Callable[[ClusteringResult], None]] = None,
                 pre_batch: Optional[Callable[[np.ndarray], None]] = None,
                 post_batch: Optional[Callable[[BatchResult], None]] = None) -> None:
    """
    Set callback functions for pipeline events.
    
    Args:
        pre_training: Called before initial training starts
        post_training: Called after initial training completes
        pre_batch: Called before each batch is processed
        post_batch: Called after each batch is processed
    """

def get_pipeline_status(self) -> Dict[str, Any]:
    """
    Get comprehensive pipeline status information.
    
    Returns:
        Dictionary containing:
        - pipeline_fitted: Whether pipeline is trained
        - batch_processing_enabled: Whether batch processing is active
        - current_model_version: Current model version
        - last_training_time: Timestamp of last training
        - performance_metrics: Recent performance metrics
    """

def get_model_info(self) -> Dict[str, Any]:
    """
    Get information about the current models.
    
    Returns:
        Dictionary containing:
        - clustering: HDBSCAN model information
        - timeseries_models: LightGBM time series model info per cluster
        - resource_models: LightGBM resource model info per cluster
        - fallback_model: kNN fallback model information
    """

def update_configuration(self, new_config: PipelineConfig) -> None:
    """
    Update pipeline configuration.
    
    Args:
        new_config: New configuration to apply
        
    Note:
        Some parameters require retraining to take effect.
    """

def reset_pipeline_state(self) -> None:
    """
    Reset pipeline state (useful for testing or reinitialization).
    
    Warning:
        This will clear all trained models and require retraining.
    """
```

#### Error Handling Methods

```python
def get_error_statistics(self) -> Dict[str, Any]:
    """
    Get comprehensive error statistics from the error handler.
    
    Returns:
        Dictionary containing error counts, types, and recent errors
    """

def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent error records for debugging.
    
    Args:
        limit: Maximum number of recent errors to return
        
    Returns:
        List of error records with timestamps and details
    """

def enable_graceful_degradation(self, enable: bool = True) -> None:
    """
    Enable or disable graceful degradation for error handling.
    
    Args:
        enable: Whether to enable graceful degradation
    """

def diagnose_pipeline_health(self) -> Dict[str, Any]:
    """
    Perform comprehensive pipeline health diagnosis.
    
    Returns:
        Dictionary containing:
        - pipeline_status: Overall pipeline status
        - component_health: Health status of each component
        - performance_metrics: Recent performance indicators
        - recommendations: List of recommended actions
    """
```

#### Performance Monitoring Methods

```python
def get_performance_metrics(self, 
                          metric_names: Optional[List[str]] = None,
                          time_window: Optional[timedelta] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get performance metrics from the performance monitor.
    
    Args:
        metric_names: Specific metrics to retrieve (None for all)
        time_window: Time window for metrics (None for all time)
        
    Returns:
        Dictionary mapping metric names to lists of metric records
    """

def get_alert_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
    """
    Get alert summary from the performance monitor.
    
    Args:
        time_window: Time window for alerts (None for all time)
        
    Returns:
        Dictionary containing alert counts and summaries
    """

def perform_health_check(self) -> Dict[str, Any]:
    """
    Perform comprehensive pipeline health check.
    
    Returns:
        Dictionary containing health status and recommendations
    """

def update_performance_thresholds(self, 
                                alert_thresholds: Optional[Dict[str, float]] = None,
                                drift_thresholds: Optional[Dict[str, float]] = None) -> None:
    """
    Update performance monitoring thresholds.
    
    Args:
        alert_thresholds: New alert thresholds
        drift_thresholds: New drift detection thresholds
    """

def export_performance_data(self, file_path: str) -> None:
    """
    Export performance monitoring data to file.
    
    Args:
        file_path: Path to save performance data
    """
```

## Configuration Management

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    """
    Configuration class for the MLflow Time Series Clustering Pipeline.
    
    This class manages all configurable parameters for HDBSCAN clustering,
    LightGBM models, kNN fallback, and pipeline behavior.
    """
    
    # HDBSCAN clustering parameters
    hdbscan_params: Dict[str, Any] = field(default_factory=lambda: {
        'min_cluster_size': 5,
        'min_samples': 3,
        'cluster_selection_epsilon': 0.0,
        'prediction_data': True,
        'cluster_selection_method': 'eom'
    })
    
    # LightGBM parameters for time series models
    lgb_timeseries_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    })
    
    # LightGBM parameters for resource usage models
    lgb_resource_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    })
    
    # kNN fallback parameters
    knn_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_neighbors': 5,
        'weights': 'distance',
        'algorithm': 'auto',
        'metric': 'euclidean'
    })
    
    # Pipeline behavior parameters
    noise_threshold: float = 0.3
    batch_size: int = 1000
    expanding_window_size: int = 10000
    
    # MLflow configuration
    experiment_name: str = "TimeSeries-Clustering-Pipeline"
    model_registry_prefix: str = "timeseries-clustering"
    hospital_profile_id: Optional[str] = None
```

#### Methods

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert configuration to dictionary for MLflow logging."""

@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
    """Create PipelineConfig from dictionary."""
```

## Data Models

### ClusteringResult

```python
@dataclass
class ClusteringResult:
    """
    Result of clustering operation.
    
    Attributes:
        labels: Cluster labels for each sample
        cluster_centers: Coordinates of cluster centers
        noise_ratio: Ratio of noise points to total points
        metrics: Dictionary of clustering quality metrics
    """
    labels: np.ndarray
    cluster_centers: np.ndarray
    noise_ratio: float
    metrics: Dict[str, Any]
```

### BatchResult

```python
@dataclass
class BatchResult:
    """
    Result of batch processing operation.
    
    Attributes:
        cluster_assignments: Cluster assignments for batch samples
        timeseries_predictions: Time series predictions for each sample
        resource_predictions: Resource usage predictions per cluster
        noise_ratio: Ratio of noise points in the batch
        processing_time: Time taken to process the batch
    """
    cluster_assignments: np.ndarray
    timeseries_predictions: np.ndarray
    resource_predictions: Dict[int, Dict[str, float]]
    noise_ratio: float
    processing_time: float
```

### PerformanceReport

```python
@dataclass
class PerformanceReport:
    """
    Comprehensive performance report.
    
    Attributes:
        clustering_metrics: Clustering quality metrics
        model_metrics: Model performance metrics per cluster
        visualizations: Dictionary of visualization file paths
        timestamp: Report generation timestamp
        model_version: Model version at report generation
    """
    clustering_metrics: Dict[str, Any]
    model_metrics: Dict[int, Dict[str, float]]
    visualizations: Dict[str, str]
    timestamp: datetime
    model_version: str
```

## Clustering Components

### AdaptiveClusteringEngine

```python
class AdaptiveClusteringEngine:
    """
    Handles HDBSCAN clustering with kNN fallback and noise detection.
    
    This class manages the primary clustering algorithm and provides
    fallback mechanisms for noise point handling.
    """
```

#### Constructor

```python
def __init__(self, hdbscan_params: Dict[str, Any], knn_params: Dict[str, Any]) -> None:
    """
    Initialize the adaptive clustering engine.
    
    Args:
        hdbscan_params: Parameters for HDBSCAN clustering
        knn_params: Parameters for kNN fallback model
    """
```

#### Methods

```python
def fit(self, X: np.ndarray) -> ClusteringResult:
    """
    Fit the clustering model on training data.
    
    Args:
        X: Training features, shape (n_samples, n_features)
        
    Returns:
        ClusteringResult with cluster labels and metrics
    """

def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Predict cluster assignments for new data.
    
    Args:
        X: New data features, shape (n_samples, n_features)
        
    Returns:
        Cluster assignments, shape (n_samples,)
    """

def get_clustering_metrics(self) -> Dict[str, Any]:
    """
    Get clustering quality metrics.
    
    Returns:
        Dictionary containing silhouette score, cluster count, etc.
    """

def update_fallback_model(self, X: np.ndarray, labels: np.ndarray) -> None:
    """
    Update the kNN fallback model with new data.
    
    Args:
        X: Training features for fallback model
        labels: Cluster labels for training data
    """
```

### KNNFallback

```python
class KNNFallback:
    """
    k-Nearest Neighbors fallback model for noise point handling.
    
    This class provides cluster assignment for points that HDBSCAN
    cannot confidently assign to any cluster.
    """
```

#### Methods

```python
def fit(self, X: np.ndarray, labels: np.ndarray) -> None:
    """
    Fit the kNN fallback model.
    
    Args:
        X: Training features
        labels: Cluster labels from HDBSCAN
    """

def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Predict cluster assignments using kNN.
    
    Args:
        X: Features to predict
        
    Returns:
        Predicted cluster assignments
    """

def get_fallback_metrics(self) -> Dict[str, Any]:
    """
    Get fallback model performance metrics.
    
    Returns:
        Dictionary containing accuracy and usage statistics
    """
```

## Model Management

### ClusterSpecificModelManager

```python
class ClusterSpecificModelManager:
    """
    Manages cluster-specific LightGBM models for time series and resource prediction.
    
    This class handles training, prediction, and management of separate
    models for each cluster identified by the clustering engine.
    """
```

#### Constructor

```python
def __init__(self, lgb_params: Dict[str, Any]) -> None:
    """
    Initialize the cluster-specific model manager.
    
    Args:
        lgb_params: Parameters for LightGBM models
    """
```

#### Methods

```python
def fit_timeseries_models(self, cluster_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[str, float]]:
    """
    Fit time series prediction models for each cluster.
    
    Args:
        cluster_data: Dictionary mapping cluster IDs to data
        
    Returns:
        Dictionary mapping cluster IDs to performance metrics
    """

def fit_resource_models(self, cluster_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[str, float]]:
    """
    Fit resource usage prediction models for each cluster.
    
    Args:
        cluster_data: Dictionary mapping cluster IDs to data
        
    Returns:
        Dictionary mapping cluster IDs to performance metrics
    """

def predict_timeseries(self, cluster_id: int, X: np.ndarray) -> np.ndarray:
    """
    Make time series predictions for a specific cluster.
    
    Args:
        cluster_id: ID of the cluster
        X: Features for prediction
        
    Returns:
        Time series predictions
    """

def predict_resources(self, cluster_id: int, X: np.ndarray) -> Dict[str, float]:
    """
    Predict resource usage for a specific cluster.
    
    Args:
        cluster_id: ID of the cluster
        X: Features for prediction
        
    Returns:
        Dictionary of resource usage predictions
    """

def get_model_metrics(self) -> Dict[int, Dict[str, float]]:
    """
    Get performance metrics for all models.
    
    Returns:
        Dictionary mapping cluster IDs to model metrics
    """
```

## MLflow Integration

### ExperimentManager

```python
class ExperimentManager:
    """
    Manages MLflow experiments and runs for the pipeline.
    
    This class handles experiment creation, run management, and
    parameter/metric logging for the entire pipeline.
    """
```

#### Methods

```python
def create_experiment(self, experiment_name: str) -> str:
    """
    Create or get MLflow experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Experiment ID
    """

def start_run(self, run_name: str, nested: bool = False) -> str:
    """
    Start a new MLflow run.
    
    Args:
        run_name: Name of the run
        nested: Whether this is a nested run
        
    Returns:
        Run ID
    """

def log_params(self, params: Dict[str, Any]) -> None:
    """
    Log parameters to current MLflow run.
    
    Args:
        params: Dictionary of parameters to log
    """

def log_metrics(self, metrics: Dict[str, float]) -> None:
    """
    Log metrics to current MLflow run.
    
    Args:
        metrics: Dictionary of metrics to log
    """

def log_artifact(self, artifact_path: str, artifact_name: str) -> None:
    """
    Log artifact to current MLflow run.
    
    Args:
        artifact_path: Local path to artifact
        artifact_name: Name for artifact in MLflow
    """

def end_run(self) -> None:
    """End the current MLflow run."""
```

### ArtifactManager

```python
class ArtifactManager:
    """
    Manages MLflow artifacts including model serialization and storage.
    
    This class handles saving and loading of models, reports, and
    other artifacts with proper organization by hospital profile.
    """
```

#### Methods

```python
def save_model(self, model: Any, model_name: str, model_type: str) -> str:
    """
    Save a model as MLflow artifact.
    
    Args:
        model: Model object to save
        model_name: Name for the model
        model_type: Type of model (clustering, timeseries, resource)
        
    Returns:
        Artifact path
    """

def load_model(self, model_name: str, model_type: str) -> Any:
    """
    Load a model from MLflow artifacts.
    
    Args:
        model_name: Name of the model
        model_type: Type of model
        
    Returns:
        Loaded model object
    """

def save_report(self, report: Dict[str, Any], report_name: str) -> str:
    """
    Save a report as MLflow artifact.
    
    Args:
        report: Report data to save
        report_name: Name for the report
        
    Returns:
        Artifact path
    """

def get_artifact_path(self, artifact_name: str) -> str:
    """
    Get the full path for an artifact.
    
    Args:
        artifact_name: Name of the artifact
        
    Returns:
        Full artifact path
    """
```

## Monitoring and Retraining

### AdaptiveRetrainingManager

```python
class AdaptiveRetrainingManager:
    """
    Manages adaptive retraining based on performance monitoring and drift detection.
    
    This class monitors pipeline performance and triggers retraining
    when performance degrades or data drift is detected.
    """
```

#### Methods

```python
def monitor_batch_result(self, batch_result: BatchResult, 
                        batch_data: Dict[str, np.ndarray]) -> bool:
    """
    Monitor batch result and trigger retraining if needed.
    
    Args:
        batch_result: Result of batch processing
        batch_data: Raw batch data for potential retraining
        
    Returns:
        True if retraining was triggered, False otherwise
    """

def request_manual_retraining(self, reason: str) -> bool:
    """
    Request manual retraining.
    
    Args:
        reason: Reason for manual retraining
        
    Returns:
        True if retraining was initiated, False otherwise
    """

def check_scheduled_retraining(self, schedule_interval: timedelta) -> bool:
    """
    Check if scheduled retraining is due.
    
    Args:
        schedule_interval: Interval between scheduled retrainings
        
    Returns:
        True if retraining was triggered, False otherwise
    """

def get_retraining_status(self) -> Dict[str, Any]:
    """
    Get current retraining status and statistics.
    
    Returns:
        Dictionary containing retraining statistics and status
    """

def get_retraining_recommendations(self) -> List[str]:
    """
    Get recommendations for retraining optimization.
    
    Returns:
        List of recommendation strings
    """
```

### PerformanceMonitor

```python
class PerformanceMonitor:
    """
    Monitors pipeline performance and generates alerts.
    
    This class tracks various performance metrics and generates
    alerts when thresholds are exceeded or trends are detected.
    """
```

#### Methods

```python
def record_metric(self, metric_name: str, value: float, 
                 component: str = "pipeline") -> None:
    """
    Record a performance metric.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        component: Component that generated the metric
    """

def check_alerts(self) -> List[PerformanceAlert]:
    """
    Check for performance alerts.
    
    Returns:
        List of active performance alerts
    """

def get_performance_summary(self) -> Dict[str, Any]:
    """
    Get performance summary statistics.
    
    Returns:
        Dictionary containing performance summaries
    """

def detect_drift(self, metric_name: str, window_size: int = 10) -> bool:
    """
    Detect drift in a performance metric.
    
    Args:
        metric_name: Name of metric to check for drift
        window_size: Size of window for drift detection
        
    Returns:
        True if drift is detected, False otherwise
    """
```

## Error Handling

### Custom Exceptions

```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class DataValidationError(PipelineError):
    """Raised when input data validation fails."""
    pass

class ModelTrainingError(PipelineError):
    """Raised when model training fails."""
    pass

class PredictionError(PipelineError):
    """Raised when prediction fails."""
    pass

class MLflowIntegrationError(PipelineError):
    """Raised when MLflow operations fail."""
    pass

class RetrainingError(PipelineError):
    """Raised when retraining fails."""
    pass
```

### ErrorHandler

```python
class ErrorHandler:
    """
    Handles errors and provides graceful degradation capabilities.
    
    This class manages error logging, recovery strategies, and
    graceful degradation when components fail.
    """
```

#### Methods

```python
def handle_error(self, error: Exception, context: str) -> None:
    """
    Handle an error with appropriate logging and recovery.
    
    Args:
        error: Exception that occurred
        context: Context where error occurred
    """

def get_error_statistics(self) -> Dict[str, Any]:
    """
    Get error statistics and summaries.
    
    Returns:
        Dictionary containing error counts and statistics
    """

def clear_error_history(self) -> None:
    """Clear error history for maintenance."""

def enable_graceful_degradation(self, enable: bool = True) -> None:
    """
    Enable or disable graceful degradation.
    
    Args:
        enable: Whether to enable graceful degradation
    """
```

## Utilities and Helpers

### Validation Functions

```python
def validate_input_data(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    """
    Validate input data format and content.
    
    Args:
        X: Feature matrix to validate
        y: Optional target vector to validate
        
    Raises:
        DataValidationError: If data is invalid
    """

def validate_batch_data(X_batch: np.ndarray, expected_features: int) -> None:
    """
    Validate batch data format.
    
    Args:
        X_batch: Batch data to validate
        expected_features: Expected number of features
        
    Raises:
        DataValidationError: If batch data is invalid
    """
```

### Metric Calculation Functions

```python
def calculate_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate clustering quality metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Dictionary of clustering metrics
    """

def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate model performance metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of performance metrics
    """
```

### Visualization Functions

```python
def create_cluster_visualization(X: np.ndarray, labels: np.ndarray, 
                               save_path: str) -> str:
    """
    Create cluster visualization plot.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        save_path: Path to save visualization
        
    Returns:
        Path to saved visualization
    """

def create_performance_plots(metrics_history: List[Dict[str, float]], 
                           save_path: str) -> str:
    """
    Create performance trend plots.
    
    Args:
        metrics_history: Historical metrics data
        save_path: Path to save plots
        
    Returns:
        Path to saved plots
    """
```

## Usage Examples

### Basic Pipeline Usage

```python
from mlflow_timeseries_clustering.core.config import PipelineConfig
from mlflow_timeseries_clustering.pipeline.timeseries_clustering_pipeline import TimeSeriesClusteringPipeline

# Create configuration
config = PipelineConfig(
    noise_threshold=0.25,
    batch_size=500,
    experiment_name="my-clustering-experiment"
)

# Initialize pipeline
pipeline = TimeSeriesClusteringPipeline(config)

# Train pipeline
X_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000)
clustering_result = pipeline.fit_initial(X_train, y_train)

# Process new batch
X_batch = np.random.randn(100, 10)
batch_result = pipeline.process_batch(X_batch)

# Get performance report
report = pipeline.get_performance_report()
```

### Advanced Configuration

```python
# Custom configuration for medical data
medical_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 25,
        'min_samples': 10,
        'cluster_selection_epsilon': 0.1
    },
    lgb_timeseries_params={
        'num_leaves': 50,
        'learning_rate': 0.04,
        'n_estimators': 150
    },
    noise_threshold=0.2,
    experiment_name="medical-patient-clustering",
    hospital_profile_id="hospital_001"
)

pipeline = TimeSeriesClusteringPipeline(medical_config)
```

### Error Handling

```python
try:
    result = pipeline.process_batch(X_batch)
except DataValidationError as e:
    print(f"Data validation failed: {e}")
except PredictionError as e:
    print(f"Prediction failed: {e}")
except PipelineError as e:
    print(f"Pipeline error: {e}")
```

### Performance Monitoring

```python
# Set up performance monitoring
pipeline.update_performance_thresholds(
    alert_thresholds={'noise_ratio': 0.3, 'processing_time': 5.0},
    drift_thresholds={'noise_ratio_trend': 0.05}
)

# Get performance metrics
metrics = pipeline.get_performance_metrics(['noise_ratio', 'processing_time'])
alerts = pipeline.get_alert_summary()
health = pipeline.perform_health_check()
```

This API documentation provides comprehensive information about all classes, methods, and functions available in the MLflow Time Series Clustering Pipeline, along with usage examples and best practices.