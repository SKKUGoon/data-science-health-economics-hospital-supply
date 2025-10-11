"""
Unified configuration management for the project.

This module provides the PipelineConfig class that manages all configurable
parameters with validation using Pydantic BaseModel for the entire pipeline.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import warnings


class PipelineConfig(BaseModel):
    """
    Unified configuration class for the MLflow Time Series Clustering Pipeline.

    This class manages all configurable parameters for HDBSCAN clustering,
    LightGBM models, kNN prediction, and pipeline behavior using Pydantic
    for automatic validation and serialization.

    **Configuration Categories:**

    1. **HDBSCAN Clustering Parameters** (`hdbscan_params`):
       - Controls density-based clustering behavior
       - Includes min_cluster_size, min_samples, cluster_selection_method
       - Automatically enables prediction_data for approximate predictions

    2. **LightGBM Model Parameters**:
       - `lgb_timeseries_params`: For time series prediction models
       - `lgb_resource_params`: For resource usage prediction models
       - Separate configurations allow different optimization strategies

    3. **kNN Prediction Parameters** (`knn_params`):
       - Assigns clusters to new data using the trained reference set
       - Configurable neighbors, weights, and distance metrics

    4. **Pipeline Behavior**:
       - `noise_threshold`: Triggers retraining when exceeded
       - `batch_size`: Controls memory usage and processing speed
       - `expanding_window_size`: Determines training data retention

    5. **MLflow Integration**:
       - `experiment_name`: Organizes runs in MLflow
       - `model_registry_prefix`: Consistent model naming

    **Validation Features:**
    - Automatic parameter validation with meaningful error messages
    - Range checking for numeric parameters
    - Compatibility warnings for parameter combinations
    - Type validation for all configuration fields

    **Usage Examples:**

    ```python
    # Basic configuration
    config = PipelineConfig()

    # Custom configuration with validation
    config = PipelineConfig(
        hdbscan_params={
            'min_cluster_size': 10,
            'min_samples': 5,
            'cluster_selection_epsilon': 0.1
        },
        noise_threshold=0.2,
        batch_size=2000,
    )

    # Get MLflow-compatible parameters
    mlflow_params = config.get_mlflow_params()

    # Create from dictionary
    config_dict = {...}
    config = PipelineConfig.from_dict(config_dict)
    ```

    **Integration with Other Components:**
    - Used by all models and pipeline components
    - Provides consistent parameter access across the system
    - Enables centralized configuration management
    - Supports configuration serialization for reproducibility
    """
    umap_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'n_components': 32,
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'n_jobs': -1,
        },
        description="UMAP dimensionality reduction parameters"
    )

    # HDBSCAN clustering parameters
    hdbscan_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'min_cluster_size': 5,
            'min_samples': 3,
            'cluster_selection_epsilon': 0.0,
            'prediction_data': False,
            'cluster_selection_method': 'eom'
        },
        description="HDBSCAN clustering algorithm parameters"
    )

    # kNN prediction parameters
    knn_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            'n_neighbors': 5,
            'weights': 'distance',
            'algorithm': 'auto',
            'metric': 'euclidean'
        },
        description="k-Nearest Neighbors prediction model parameters"
    )

    # LightGBM parameters for time series models
    lgb_timeseries_params: Dict[str, Any] = Field(
        default_factory=lambda: {
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
        },
        description="LightGBM parameters for time series prediction models"
    )

    # LightGBM parameters for resource usage models
    lgb_resource_params: Dict[str, Any] = Field(
        default_factory=lambda: {
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
        },
        description="LightGBM parameters for resource usage prediction models"
    )

    # Pipeline behavior parameters
    noise_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for noise ratio to trigger retraining"
    )

    batch_size: int = Field(
        default=1000,
        gt=0,
        description="Number of samples to process in each batch"
    )

    expanding_window_size: int = Field(
        default=10000,
        gt=0,
        description="Size of the expanding window for model training"
    )

    # MLflow configuration
    experiment_name: str = Field(
        default="TimeSeries-Clustering-Pipeline",
        min_length=1,
        description="MLflow experiment name"
    )

    model_registry_prefix: str = Field(
        default="timeseries-clustering",
        min_length=1,
        description="Prefix for model names in MLflow model registry"
    )

    # Additional configuration for enhanced functionality
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring and alerting"
    )

    enable_drift_detection: bool = Field(
        default=True,
        description="Enable data drift detection"
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level for the pipeline"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }

    @field_validator('hdbscan_params')
    @classmethod
    def validate_hdbscan_params(cls, v):
        """Validate HDBSCAN parameters."""
        if not isinstance(v, dict):
            raise ValueError("hdbscan_params must be a dictionary")

        min_cluster_size = v.get('min_cluster_size', 5)
        min_samples = v.get('min_samples', 3)

        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")

        # Warning for suboptimal settings
        if min_cluster_size < 5:
            warnings.warn("min_cluster_size < 5 may result in many small clusters")
        if min_samples >= min_cluster_size:
            warnings.warn("min_samples >= min_cluster_size may result in conservative clustering")

        return v

    @field_validator('lgb_timeseries_params', 'lgb_resource_params')
    @classmethod
    def validate_lgb_params(cls, v):
        """Validate LightGBM parameters."""
        if not isinstance(v, dict):
            raise ValueError("lgb_params must be a dictionary")

        learning_rate = v.get('learning_rate', 0.1)
        num_leaves = v.get('num_leaves', 31)

        if not 0.001 <= learning_rate <= 1.0:
            raise ValueError("lgb_params learning_rate must be between 0.001 and 1.0")
        if num_leaves < 2:
            raise ValueError("lgb_params num_leaves must be at least 2")

        # Warning for suboptimal settings
        if learning_rate > 0.3:
            warnings.warn("lgb_params learning_rate > 0.3 may cause overfitting")
        if num_leaves > 100:
            warnings.warn("lgb_params num_leaves > 100 may cause overfitting")

        return v

    @field_validator('knn_params')
    @classmethod
    def validate_knn_params(cls, v):
        """Validate kNN parameters."""
        if not isinstance(v, dict):
            raise ValueError("knn_params must be a dictionary")

        n_neighbors = v.get('n_neighbors', 5)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")

        # Warning for suboptimal settings
        if n_neighbors > 20:
            warnings.warn("n_neighbors > 20 may reduce fallback model performance")

        return v

    @field_validator('umap_params')
    @classmethod
    def validate_umap_params(cls, v):
        """Validate UMAP parameters."""
        if not isinstance(v, dict):
            raise ValueError("umap_params must be a dictionary")

        n_components = v.get('n_components', 128)
        if n_components < 2:
            raise ValueError("umap_params n_components must be at least 2")

        n_neighbors = v.get('n_neighbors', 15)
        if n_neighbors < 2:
            raise ValueError("umap_params n_neighbors must be at least 2")

        min_dist = v.get('min_dist', 0.1)
        if not 0.0 <= min_dist <= 1.0:
            raise ValueError("umap_params min_dist must be between 0 and 1")

        n_jobs = v.get('n_jobs', None)
        if n_jobs is not None:
            if not isinstance(n_jobs, int):
                raise ValueError("umap_params n_jobs must be an integer")
            if n_jobs == 0 or n_jobs < -1:
                raise ValueError("umap_params n_jobs must be -1 (all cores) or a positive integer")

        return v

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @model_validator(mode='after')
    def validate_expanding_window_size(self):
        """Validate that expanding_window_size is at least batch_size."""
        if self.expanding_window_size < self.batch_size:
            raise ValueError("expanding_window_size must be at least batch_size")
        return self

    def check_parameter_compatibility(self):
        """Check for parameter combinations that may cause issues."""
        hdbscan_min_cluster = self.hdbscan_params.get('min_cluster_size', 5)
        knn_neighbors = self.knn_params.get('n_neighbors', 5)


        # Additional compatibility checks
        if knn_neighbors >= hdbscan_min_cluster:
            warnings.warn(
                "kNN n_neighbors >= HDBSCAN min_cluster_size may cause inconsistent predictions"
            )

        if self.noise_threshold > 0.5:
            warnings.warn("noise_threshold > 0.5 may trigger frequent retraining")
        if self.batch_size > 10000:
            warnings.warn("batch_size > 10000 may cause memory issues")
        umap_neighbors = self.umap_params.get('n_neighbors', 15)
        if umap_neighbors > 100:
            warnings.warn("umap_params n_neighbors > 100 may lead to very smooth manifolds and slow training")

    def get_mlflow_params(self) -> Dict[str, Any]:
        """Get parameters suitable for MLflow logging."""
        return {
            'hdbscan_params': self.hdbscan_params,
            'umap_params': self.umap_params,
            'lgb_timeseries_params': self.lgb_timeseries_params,
            'lgb_resource_params': self.lgb_resource_params,
            'knn_params': self.knn_params,
            'noise_threshold': self.noise_threshold,
            'batch_size': self.batch_size,
            'expanding_window_size': self.expanding_window_size,
            'experiment_name': self.experiment_name,
            'model_registry_prefix': self.model_registry_prefix,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_drift_detection': self.enable_drift_detection,
            'log_level': self.log_level
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create PipelineConfig from dictionary."""
        return cls(**config_dict)

    def model_post_init(self, __context) -> None:
        """Post-initialization validation and warnings."""
        self.check_parameter_compatibility()
