"""
Configuration management for the MLflow Time Series Clustering Pipeline.

This module provides the PipelineConfig class that manages all configurable
parameters with validation for the entire pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import warnings


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

    # HospitalProfile integration
    hospital_profile_id: Optional[str] = None

    def __post_init__(self):
        """Validate configuration parameters and log warnings for suboptimal settings."""
        self._validate_hdbscan_params()
        self._validate_lgb_params()
        self._validate_knn_params()
        self._validate_pipeline_params()
        self._validate_mlflow_params()
        self._check_parameter_compatibility()

    def _validate_hdbscan_params(self):
        """Validate HDBSCAN parameters."""
        if not isinstance(self.hdbscan_params, dict):
            raise ValueError("hdbscan_params must be a dictionary")

        min_cluster_size = self.hdbscan_params.get('min_cluster_size', 5)
        min_samples = self.hdbscan_params.get('min_samples', 3)

        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")

        # Warning for suboptimal settings
        if min_cluster_size < 5:
            warnings.warn("min_cluster_size < 5 may result in many small clusters")
        if min_samples >= min_cluster_size:
            warnings.warn("min_samples >= min_cluster_size may result in conservative clustering")

    def _validate_lgb_params(self):
        """Validate LightGBM parameters."""
        for param_name, params in [
            ('lgb_timeseries_params', self.lgb_timeseries_params),
            ('lgb_resource_params', self.lgb_resource_params)
        ]:
            if not isinstance(params, dict):
                raise ValueError(f"{param_name} must be a dictionary")

            learning_rate = params.get('learning_rate', 0.1)
            num_leaves = params.get('num_leaves', 31)

            if not 0.001 <= learning_rate <= 1.0:
                raise ValueError(f"{param_name} learning_rate must be between 0.001 and 1.0")
            if num_leaves < 2:
                raise ValueError(f"{param_name} num_leaves must be at least 2")

            # Warning for suboptimal settings
            if learning_rate > 0.3:
                warnings.warn(f"{param_name} learning_rate > 0.3 may cause overfitting")
            if num_leaves > 100:
                warnings.warn(f"{param_name} num_leaves > 100 may cause overfitting")

    def _validate_knn_params(self):
        """Validate kNN parameters."""
        if not isinstance(self.knn_params, dict):
            raise ValueError("knn_params must be a dictionary")

        n_neighbors = self.knn_params.get('n_neighbors', 5)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")

        # Warning for suboptimal settings
        if n_neighbors > 20:
            warnings.warn("n_neighbors > 20 may reduce fallback model performance")

    def _validate_pipeline_params(self):
        """Validate pipeline behavior parameters."""
        if not 0.0 <= self.noise_threshold <= 1.0:
            raise ValueError("noise_threshold must be between 0.0 and 1.0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.expanding_window_size < self.batch_size:
            raise ValueError("expanding_window_size must be at least batch_size")

        # Warning for suboptimal settings
        if self.noise_threshold > 0.5:
            warnings.warn("noise_threshold > 0.5 may trigger frequent retraining")
        if self.batch_size > 10000:
            warnings.warn("batch_size > 10000 may cause memory issues")

    def _validate_mlflow_params(self):
        """Validate MLflow configuration parameters."""
        if not isinstance(self.experiment_name, str) or not self.experiment_name:
            raise ValueError("experiment_name must be a non-empty string")
        if not isinstance(self.model_registry_prefix, str) or not self.model_registry_prefix:
            raise ValueError("model_registry_prefix must be a non-empty string")

        if self.hospital_profile_id is not None:
            if not isinstance(self.hospital_profile_id, str) or not self.hospital_profile_id:
                raise ValueError("hospital_profile_id must be a non-empty string or None")

    def _check_parameter_compatibility(self):
        """Check for parameter combinations that may cause issues."""
        hdbscan_min_cluster = self.hdbscan_params.get('min_cluster_size', 5)
        knn_neighbors = self.knn_params.get('n_neighbors', 5)

        if knn_neighbors >= hdbscan_min_cluster:
            warnings.warn(
                "kNN n_neighbors >= HDBSCAN min_cluster_size may cause inconsistent predictions"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for MLflow logging."""
        return {
            'hdbscan_params': self.hdbscan_params,
            'lgb_timeseries_params': self.lgb_timeseries_params,
            'lgb_resource_params': self.lgb_resource_params,
            'knn_params': self.knn_params,
            'noise_threshold': self.noise_threshold,
            'batch_size': self.batch_size,
            'expanding_window_size': self.expanding_window_size,
            'experiment_name': self.experiment_name,
            'model_registry_prefix': self.model_registry_prefix,
            'hospital_profile_id': self.hospital_profile_id
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create PipelineConfig from dictionary."""
        return cls(**config_dict)