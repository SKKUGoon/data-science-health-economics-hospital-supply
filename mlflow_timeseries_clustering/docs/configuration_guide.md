# MLflow Time Series Clustering Pipeline - Configuration Guide

This comprehensive guide covers all configurable parameters for the MLflow Time Series Clustering Pipeline, including parameter tuning guidelines, best practices, and troubleshooting information.

## Table of Contents

1. [Overview](#overview)
2. [Core Configuration Parameters](#core-configuration-parameters)
3. [HDBSCAN Clustering Parameters](#hdbscan-clustering-parameters)
4. [LightGBM Model Parameters](#lightgbm-model-parameters)
5. [kNN Fallback Parameters](#knn-fallback-parameters)
6. [Pipeline Behavior Parameters](#pipeline-behavior-parameters)
7. [MLflow Integration Parameters](#mlflow-integration-parameters)
8. [Parameter Tuning Guidelines](#parameter-tuning-guidelines)
9. [Configuration Examples](#configuration-examples)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Performance Optimization](#performance-optimization)

## Overview

The `PipelineConfig` class manages all configurable parameters for the MLflow Time Series Clustering Pipeline. It provides validation, default values, and warnings for suboptimal settings.

### Basic Usage

```python
from mlflow_timeseries_clustering.core.config import PipelineConfig

# Use default configuration
config = PipelineConfig()

# Customize specific parameters
config = PipelineConfig(
    noise_threshold=0.25,
    batch_size=500,
    experiment_name="my-clustering-experiment"
)

# Load from dictionary
config_dict = {...}  # Your configuration dictionary
config = PipelineConfig.from_dict(config_dict)
```

## Core Configuration Parameters

### PipelineConfig Class Structure

```python
@dataclass
class PipelineConfig:
    # HDBSCAN clustering parameters
    hdbscan_params: Dict[str, Any]
    
    # LightGBM parameters for time series models
    lgb_timeseries_params: Dict[str, Any]
    
    # LightGBM parameters for resource usage models
    lgb_resource_params: Dict[str, Any]
    
    # kNN fallback parameters
    knn_params: Dict[str, Any]
    
    # Pipeline behavior parameters
    noise_threshold: float = 0.3
    batch_size: int = 1000
    expanding_window_size: int = 10000
    
    # MLflow configuration
    experiment_name: str = "TimeSeries-Clustering-Pipeline"
    model_registry_prefix: str = "timeseries-clustering"
    hospital_profile_id: Optional[str] = None
```

## HDBSCAN Clustering Parameters

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is the primary clustering algorithm used in the pipeline.

### Default Parameters

```python
hdbscan_params = {
    'min_cluster_size': 5,           # Minimum size of clusters
    'min_samples': 3,                # Minimum samples in neighborhood
    'cluster_selection_epsilon': 0.0, # Distance threshold for cluster selection
    'prediction_data': True,         # Enable prediction for new data
    'cluster_selection_method': 'eom' # Excess of mass method
}
```

### Parameter Descriptions

#### `min_cluster_size` (int, default: 5)
- **Description**: Minimum number of samples in a cluster
- **Range**: 2 to N (where N is your dataset size)
- **Impact**: 
  - Lower values: More, smaller clusters
  - Higher values: Fewer, larger clusters
- **Recommendations**:
  - Small datasets (< 1000 samples): 5-15
  - Medium datasets (1000-10000 samples): 15-50
  - Large datasets (> 10000 samples): 50-200

```python
# Example configurations
small_dataset_config = {'min_cluster_size': 10}
medium_dataset_config = {'min_cluster_size': 25}
large_dataset_config = {'min_cluster_size': 100}
```

#### `min_samples` (int, default: 3)
- **Description**: Minimum samples in a neighborhood for a point to be considered core
- **Range**: 1 to min_cluster_size
- **Impact**:
  - Lower values: More liberal clustering, more noise points become core
  - Higher values: More conservative clustering, stricter core point requirements
- **Recommendations**:
  - Conservative clustering: min_samples = min_cluster_size // 2
  - Balanced clustering: min_samples = min_cluster_size // 3
  - Liberal clustering: min_samples = min_cluster_size // 5

```python
# Conservative clustering
conservative_params = {
    'min_cluster_size': 30,
    'min_samples': 15  # 30 // 2
}

# Balanced clustering
balanced_params = {
    'min_cluster_size': 30,
    'min_samples': 10  # 30 // 3
}
```

#### `cluster_selection_epsilon` (float, default: 0.0)
- **Description**: Distance threshold for cluster selection
- **Range**: 0.0 to infinity
- **Impact**:
  - 0.0: Use standard HDBSCAN cluster selection
  - > 0.0: Flatten hierarchy at this distance, similar to DBSCAN
- **Recommendations**:
  - Standard use: 0.0
  - Need DBSCAN-like behavior: 0.1-0.5
  - High noise tolerance: 0.1-0.3

#### `prediction_data` (bool, default: True)
- **Description**: Whether to store data for predicting cluster membership of new points
- **Impact**: Required for batch processing new data
- **Recommendation**: Always True for production pipelines

#### `cluster_selection_method` (str, default: 'eom')
- **Description**: Method for selecting clusters from hierarchy
- **Options**:
  - 'eom': Excess of Mass (recommended for most cases)
  - 'leaf': Select leaf clusters (more sensitive to outliers)
- **Recommendations**:
  - Stable clustering: 'eom'
  - Outlier-sensitive clustering: 'leaf'

### HDBSCAN Tuning Examples

```python
# High-quality patient clustering
medical_hdbscan_params = {
    'min_cluster_size': 25,          # Meaningful patient groups
    'min_samples': 10,               # Conservative clustering
    'cluster_selection_epsilon': 0.1, # Some noise tolerance
    'prediction_data': True,
    'cluster_selection_method': 'eom',
    'metric': 'euclidean'
}

# Fast processing with larger clusters
fast_hdbscan_params = {
    'min_cluster_size': 50,          # Larger clusters
    'min_samples': 20,               # More conservative
    'cluster_selection_epsilon': 0.0,
    'prediction_data': True,
    'cluster_selection_method': 'eom'
}

# Sensitive to small patterns
sensitive_hdbscan_params = {
    'min_cluster_size': 8,           # Smaller clusters
    'min_samples': 3,                # Less conservative
    'cluster_selection_epsilon': 0.2, # More noise tolerance
    'prediction_data': True,
    'cluster_selection_method': 'leaf'
}
```

## LightGBM Model Parameters

The pipeline uses LightGBM for both time series prediction and resource usage modeling.

### Time Series Model Parameters

Default configuration for predicting target values:

```python
lgb_timeseries_params = {
    'objective': 'regression',       # Regression task
    'metric': 'rmse',               # Root Mean Square Error
    'boosting_type': 'gbdt',        # Gradient Boosting Decision Tree
    'num_leaves': 31,               # Number of leaves in trees
    'learning_rate': 0.05,          # Learning rate
    'feature_fraction': 0.9,        # Fraction of features to use
    'bagging_fraction': 0.8,        # Fraction of data to use
    'bagging_freq': 5,              # Frequency of bagging
    'verbose': -1,                  # Suppress output
    'random_state': 42              # Reproducibility
}
```

### Resource Usage Model Parameters

Default configuration for predicting computational resources:

```python
lgb_resource_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 15,               # Simpler model for resources
    'learning_rate': 0.1,           # Faster learning
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
```

### Key LightGBM Parameters

#### `objective` (str, default: 'regression')
- **Options**: 'regression', 'regression_l1', 'huber', 'fair'
- **Recommendations**:
  - Standard regression: 'regression'
  - Robust to outliers: 'huber'
  - L1 loss: 'regression_l1'

#### `num_leaves` (int, default: 31 for timeseries, 15 for resource)
- **Description**: Maximum number of leaves in one tree
- **Range**: 2 to 131072
- **Impact**:
  - Lower values: Simpler model, less overfitting
  - Higher values: More complex model, potential overfitting
- **Recommendations**:
  - Small datasets: 10-31
  - Medium datasets: 31-63
  - Large datasets: 63-127

#### `learning_rate` (float, default: 0.05 for timeseries, 0.1 for resource)
- **Description**: Shrinkage rate for boosting
- **Range**: 0.001 to 1.0
- **Impact**:
  - Lower values: Slower learning, better generalization
  - Higher values: Faster learning, potential overfitting
- **Recommendations**:
  - High accuracy needed: 0.01-0.05
  - Balanced performance: 0.05-0.1
  - Fast training: 0.1-0.3

#### `feature_fraction` (float, default: 0.9 for timeseries, 0.8 for resource)
- **Description**: Fraction of features to use in each iteration
- **Range**: 0.1 to 1.0
- **Impact**: Helps prevent overfitting and speeds up training
- **Recommendations**:
  - High-dimensional data: 0.6-0.8
  - Medium-dimensional data: 0.8-0.9
  - Low-dimensional data: 0.9-1.0

### LightGBM Tuning Examples

```python
# High-accuracy time series model
high_accuracy_lgb = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 63,
    'learning_rate': 0.03,          # Slower learning
    'feature_fraction': 0.95,
    'bagging_fraction': 0.85,
    'lambda_l1': 0.1,               # L1 regularization
    'lambda_l2': 0.1,               # L2 regularization
    'min_data_in_leaf': 20,         # Minimum samples per leaf
    'n_estimators': 200,            # More trees
    'verbose': -1
}

# Fast training model
fast_training_lgb = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 15,
    'learning_rate': 0.15,          # Faster learning
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'n_estimators': 50,             # Fewer trees
    'verbose': -1
}

# Robust to outliers
robust_lgb = {
    'objective': 'huber',           # Robust objective
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'alpha': 0.9,                   # Huber parameter
    'verbose': -1
}
```

## kNN Fallback Parameters

The k-Nearest Neighbors model handles noise points that HDBSCAN cannot cluster.

### Default Parameters

```python
knn_params = {
    'n_neighbors': 5,               # Number of neighbors
    'weights': 'distance',          # Distance-weighted predictions
    'algorithm': 'auto',            # Algorithm selection
    'metric': 'euclidean'           # Distance metric
}
```

### Parameter Descriptions

#### `n_neighbors` (int, default: 5)
- **Description**: Number of neighbors to use for predictions
- **Range**: 1 to N (dataset size)
- **Impact**:
  - Lower values: More sensitive to local patterns
  - Higher values: Smoother predictions, less sensitive
- **Recommendations**:
  - Small datasets: 3-7
  - Medium datasets: 5-15
  - Large datasets: 10-25

#### `weights` (str, default: 'distance')
- **Options**: 'uniform', 'distance'
- **Impact**:
  - 'uniform': All neighbors weighted equally
  - 'distance': Closer neighbors weighted more heavily
- **Recommendation**: 'distance' for most cases

#### `algorithm` (str, default: 'auto')
- **Options**: 'auto', 'ball_tree', 'kd_tree', 'brute'
- **Recommendation**: 'auto' lets sklearn choose the best algorithm

### kNN Tuning Examples

```python
# Conservative fallback
conservative_knn = {
    'n_neighbors': 10,              # More neighbors for stability
    'weights': 'distance',
    'algorithm': 'auto',
    'metric': 'euclidean'
}

# Sensitive fallback
sensitive_knn = {
    'n_neighbors': 3,               # Fewer neighbors for sensitivity
    'weights': 'distance',
    'algorithm': 'auto',
    'metric': 'euclidean'
}

# High-dimensional data
high_dim_knn = {
    'n_neighbors': 7,
    'weights': 'distance',
    'algorithm': 'ball_tree',       # Better for high dimensions
    'metric': 'manhattan'           # Alternative metric
}
```

## Pipeline Behavior Parameters

These parameters control the overall behavior of the pipeline.

### `noise_threshold` (float, default: 0.3)
- **Description**: Noise ratio threshold that triggers automatic retraining
- **Range**: 0.0 to 1.0
- **Impact**:
  - Lower values: More frequent retraining, higher adaptation
  - Higher values: Less frequent retraining, more stability
- **Recommendations**:
  - High stability needed: 0.4-0.6
  - Balanced approach: 0.25-0.35
  - High adaptability: 0.15-0.25

### `batch_size` (int, default: 1000)
- **Description**: Number of samples processed in each batch
- **Range**: 1 to dataset size
- **Impact**:
  - Smaller batches: More frequent updates, higher overhead
  - Larger batches: Less frequent updates, more efficient processing
- **Recommendations**:
  - Real-time processing: 50-200
  - Balanced processing: 200-1000
  - Batch processing: 1000-5000

### `expanding_window_size` (int, default: 10000)
- **Description**: Maximum number of samples to keep for retraining
- **Range**: batch_size to infinity
- **Impact**:
  - Smaller windows: Faster adaptation, less historical context
  - Larger windows: Slower adaptation, more historical context
- **Recommendations**:
  - Fast adaptation: 2000-5000
  - Balanced approach: 5000-15000
  - Long-term stability: 15000-50000

### Pipeline Parameter Examples

```python
# Real-time processing configuration
realtime_config = {
    'noise_threshold': 0.2,         # Quick adaptation
    'batch_size': 100,              # Small batches
    'expanding_window_size': 3000   # Limited history
}

# Batch processing configuration
batch_config = {
    'noise_threshold': 0.35,        # More stability
    'batch_size': 2000,             # Large batches
    'expanding_window_size': 20000  # More history
}

# High-stability configuration
stable_config = {
    'noise_threshold': 0.5,         # Very stable
    'batch_size': 1000,             # Standard batches
    'expanding_window_size': 50000  # Long history
}
```

## MLflow Integration Parameters

These parameters configure MLflow experiment tracking and model registry.

### `experiment_name` (str, default: "TimeSeries-Clustering-Pipeline")
- **Description**: Name of the MLflow experiment
- **Recommendations**: Use descriptive names that include:
  - Project identifier
  - Environment (dev/staging/prod)
  - Version or date

### `model_registry_prefix` (str, default: "timeseries-clustering")
- **Description**: Prefix for model names in MLflow model registry
- **Recommendations**: Use consistent naming conventions:
  - Include project/team identifier
  - Use lowercase with hyphens
  - Keep it concise but descriptive

### `hospital_profile_id` (Optional[str], default: None)
- **Description**: Hospital or organization identifier for artifact organization
- **Impact**: Organizes artifacts in separate folders by hospital
- **Recommendations**: Use when deploying across multiple organizations

### MLflow Configuration Examples

```python
# Development environment
dev_mlflow_config = {
    'experiment_name': 'patient-clustering-dev-v2',
    'model_registry_prefix': 'dev-patient-clustering',
    'hospital_profile_id': 'dev_hospital'
}

# Production environment
prod_mlflow_config = {
    'experiment_name': 'patient-clustering-prod',
    'model_registry_prefix': 'prod-patient-clustering',
    'hospital_profile_id': 'hospital_001'
}

# Multi-tenant configuration
multitenant_mlflow_config = {
    'experiment_name': 'clustering-pipeline-{hospital_id}',
    'model_registry_prefix': 'clustering-{hospital_id}',
    'hospital_profile_id': '{hospital_id}'  # Replaced at runtime
}
```

## Parameter Tuning Guidelines

### Step-by-Step Tuning Process

1. **Start with Default Configuration**
   ```python
   config = PipelineConfig()
   ```

2. **Tune HDBSCAN Parameters First**
   - Start with `min_cluster_size` based on your domain knowledge
   - Adjust `min_samples` to control clustering conservativeness
   - Use `cluster_selection_epsilon` if you need noise tolerance

3. **Optimize LightGBM Parameters**
   - Begin with `num_leaves` and `learning_rate`
   - Add regularization if overfitting occurs
   - Adjust `n_estimators` based on performance requirements

4. **Configure Pipeline Behavior**
   - Set `noise_threshold` based on your stability requirements
   - Choose `batch_size` based on your processing constraints
   - Set `expanding_window_size` based on available memory and adaptation needs

5. **Test and Validate**
   - Use cross-validation for parameter selection
   - Monitor performance metrics over time
   - Adjust based on production feedback

### Automated Parameter Tuning

```python
from sklearn.model_selection import ParameterGrid

# Define parameter grid
param_grid = {
    'min_cluster_size': [15, 25, 35],
    'learning_rate': [0.03, 0.05, 0.08],
    'num_leaves': [31, 63, 127],
    'noise_threshold': [0.2, 0.3, 0.4]
}

# Grid search example
best_config = None
best_score = float('inf')

for params in ParameterGrid(param_grid):
    config = PipelineConfig(
        hdbscan_params={'min_cluster_size': params['min_cluster_size']},
        lgb_timeseries_params={
            'learning_rate': params['learning_rate'],
            'num_leaves': params['num_leaves']
        },
        noise_threshold=params['noise_threshold']
    )
    
    # Evaluate configuration
    score = evaluate_configuration(config)
    
    if score < best_score:
        best_score = score
        best_config = config
```

## Configuration Examples

### Small Dataset Configuration (< 1000 samples)

```python
small_dataset_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 8,
        'min_samples': 3,
        'cluster_selection_epsilon': 0.1,
        'prediction_data': True,
        'cluster_selection_method': 'eom'
    },
    lgb_timeseries_params={
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 15,
        'learning_rate': 0.08,
        'n_estimators': 50,
        'verbose': -1
    },
    lgb_resource_params={
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 10,
        'learning_rate': 0.12,
        'n_estimators': 30,
        'verbose': -1
    },
    knn_params={
        'n_neighbors': 3,
        'weights': 'distance'
    },
    noise_threshold=0.25,
    batch_size=50,
    expanding_window_size=1000,
    experiment_name="small-dataset-clustering"
)
```

### Large Dataset Configuration (> 10000 samples)

```python
large_dataset_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 100,
        'min_samples': 30,
        'cluster_selection_epsilon': 0.0,
        'prediction_data': True,
        'cluster_selection_method': 'eom'
    },
    lgb_timeseries_params={
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 127,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'n_estimators': 200,
        'verbose': -1
    },
    lgb_resource_params={
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'verbose': -1
    },
    knn_params={
        'n_neighbors': 15,
        'weights': 'distance'
    },
    noise_threshold=0.35,
    batch_size=2000,
    expanding_window_size=30000,
    experiment_name="large-dataset-clustering"
)
```

### High-Performance Configuration

```python
high_performance_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 50,
        'min_samples': 15,
        'cluster_selection_epsilon': 0.0,
        'prediction_data': True,
        'cluster_selection_method': 'eom'
    },
    lgb_timeseries_params={
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'n_estimators': 50,
        'verbose': -1
    },
    lgb_resource_params={
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 15,
        'learning_rate': 0.15,
        'n_estimators': 30,
        'verbose': -1
    },
    knn_params={
        'n_neighbors': 5,
        'weights': 'distance',
        'algorithm': 'ball_tree'
    },
    noise_threshold=0.4,
    batch_size=1000,
    expanding_window_size=10000,
    experiment_name="high-performance-clustering"
)
```

### Medical/Healthcare Configuration

```python
medical_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 25,      # Meaningful patient groups
        'min_samples': 10,           # Conservative clustering
        'cluster_selection_epsilon': 0.1,  # Some noise tolerance
        'prediction_data': True,
        'cluster_selection_method': 'eom'
    },
    lgb_timeseries_params={
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 50,
        'learning_rate': 0.04,       # Conservative learning
        'feature_fraction': 0.9,     # Use most features
        'bagging_fraction': 0.8,
        'lambda_l1': 0.1,            # Regularization
        'lambda_l2': 0.1,
        'min_data_in_leaf': 15,      # Minimum samples per leaf
        'n_estimators': 150,
        'verbose': -1
    },
    lgb_resource_params={
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 25,
        'learning_rate': 0.06,
        'n_estimators': 80,
        'verbose': -1
    },
    knn_params={
        'n_neighbors': 8,            # More neighbors for stability
        'weights': 'distance'
    },
    noise_threshold=0.25,            # Moderate adaptation
    batch_size=200,                  # Reasonable batch size
    expanding_window_size=8000,      # Good historical context
    experiment_name="medical-patient-clustering",
    model_registry_prefix="medical-clustering",
    hospital_profile_id="hospital_001"
)
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Too Many Small Clusters

**Symptoms:**
- Large number of clusters (> 20% of data points)
- Many clusters with only a few points
- High noise ratio

**Solutions:**
```python
# Increase min_cluster_size
config.hdbscan_params['min_cluster_size'] = 50  # Increase from default

# Increase min_samples
config.hdbscan_params['min_samples'] = 20  # More conservative

# Reduce cluster_selection_epsilon
config.hdbscan_params['cluster_selection_epsilon'] = 0.0  # More strict
```

#### Issue: Too Few Clusters

**Symptoms:**
- Very few clusters (< 3)
- Large clusters with diverse patterns
- Low silhouette score

**Solutions:**
```python
# Decrease min_cluster_size
config.hdbscan_params['min_cluster_size'] = 10  # Decrease from default

# Decrease min_samples
config.hdbscan_params['min_samples'] = 3  # Less conservative

# Try leaf selection method
config.hdbscan_params['cluster_selection_method'] = 'leaf'
```

#### Issue: High Noise Ratio

**Symptoms:**
- Noise ratio > 30%
- Frequent retraining triggers
- Poor clustering quality

**Solutions:**
```python
# Increase noise tolerance
config.hdbscan_params['cluster_selection_epsilon'] = 0.2

# Decrease min_cluster_size
config.hdbscan_params['min_cluster_size'] = 15

# Increase noise threshold
config.noise_threshold = 0.4  # Less frequent retraining
```

#### Issue: Poor Time Series Prediction

**Symptoms:**
- High RMSE values
- Poor R² scores
- Inconsistent predictions

**Solutions:**
```python
# Increase model complexity
config.lgb_timeseries_params['num_leaves'] = 63
config.lgb_timeseries_params['n_estimators'] = 200

# Decrease learning rate
config.lgb_timeseries_params['learning_rate'] = 0.03

# Add regularization
config.lgb_timeseries_params['lambda_l1'] = 0.1
config.lgb_timeseries_params['lambda_l2'] = 0.1
```

#### Issue: Slow Processing

**Symptoms:**
- High batch processing times
- Memory issues
- Timeout errors

**Solutions:**
```python
# Reduce model complexity
config.lgb_timeseries_params['num_leaves'] = 15
config.lgb_timeseries_params['n_estimators'] = 50

# Reduce batch size
config.batch_size = 500

# Simplify HDBSCAN
config.hdbscan_params['min_cluster_size'] = 50
```

#### Issue: Frequent Retraining

**Symptoms:**
- Retraining triggered too often
- System instability
- High computational costs

**Solutions:**
```python
# Increase noise threshold
config.noise_threshold = 0.4

# Increase expanding window size
config.expanding_window_size = 20000

# Make clustering more stable
config.hdbscan_params['min_cluster_size'] = 30
config.hdbscan_params['min_samples'] = 15
```

### Validation Warnings and Solutions

The configuration system provides warnings for suboptimal settings:

#### Warning: "min_cluster_size < 5 may result in many small clusters"
```python
# Solution: Increase min_cluster_size
config.hdbscan_params['min_cluster_size'] = 10
```

#### Warning: "learning_rate > 0.3 may cause overfitting"
```python
# Solution: Decrease learning rate
config.lgb_timeseries_params['learning_rate'] = 0.1
```

#### Warning: "noise_threshold > 0.5 may trigger frequent retraining"
```python
# Solution: Decrease noise threshold
config.noise_threshold = 0.3
```

### Debugging Configuration Issues

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate configuration
try:
    config = PipelineConfig(**your_params)
    print("Configuration valid")
except ValueError as e:
    print(f"Configuration error: {e}")

# Check parameter compatibility
config_dict = config.to_dict()
print("Configuration summary:")
for key, value in config_dict.items():
    print(f"  {key}: {value}")
```

## Performance Optimization

### Memory Optimization

```python
# Reduce memory usage
memory_optimized_config = PipelineConfig(
    batch_size=500,                  # Smaller batches
    expanding_window_size=5000,      # Smaller window
    lgb_timeseries_params={
        'num_leaves': 31,            # Moderate complexity
        'feature_fraction': 0.7,     # Use fewer features
        'bagging_fraction': 0.7,     # Use less data
        'verbose': -1
    }
)
```

### Speed Optimization

```python
# Optimize for speed
speed_optimized_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 50,      # Larger clusters = faster
        'min_samples': 20,
        'prediction_data': True
    },
    lgb_timeseries_params={
        'num_leaves': 15,            # Simpler trees
        'learning_rate': 0.15,       # Faster learning
        'n_estimators': 30,          # Fewer trees
        'verbose': -1
    },
    knn_params={
        'n_neighbors': 3,            # Fewer neighbors
        'algorithm': 'ball_tree'     # Faster algorithm
    },
    batch_size=2000                  # Larger batches
)
```

### Accuracy Optimization

```python
# Optimize for accuracy
accuracy_optimized_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 15,      # More sensitive clustering
        'min_samples': 5,
        'cluster_selection_epsilon': 0.1,
        'prediction_data': True
    },
    lgb_timeseries_params={
        'num_leaves': 127,           # More complex trees
        'learning_rate': 0.02,       # Slower learning
        'n_estimators': 300,         # More trees
        'lambda_l1': 0.05,           # Light regularization
        'lambda_l2': 0.05,
        'verbose': -1
    },
    knn_params={
        'n_neighbors': 10,           # More neighbors
        'weights': 'distance'
    },
    noise_threshold=0.2              # More sensitive to drift
)
```

### Configuration Validation Script

```python
def validate_configuration(config: PipelineConfig) -> Dict[str, Any]:
    """Validate configuration and provide recommendations."""
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'recommendations': [],
        'estimated_performance': {}
    }
    
    # Check HDBSCAN parameters
    min_cluster_size = config.hdbscan_params.get('min_cluster_size', 5)
    min_samples = config.hdbscan_params.get('min_samples', 3)
    
    if min_cluster_size < 5:
        validation_results['warnings'].append(
            "Small min_cluster_size may create many small clusters"
        )
    
    if min_samples >= min_cluster_size:
        validation_results['warnings'].append(
            "min_samples >= min_cluster_size may be too conservative"
        )
    
    # Check LightGBM parameters
    learning_rate = config.lgb_timeseries_params.get('learning_rate', 0.1)
    num_leaves = config.lgb_timeseries_params.get('num_leaves', 31)
    
    if learning_rate > 0.3:
        validation_results['warnings'].append(
            "High learning_rate may cause overfitting"
        )
    
    if num_leaves > 100:
        validation_results['warnings'].append(
            "High num_leaves may cause overfitting"
        )
    
    # Estimate performance characteristics
    validation_results['estimated_performance'] = {
        'clustering_speed': 'fast' if min_cluster_size > 30 else 'moderate',
        'model_complexity': 'high' if num_leaves > 50 else 'moderate',
        'memory_usage': 'high' if config.expanding_window_size > 15000 else 'moderate',
        'adaptation_speed': 'fast' if config.noise_threshold < 0.25 else 'moderate'
    }
    
    return validation_results

# Usage
config = PipelineConfig()
results = validate_configuration(config)
print("Validation results:", results)
```

This comprehensive configuration guide provides all the information needed to effectively configure and tune the MLflow Time Series Clustering Pipeline for various use cases and requirements.