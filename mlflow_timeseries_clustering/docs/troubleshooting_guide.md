# MLflow Time Series Clustering Pipeline - Troubleshooting Guide

This comprehensive troubleshooting guide helps you diagnose and resolve common issues with the MLflow Time Series Clustering Pipeline.

## Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [Installation and Setup Issues](#installation-and-setup-issues)
3. [Configuration Problems](#configuration-problems)
4. [Data-Related Issues](#data-related-issues)
5. [Clustering Problems](#clustering-problems)
6. [Model Training Issues](#model-training-issues)
7. [Batch Processing Problems](#batch-processing-problems)
8. [MLflow Integration Issues](#mlflow-integration-issues)
9. [Performance Problems](#performance-problems)
10. [Retraining Issues](#retraining-issues)
11. [Memory and Resource Problems](#memory-and-resource-problems)
12. [Error Messages and Solutions](#error-messages-and-solutions)
13. [Debugging Tools and Techniques](#debugging-tools-and-techniques)
14. [Best Practices for Prevention](#best-practices-for-prevention)

## Quick Diagnostic Checklist

Before diving into specific issues, run through this quick checklist:

### Basic Health Check

```python
from mlflow_timeseries_clustering.pipeline.timeseries_clustering_pipeline import TimeSeriesClusteringPipeline

# Perform pipeline health diagnosis
pipeline = TimeSeriesClusteringPipeline(config)
health_report = pipeline.diagnose_pipeline_health()

print("Pipeline Health Report:")
print(f"Overall Status: {health_report['pipeline_status']}")
print(f"Component Health: {health_report['component_health']}")
print(f"Recommendations: {health_report['recommendations']}")
```

### Configuration Validation

```python
from mlflow_timeseries_clustering.core.config import PipelineConfig

# Validate configuration
try:
    config = PipelineConfig(**your_config_params)
    print("✅ Configuration is valid")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
```

### Data Validation

```python
import numpy as np

def quick_data_check(X, y=None):
    """Quick data validation check."""
    issues = []
    
    # Check basic properties
    if not isinstance(X, np.ndarray):
        issues.append("X must be a numpy array")
    
    if X.ndim != 2:
        issues.append(f"X must be 2D, got {X.ndim}D")
    
    if np.isnan(X).any():
        issues.append("X contains NaN values")
    
    if np.isinf(X).any():
        issues.append("X contains infinite values")
    
    if y is not None:
        if not isinstance(y, np.ndarray):
            issues.append("y must be a numpy array")
        
        if len(y) != len(X):
            issues.append(f"X and y length mismatch: {len(X)} vs {len(y)}")
    
    return issues

# Check your data
issues = quick_data_check(X_train, y_train)
if issues:
    print("❌ Data issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ Data validation passed")
```

## Installation and Setup Issues

### Issue: Import Errors

**Symptoms:**
```
ImportError: No module named 'mlflow_timeseries_clustering'
ModuleNotFoundError: No module named 'hdbscan'
```

**Solutions:**

1. **Check Installation:**
   ```bash
   pip list | grep mlflow
   pip list | grep hdbscan
   pip list | grep lightgbm
   ```

2. **Install Missing Dependencies:**
   ```bash
   pip install mlflow
   pip install hdbscan
   pip install lightgbm
   pip install scikit-learn
   pip install numpy pandas seaborn matplotlib
   ```

3. **Install from Requirements:**
   ```bash
   pip install -r mlflow_timeseries_clustering/requirements.txt
   ```

4. **Virtual Environment Issues:**
   ```bash
   # Activate your virtual environment
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   
   # Verify Python path
   which python
   python -c "import sys; print(sys.path)"
   ```

### Issue: MLflow Server Connection

**Symptoms:**
```
ConnectionError: Failed to connect to MLflow tracking server
mlflow.exceptions.MlflowException: Could not connect to tracking server
```

**Solutions:**

1. **Start MLflow Server:**
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

2. **Set Tracking URI:**
   ```python
   import mlflow
   mlflow.set_tracking_uri("http://localhost:5000")
   ```

3. **Use Local File Store:**
   ```python
   import mlflow
   mlflow.set_tracking_uri("file:./mlruns")
   ```

4. **Check Network Connectivity:**
   ```bash
   curl http://localhost:5000/health
   ```

## Configuration Problems

### Issue: Invalid Parameter Values

**Symptoms:**
```
ValueError: min_cluster_size must be at least 2
ValueError: learning_rate must be between 0.001 and 1.0
```

**Solutions:**

1. **Check Parameter Ranges:**
   ```python
   # Valid HDBSCAN parameters
   hdbscan_params = {
       'min_cluster_size': 5,      # >= 2
       'min_samples': 3,           # >= 1
       'cluster_selection_epsilon': 0.0,  # >= 0.0
   }
   
   # Valid LightGBM parameters
   lgb_params = {
       'learning_rate': 0.05,      # 0.001 to 1.0
       'num_leaves': 31,           # >= 2
   }
   ```

2. **Use Configuration Validation:**
   ```python
   from mlflow_timeseries_clustering.core.config import PipelineConfig
   
   try:
       config = PipelineConfig(
           hdbscan_params={'min_cluster_size': 1}  # Invalid
       )
   except ValueError as e:
       print(f"Configuration error: {e}")
       # Fix the parameter
       config = PipelineConfig(
           hdbscan_params={'min_cluster_size': 5}  # Valid
       )
   ```

### Issue: Parameter Compatibility Warnings

**Symptoms:**
```
Warning: min_samples >= min_cluster_size may result in conservative clustering
Warning: kNN n_neighbors >= HDBSCAN min_cluster_size may cause inconsistent predictions
```

**Solutions:**

1. **Adjust Parameter Relationships:**
   ```python
   # Ensure proper parameter relationships
   min_cluster_size = 20
   min_samples = min_cluster_size // 3  # 6-7
   knn_neighbors = min(min_cluster_size // 2, 10)  # 10
   
   config = PipelineConfig(
       hdbscan_params={
           'min_cluster_size': min_cluster_size,
           'min_samples': min_samples
       },
       knn_params={
           'n_neighbors': knn_neighbors
       }
   )
   ```

2. **Use Recommended Configurations:**
   ```python
   # Conservative configuration
   conservative_config = PipelineConfig(
       hdbscan_params={
           'min_cluster_size': 30,
           'min_samples': 15,
           'cluster_selection_epsilon': 0.05
       },
       knn_params={'n_neighbors': 10}
   )
   
   # Balanced configuration
   balanced_config = PipelineConfig(
       hdbscan_params={
           'min_cluster_size': 20,
           'min_samples': 8,
           'cluster_selection_epsilon': 0.1
       },
       knn_params={'n_neighbors': 6}
   )
   ```

## Data-Related Issues

### Issue: Data Format Problems

**Symptoms:**
```
DataValidationError: X must be a 2D numpy array
DataValidationError: X and y must have the same number of samples
```

**Solutions:**

1. **Convert Data to Proper Format:**
   ```python
   import numpy as np
   import pandas as pd
   
   # Convert pandas DataFrame to numpy
   if isinstance(X, pd.DataFrame):
       X = X.values
   
   # Ensure 2D array
   if X.ndim == 1:
       X = X.reshape(-1, 1)
   
   # Ensure proper data types
   X = X.astype(np.float64)
   y = y.astype(np.float64) if y is not None else None
   ```

2. **Handle Missing Values:**
   ```python
   from sklearn.impute import SimpleImputer
   
   # Check for missing values
   if np.isnan(X).any():
       print(f"Found {np.isnan(X).sum()} missing values")
       
       # Impute missing values
       imputer = SimpleImputer(strategy='mean')
       X = imputer.fit_transform(X)
   ```

3. **Handle Infinite Values:**
   ```python
   # Replace infinite values
   X = np.where(np.isinf(X), np.nan, X)
   
   # Then impute
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy='mean')
   X = imputer.fit_transform(X)
   ```

### Issue: Data Scale Problems

**Symptoms:**
- Poor clustering results
- Slow convergence
- Numerical instability

**Solutions:**

1. **Check Data Ranges:**
   ```python
   print(f"X range: {X.min():.3f} to {X.max():.3f}")
   print(f"X std: {X.std():.3f}")
   
   # Large ranges may need scaling
   if X.max() - X.min() > 100:
       print("Consider scaling your data")
   ```

2. **Manual Scaling (if needed):**
   ```python
   from sklearn.preprocessing import StandardScaler
   
   # The pipeline handles scaling automatically, but you can pre-scale
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

### Issue: Insufficient Data

**Symptoms:**
```
ModelTrainingError: Insufficient data for cluster 2
Warning: Cluster 1 has only 3 samples, skipping model training
```

**Solutions:**

1. **Check Data Size Requirements:**
   ```python
   min_samples_needed = config.hdbscan_params['min_cluster_size'] * 3
   print(f"Minimum samples needed: {min_samples_needed}")
   print(f"Available samples: {len(X)}")
   
   if len(X) < min_samples_needed:
       print("❌ Insufficient data for current configuration")
   ```

2. **Adjust Configuration for Small Datasets:**
   ```python
   # Configuration for small datasets
   small_data_config = PipelineConfig(
       hdbscan_params={
           'min_cluster_size': max(5, len(X) // 20),  # Adaptive
           'min_samples': 3
       },
       lgb_timeseries_params={
           'num_leaves': 15,  # Simpler model
           'n_estimators': 50
       }
   )
   ```

3. **Generate More Data or Use Data Augmentation:**
   ```python
   from sklearn.utils import resample
   
   # Bootstrap sampling to increase data size
   if len(X) < 1000:
       X_resampled, y_resampled = resample(X, y, n_samples=1000, random_state=42)
   ```

## Clustering Problems

### Issue: Too Many Small Clusters

**Symptoms:**
- Large number of clusters (> 20% of data points)
- Many single-point clusters
- High noise ratio

**Diagnostic:**
```python
clustering_result = pipeline.fit_initial(X, y)
n_clusters = clustering_result.metrics['n_clusters']
noise_ratio = clustering_result.noise_ratio

print(f"Number of clusters: {n_clusters}")
print(f"Noise ratio: {noise_ratio:.3f}")
print(f"Cluster ratio: {n_clusters / len(X):.3f}")

if n_clusters / len(X) > 0.2:
    print("❌ Too many small clusters detected")
```

**Solutions:**

1. **Increase Minimum Cluster Size:**
   ```python
   config.hdbscan_params['min_cluster_size'] = 50  # Increase from default
   config.hdbscan_params['min_samples'] = 20
   ```

2. **Reduce Noise Tolerance:**
   ```python
   config.hdbscan_params['cluster_selection_epsilon'] = 0.0  # More strict
   ```

3. **Use More Conservative Method:**
   ```python
   config.hdbscan_params['cluster_selection_method'] = 'eom'  # More stable
   ```

### Issue: Too Few Clusters

**Symptoms:**
- Very few clusters (< 3)
- Large, diverse clusters
- Low silhouette score

**Solutions:**

1. **Decrease Minimum Cluster Size:**
   ```python
   config.hdbscan_params['min_cluster_size'] = 10  # Decrease
   config.hdbscan_params['min_samples'] = 3
   ```

2. **Increase Noise Tolerance:**
   ```python
   config.hdbscan_params['cluster_selection_epsilon'] = 0.2
   ```

3. **Try Leaf Selection:**
   ```python
   config.hdbscan_params['cluster_selection_method'] = 'leaf'
   ```

### Issue: High Noise Ratio

**Symptoms:**
- Noise ratio > 30%
- Frequent retraining triggers
- Poor clustering quality

**Diagnostic:**
```python
def analyze_noise_pattern(X, labels):
    """Analyze noise point patterns."""
    noise_mask = labels == -1
    noise_points = X[noise_mask]
    
    if len(noise_points) > 0:
        print(f"Noise points: {len(noise_points)} ({len(noise_points)/len(X)*100:.1f}%)")
        print(f"Noise point statistics:")
        print(f"  Mean: {noise_points.mean(axis=0)}")
        print(f"  Std: {noise_points.std(axis=0)}")
    
    return noise_points

noise_points = analyze_noise_pattern(X, clustering_result.labels)
```

**Solutions:**

1. **Adjust Clustering Parameters:**
   ```python
   # More tolerant to noise
   config.hdbscan_params['cluster_selection_epsilon'] = 0.2
   config.hdbscan_params['min_cluster_size'] = 15  # Smaller clusters
   ```

2. **Improve kNN Fallback:**
   ```python
   config.knn_params['n_neighbors'] = 8  # More neighbors
   config.knn_params['weights'] = 'distance'
   ```

3. **Increase Noise Threshold:**
   ```python
   config.noise_threshold = 0.4  # Less frequent retraining
   ```

## Model Training Issues

### Issue: LightGBM Training Failures

**Symptoms:**
```
ModelTrainingError: LightGBM training failed for cluster 1
LightGBMError: Check failed: (num_data) > (0) at dataset.cpp
```

**Solutions:**

1. **Check Cluster Data Size:**
   ```python
   def diagnose_cluster_data(cluster_data):
       """Diagnose cluster data for training issues."""
       for cluster_id, data in cluster_data.items():
           X_cluster = data['X']
           y_cluster = data['y']
           
           print(f"Cluster {cluster_id}:")
           print(f"  Samples: {len(X_cluster)}")
           print(f"  Features: {X_cluster.shape[1] if len(X_cluster) > 0 else 0}")
           print(f"  Target range: {y_cluster.min():.3f} to {y_cluster.max():.3f}")
           
           if len(X_cluster) < 10:
               print(f"  ❌ Insufficient data for training")
           else:
               print(f"  ✅ Sufficient data")
   ```

2. **Adjust LightGBM Parameters for Small Clusters:**
   ```python
   # Parameters for small clusters
   small_cluster_lgb_params = {
       'objective': 'regression',
       'metric': 'rmse',
       'num_leaves': 10,        # Smaller trees
       'learning_rate': 0.1,    # Faster learning
       'min_data_in_leaf': 3,   # Minimum data per leaf
       'n_estimators': 30,      # Fewer trees
       'verbose': -1
   }
   ```

3. **Skip Training for Small Clusters:**
   ```python
   # The pipeline automatically skips clusters with insufficient data
   # You can adjust the minimum threshold
   MIN_CLUSTER_SIZE_FOR_TRAINING = 15
   ```

### Issue: Overfitting in Models

**Symptoms:**
- High training accuracy, poor validation accuracy
- Large difference between train and test metrics
- Poor generalization to new data

**Solutions:**

1. **Add Regularization:**
   ```python
   regularized_lgb_params = {
       'objective': 'regression',
       'metric': 'rmse',
       'num_leaves': 31,
       'learning_rate': 0.05,
       'lambda_l1': 0.1,        # L1 regularization
       'lambda_l2': 0.1,        # L2 regularization
       'min_data_in_leaf': 20,  # Minimum samples per leaf
       'feature_fraction': 0.8, # Use subset of features
       'bagging_fraction': 0.8, # Use subset of data
       'verbose': -1
   }
   ```

2. **Reduce Model Complexity:**
   ```python
   simple_lgb_params = {
       'objective': 'regression',
       'metric': 'rmse',
       'num_leaves': 15,        # Fewer leaves
       'learning_rate': 0.1,
       'n_estimators': 50,      # Fewer trees
       'verbose': -1
   }
   ```

3. **Early Stopping:**
   ```python
   early_stopping_lgb_params = {
       'objective': 'regression',
       'metric': 'rmse',
       'num_leaves': 31,
       'learning_rate': 0.05,
       'n_estimators': 1000,
       'early_stopping_rounds': 50,  # Stop if no improvement
       'verbose': -1
   }
   ```

## Batch Processing Problems

### Issue: Slow Batch Processing

**Symptoms:**
- High processing times per batch
- Memory usage spikes
- Timeout errors

**Diagnostic:**
```python
import time

def benchmark_batch_processing(pipeline, X_batch):
    """Benchmark batch processing performance."""
    start_time = time.time()
    
    try:
        result = pipeline.process_batch(X_batch)
        processing_time = time.time() - start_time
        
        print(f"Batch processing time: {processing_time:.3f}s")
        print(f"Samples per second: {len(X_batch) / processing_time:.1f}")
        print(f"Noise ratio: {result.noise_ratio:.3f}")
        
        return result, processing_time
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return None, None
```

**Solutions:**

1. **Reduce Batch Size:**
   ```python
   config.batch_size = 500  # Reduce from default 1000
   ```

2. **Optimize Model Parameters:**
   ```python
   # Faster LightGBM parameters
   fast_lgb_params = {
       'objective': 'regression',
       'metric': 'rmse',
       'num_leaves': 15,        # Simpler trees
       'learning_rate': 0.15,   # Faster learning
       'n_estimators': 30,      # Fewer trees
       'verbose': -1
   }
   ```

3. **Optimize kNN Parameters:**
   ```python
   fast_knn_params = {
       'n_neighbors': 3,        # Fewer neighbors
       'algorithm': 'ball_tree', # Faster algorithm
       'weights': 'uniform'     # Simpler weighting
   }
   ```

### Issue: Inconsistent Batch Results

**Symptoms:**
- Highly variable noise ratios
- Inconsistent cluster assignments
- Erratic performance metrics

**Solutions:**

1. **Check Data Consistency:**
   ```python
   def check_batch_consistency(batches):
       """Check consistency across batches."""
       batch_stats = []
       
       for i, (X_batch, y_batch) in enumerate(batches):
           stats = {
               'batch_id': i,
               'size': len(X_batch),
               'mean': X_batch.mean(),
               'std': X_batch.std(),
               'range': X_batch.max() - X_batch.min()
           }
           batch_stats.append(stats)
       
       # Check for outlier batches
       means = [s['mean'] for s in batch_stats]
       mean_std = np.std(means)
       
       if mean_std > 1.0:
           print("❌ High variability in batch statistics")
       else:
           print("✅ Batch statistics are consistent")
       
       return batch_stats
   ```

2. **Stabilize Clustering:**
   ```python
   # More stable clustering parameters
   stable_hdbscan_params = {
       'min_cluster_size': 30,      # Larger clusters
       'min_samples': 15,           # More conservative
       'cluster_selection_epsilon': 0.05,  # Less noise tolerance
       'cluster_selection_method': 'eom'   # More stable method
   }
   ```

3. **Increase Noise Threshold:**
   ```python
   config.noise_threshold = 0.35  # Less sensitive to variations
   ```

## MLflow Integration Issues

### Issue: Experiment Creation Failures

**Symptoms:**
```
MLflowIntegrationError: Failed to create experiment
mlflow.exceptions.MlflowException: Experiment 'name' already exists
```

**Solutions:**

1. **Handle Existing Experiments:**
   ```python
   import mlflow
   
   def get_or_create_experiment(experiment_name):
       """Get existing experiment or create new one."""
       try:
           experiment = mlflow.get_experiment_by_name(experiment_name)
           if experiment:
               return experiment.experiment_id
       except:
           pass
       
       return mlflow.create_experiment(experiment_name)
   
   # Use in configuration
   experiment_id = get_or_create_experiment("my-experiment")
   ```

2. **Use Unique Experiment Names:**
   ```python
   from datetime import datetime
   
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   experiment_name = f"clustering-pipeline-{timestamp}"
   
   config = PipelineConfig(experiment_name=experiment_name)
   ```

### Issue: Artifact Storage Problems

**Symptoms:**
```
MLflowIntegrationError: Failed to log artifact
FileNotFoundError: Artifact path not found
```

**Solutions:**

1. **Check Artifact Paths:**
   ```python
   import os
   
   def verify_artifact_path(artifact_path):
       """Verify artifact path exists and is accessible."""
       if not os.path.exists(artifact_path):
           print(f"❌ Artifact path does not exist: {artifact_path}")
           return False
       
       if not os.access(artifact_path, os.R_OK):
           print(f"❌ Artifact path not readable: {artifact_path}")
           return False
       
       print(f"✅ Artifact path is valid: {artifact_path}")
       return True
   ```

2. **Set Up Proper Artifact Store:**
   ```python
   import mlflow
   
   # Use local file store
   mlflow.set_tracking_uri("file:./mlruns")
   
   # Or use remote store
   # mlflow.set_tracking_uri("http://mlflow-server:5000")
   ```

3. **Handle Artifact Storage Errors:**
   ```python
   def safe_log_artifact(artifact_path, artifact_name):
       """Safely log artifact with error handling."""
       try:
           mlflow.log_artifact(artifact_path, artifact_name)
           print(f"✅ Artifact logged: {artifact_name}")
       except Exception as e:
           print(f"❌ Failed to log artifact {artifact_name}: {e}")
   ```

## Performance Problems

### Issue: Memory Usage Issues

**Symptoms:**
- Out of memory errors
- System slowdown
- Process killed by OS

**Diagnostic:**
```python
import psutil
import os

def monitor_memory_usage():
    """Monitor memory usage during pipeline operations."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"Memory percent: {process.memory_percent():.1f}%")
    
    return memory_info.rss
```

**Solutions:**

1. **Reduce Memory Footprint:**
   ```python
   # Memory-optimized configuration
   memory_config = PipelineConfig(
       batch_size=200,                  # Smaller batches
       expanding_window_size=5000,      # Smaller window
       lgb_timeseries_params={
           'num_leaves': 15,            # Simpler models
           'feature_fraction': 0.7,     # Use fewer features
           'verbose': -1
       }
   )
   ```

2. **Process Data in Chunks:**
   ```python
   def process_large_batch_in_chunks(pipeline, X_large, chunk_size=500):
       """Process large batch in smaller chunks."""
       results = []
       
       for i in range(0, len(X_large), chunk_size):
           chunk = X_large[i:i+chunk_size]
           result = pipeline.process_batch(chunk)
           results.append(result)
           
           # Optional: clear memory
           import gc
           gc.collect()
       
       return results
   ```

3. **Use Data Types Efficiently:**
   ```python
   # Use appropriate data types
   X = X.astype(np.float32)  # Instead of float64
   y = y.astype(np.float32)
   ```

### Issue: CPU Performance Problems

**Symptoms:**
- High CPU usage
- Slow processing
- System unresponsiveness

**Solutions:**

1. **Optimize LightGBM Threading:**
   ```python
   optimized_lgb_params = {
       'objective': 'regression',
       'metric': 'rmse',
       'num_threads': 4,        # Limit threads
       'num_leaves': 31,
       'learning_rate': 0.1,    # Faster learning
       'n_estimators': 50,      # Fewer trees
       'verbose': -1
   }
   ```

2. **Use Efficient Algorithms:**
   ```python
   efficient_knn_params = {
       'n_neighbors': 5,
       'algorithm': 'ball_tree',  # Efficient for high dimensions
       'leaf_size': 30,           # Optimize tree structure
       'weights': 'distance'
   }
   ```

3. **Parallel Processing:**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   def parallel_batch_processing(pipeline, batches, max_workers=4):
       """Process multiple batches in parallel."""
       with ThreadPoolExecutor(max_workers=max_workers) as executor:
           futures = [executor.submit(pipeline.process_batch, batch) 
                     for batch in batches]
           results = [future.result() for future in futures]
       return results
   ```

## Retraining Issues

### Issue: Frequent Retraining

**Symptoms:**
- Retraining triggered too often
- System instability
- High computational costs

**Diagnostic:**
```python
def analyze_retraining_frequency(pipeline, batches):
    """Analyze retraining frequency patterns."""
    retraining_events = []
    
    for i, (X_batch, y_batch) in enumerate(batches):
        result = pipeline.process_batch(X_batch, y_batch)
        
        if result.noise_ratio > pipeline.config.noise_threshold:
            retraining_events.append({
                'batch_id': i,
                'noise_ratio': result.noise_ratio,
                'threshold': pipeline.config.noise_threshold
            })
    
    print(f"Retraining events: {len(retraining_events)}")
    print(f"Retraining frequency: {len(retraining_events) / len(batches):.2f}")
    
    return retraining_events
```

**Solutions:**

1. **Adjust Noise Threshold:**
   ```python
   config.noise_threshold = 0.4  # Increase from default 0.3
   ```

2. **Use Moving Average for Stability:**
   ```python
   class StableRetrainingManager:
       def __init__(self, window_size=5, threshold=0.3):
           self.window_size = window_size
           self.threshold = threshold
           self.noise_history = []
       
       def should_retrain(self, current_noise_ratio):
           self.noise_history.append(current_noise_ratio)
           
           if len(self.noise_history) > self.window_size:
               self.noise_history.pop(0)
           
           # Use moving average instead of single value
           avg_noise = sum(self.noise_history) / len(self.noise_history)
           return avg_noise > self.threshold
   ```

3. **Implement Cooldown Period:**
   ```python
   from datetime import datetime, timedelta
   
   class CooldownRetrainingManager:
       def __init__(self, cooldown_hours=24):
           self.cooldown_period = timedelta(hours=cooldown_hours)
           self.last_retraining = None
       
       def can_retrain(self):
           if self.last_retraining is None:
               return True
           
           return datetime.now() - self.last_retraining > self.cooldown_period
   ```

### Issue: Retraining Failures

**Symptoms:**
```
RetrainingError: Failed to retrain models
ModelTrainingError: Insufficient data for retraining
```

**Solutions:**

1. **Check Data Availability:**
   ```python
   def check_retraining_data_availability(pipeline):
       """Check if sufficient data is available for retraining."""
       window_size = pipeline.config.expanding_window_size
       available_data = len(pipeline._training_data['X']) if pipeline._training_data else 0
       
       print(f"Available data: {available_data}")
       print(f"Window size: {window_size}")
       
       if available_data < window_size * 0.5:
           print("❌ Insufficient data for reliable retraining")
           return False
       
       print("✅ Sufficient data for retraining")
       return True
   ```

2. **Graceful Retraining Fallback:**
   ```python
   def safe_retrain(pipeline, reason="Manual"):
       """Safely attempt retraining with fallback."""
       try:
           pipeline.retrain(reason)
           print("✅ Retraining successful")
           return True
       except RetrainingError as e:
           print(f"❌ Retraining failed: {e}")
           print("Continuing with current models")
           return False
   ```

## Memory and Resource Problems

### Issue: Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- System slowdown over time
- Eventually running out of memory

**Solutions:**

1. **Monitor Memory Usage:**
   ```python
   import gc
   import psutil
   
   def monitor_and_cleanup():
       """Monitor memory and perform cleanup."""
       # Get memory usage before cleanup
       before = psutil.Process().memory_info().rss / 1024 / 1024
       
       # Force garbage collection
       gc.collect()
       
       # Get memory usage after cleanup
       after = psutil.Process().memory_info().rss / 1024 / 1024
       
       print(f"Memory: {before:.1f} MB -> {after:.1f} MB (freed {before-after:.1f} MB)")
   ```

2. **Clear Large Objects:**
   ```python
   def clear_pipeline_cache(pipeline):
       """Clear pipeline caches to free memory."""
       # Clear performance history
       if hasattr(pipeline, '_performance_history'):
           pipeline._performance_history = pipeline._performance_history[-100:]  # Keep last 100
       
       # Clear error history
       pipeline.error_handler.clear_error_history()
       
       # Force garbage collection
       import gc
       gc.collect()
   ```

3. **Use Context Managers:**
   ```python
   from contextlib import contextmanager
   
   @contextmanager
   def memory_managed_processing():
       """Context manager for memory-managed processing."""
       try:
           yield
       finally:
           import gc
           gc.collect()
   
   # Usage
   with memory_managed_processing():
       result = pipeline.process_batch(X_batch)
   ```

### Issue: Disk Space Problems

**Symptoms:**
- MLflow artifact storage failures
- Model saving errors
- System disk full warnings

**Solutions:**

1. **Monitor Disk Usage:**
   ```python
   import shutil
   
   def check_disk_space(path="."):
       """Check available disk space."""
       total, used, free = shutil.disk_usage(path)
       
       print(f"Total: {total // (2**30)} GB")
       print(f"Used: {used // (2**30)} GB")
       print(f"Free: {free // (2**30)} GB")
       
       if free < 1 * (2**30):  # Less than 1 GB
           print("❌ Low disk space warning")
           return False
       
       return True
   ```

2. **Clean Up Old Artifacts:**
   ```python
   import os
   from datetime import datetime, timedelta
   
   def cleanup_old_artifacts(mlruns_path="./mlruns", days_old=30):
       """Clean up old MLflow artifacts."""
       cutoff_date = datetime.now() - timedelta(days=days_old)
       
       for root, dirs, files in os.walk(mlruns_path):
           for file in files:
               file_path = os.path.join(root, file)
               file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
               
               if file_time < cutoff_date:
                   try:
                       os.remove(file_path)
                       print(f"Removed old artifact: {file_path}")
                   except Exception as e:
                       print(f"Failed to remove {file_path}: {e}")
   ```

3. **Configure Artifact Retention:**
   ```python
   # Configure shorter artifact retention
   config = PipelineConfig(
       # Use shorter experiment names to reduce path length
       experiment_name="clustering",
       # Implement custom artifact cleanup in your pipeline
   )
   ```

## Error Messages and Solutions

### Common Error Messages

#### "DataValidationError: X must be a 2D numpy array"

**Solution:**
```python
import numpy as np

# Ensure X is 2D numpy array
if not isinstance(X, np.ndarray):
    X = np.array(X)

if X.ndim == 1:
    X = X.reshape(-1, 1)
elif X.ndim > 2:
    X = X.reshape(X.shape[0], -1)
```

#### "ModelTrainingError: Insufficient data for cluster N"

**Solution:**
```python
# Reduce minimum cluster size or increase data
config.hdbscan_params['min_cluster_size'] = 10  # Reduce from default
```

#### "PredictionError: Pipeline not fitted"

**Solution:**
```python
# Ensure pipeline is trained before use
if not pipeline.is_fitted:
    clustering_result = pipeline.fit_initial(X_train, y_train)
```

#### "MLflowIntegrationError: Failed to connect to tracking server"

**Solution:**
```python
import mlflow

# Use local file store
mlflow.set_tracking_uri("file:./mlruns")

# Or start MLflow server
# mlflow server --host 0.0.0.0 --port 5000
```

#### "RetrainingError: Expanding window data not available"

**Solution:**
```python
# Ensure sufficient batches have been processed
min_batches = config.expanding_window_size // config.batch_size
print(f"Process at least {min_batches} batches before retraining")
```

## Debugging Tools and Techniques

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable specific component logging
logger = logging.getLogger('mlflow_timeseries_clustering')
logger.setLevel(logging.DEBUG)
```

### Pipeline State Inspection

```python
def inspect_pipeline_state(pipeline):
    """Inspect current pipeline state for debugging."""
    print("Pipeline State Inspection:")
    print(f"  Fitted: {pipeline.is_fitted}")
    print(f"  Batch processing enabled: {pipeline.is_batch_processing_enabled}")
    print(f"  Model version: {pipeline.current_model_version}")
    print(f"  Health status: {pipeline.current_health_status}")
    
    # Check component states
    if pipeline.clustering_engine:
        print(f"  Clustering engine: ✅ Available")
    else:
        print(f"  Clustering engine: ❌ Not available")
    
    if pipeline.model_manager:
        print(f"  Model manager: ✅ Available")
    else:
        print(f"  Model manager: ❌ Not available")
    
    # Check error statistics
    error_stats = pipeline.get_error_statistics()
    print(f"  Total errors: {error_stats.get('total_errors', 0)}")
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_pipeline_operation(operation_func, *args, **kwargs):
    """Profile a pipeline operation for performance analysis."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = operation_func(*args, **kwargs)
    finally:
        profiler.disable()
    
    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result

# Usage
result = profile_pipeline_operation(pipeline.process_batch, X_batch)
```

### Data Quality Checks

```python
def comprehensive_data_check(X, y=None):
    """Comprehensive data quality check."""
    issues = []
    
    # Basic checks
    if not isinstance(X, np.ndarray):
        issues.append("X is not a numpy array")
    
    if X.ndim != 2:
        issues.append(f"X is not 2D (shape: {X.shape})")
    
    # Missing values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        issues.append(f"X contains {nan_count} NaN values")
    
    # Infinite values
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        issues.append(f"X contains {inf_count} infinite values")
    
    # Data range
    if X.max() - X.min() > 1000:
        issues.append("X has very large range, consider scaling")
    
    # Constant features
    constant_features = np.var(X, axis=0) == 0
    if constant_features.any():
        issues.append(f"{constant_features.sum()} constant features detected")
    
    # Target checks
    if y is not None:
        if len(y) != len(X):
            issues.append(f"X and y length mismatch: {len(X)} vs {len(y)}")
        
        if np.isnan(y).any():
            issues.append("y contains NaN values")
    
    return issues
```

## Best Practices for Prevention

### 1. Data Validation

```python
def validate_before_training(X, y):
    """Validate data before training."""
    issues = comprehensive_data_check(X, y)
    
    if issues:
        print("❌ Data validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✅ Data validation passed")
    return True

# Always validate before training
if validate_before_training(X_train, y_train):
    result = pipeline.fit_initial(X_train, y_train)
```

### 2. Configuration Testing

```python
def test_configuration(config):
    """Test configuration with small dataset."""
    # Generate small test dataset
    X_test = np.random.randn(100, 10)
    y_test = np.random.randn(100)
    
    try:
        pipeline = TimeSeriesClusteringPipeline(config)
        result = pipeline.fit_initial(X_test, y_test)
        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

# Test configuration before production use
if test_configuration(config):
    # Proceed with full pipeline
    pass
```

### 3. Monitoring Setup

```python
def setup_comprehensive_monitoring(pipeline):
    """Set up comprehensive monitoring."""
    # Set performance thresholds
    pipeline.update_performance_thresholds(
        alert_thresholds={
            'noise_ratio': 0.3,
            'processing_time': 5.0,
            'memory_usage': 1000.0
        },
        drift_thresholds={
            'noise_ratio_trend': 0.05,
            'performance_degradation': 0.1
        }
    )
    
    # Enable graceful degradation
    pipeline.enable_graceful_degradation(True)
    
    print("✅ Comprehensive monitoring enabled")
```

### 4. Regular Health Checks

```python
def regular_health_check(pipeline):
    """Perform regular health check."""
    health_report = pipeline.perform_health_check()
    
    if health_report['overall_health'] != 'healthy':
        print(f"❌ Health issue detected: {health_report['overall_health']}")
        
        if health_report.get('recommendations'):
            print("Recommendations:")
            for rec in health_report['recommendations']:
                print(f"  - {rec}")
    else:
        print("✅ Pipeline health is good")
    
    return health_report

# Schedule regular health checks
import schedule
schedule.every(1).hours.do(lambda: regular_health_check(pipeline))
```

This troubleshooting guide provides comprehensive solutions for common issues you might encounter with the MLflow Time Series Clustering Pipeline. Always start with the quick diagnostic checklist and use the appropriate debugging tools to identify the root cause of problems.