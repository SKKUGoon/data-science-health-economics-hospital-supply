# MLflow Time Series Clustering Pipeline

A comprehensive machine learning pipeline for time series patient data analysis using HDBSCAN clustering and LightGBM models, with full MLflow integration for experiment tracking and model management.

## Overview

This pipeline provides:

- **HDBSCAN Clustering**: Density-based clustering for patient segmentation
- **LightGBM Models**: Cluster-specific time series and resource usage prediction
- **kNN Fallback**: Handles noise points that HDBSCAN cannot cluster
- **Adaptive Retraining**: Automatic model updates based on data drift detection
- **MLflow Integration**: Complete experiment tracking and model registry
- **Performance Monitoring**: Real-time performance tracking and alerting
- **Error Handling**: Graceful degradation and comprehensive error management

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from mlflow_timeseries_clustering.core.config import PipelineConfig
from mlflow_timeseries_clustering.pipeline.timeseries_clustering_pipeline import TimeSeriesClusteringPipeline

# Create configuration
config = PipelineConfig(
    noise_threshold=0.25,
    batch_size=500,
    experiment_name="my-clustering-experiment"
)

# Initialize and train pipeline
pipeline = TimeSeriesClusteringPipeline(config)
clustering_result = pipeline.fit_initial(X_train, y_train)

# Process new data
batch_result = pipeline.process_batch(X_new)
```

## Examples

### 1. Initial Training Example

Complete workflow for training a new pipeline from scratch:

```bash
python mlflow_timeseries_clustering/examples/initial_training_example.py
```

**Features:**
- Data preparation and validation
- Configuration setup with parameter tuning
- Initial model training (HDBSCAN + LightGBM)
- Performance evaluation and reporting
- Model registration and artifact management

### 2. Batch Processing Example

Production-ready batch processing with monitoring:

```bash
python mlflow_timeseries_clustering/examples/batch_processing_example.py
```

**Features:**
- Loading pre-trained pipeline
- Processing multiple data batches
- Performance monitoring and drift detection
- Error handling and edge cases
- Performance optimization strategies

### 3. Retraining Configuration Example

Advanced retraining scenarios and configurations:

```bash
python mlflow_timeseries_clustering/examples/retraining_configuration_example.py
```

**Features:**
- Automatic retraining triggered by noise thresholds
- Manual retraining with custom configurations
- Scheduled retraining workflows
- Performance comparison between model versions
- Advanced retraining strategies

### 4. Complete Pipeline Example

End-to-end pipeline demonstration:

```bash
python mlflow_timeseries_clustering/examples/complete_pipeline_example.py
```

**Features:**
- Full pipeline lifecycle demonstration
- Comprehensive error handling
- Performance monitoring and alerting
- Health checks and diagnostics

## Documentation

### Configuration Guide

Comprehensive guide for configuring all pipeline parameters:

- [Configuration Guide](docs/configuration_guide.md)
  - HDBSCAN clustering parameters
  - LightGBM model parameters
  - kNN fallback parameters
  - Pipeline behavior parameters
  - Parameter tuning guidelines
  - Configuration examples for different use cases

### API Documentation

Complete API reference for all classes and methods:

- [API Documentation](docs/api_documentation.md)
  - Core classes and methods
  - Configuration management
  - Data models
  - Pipeline controller
  - Clustering components
  - Model management
  - MLflow integration
  - Monitoring and retraining

### Troubleshooting Guide

Solutions for common issues and problems:

- [Troubleshooting Guide](docs/troubleshooting_guide.md)
  - Quick diagnostic checklist
  - Installation and setup issues
  - Configuration problems
  - Data-related issues
  - Clustering problems
  - Model training issues
  - Performance problems
  - Error messages and solutions

## Architecture

### Pipeline Components

```
┌─────────────────────────────────────────────────────────────┐
│                    TimeSeriesClusteringPipeline            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ HDBSCAN         │  │ LightGBM         │  │ kNN         │ │
│  │ Clustering      │  │ Time Series      │  │ Fallback    │ │
│  │ Engine          │  │ Models           │  │ Model       │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ MLflow          │  │ Performance      │  │ Adaptive    │ │
│  │ Integration     │  │ Monitor          │  │ Retraining  │ │
│  │                 │  │                  │  │ Manager     │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ Error           │  │ Batch            │  │ Artifact    │ │
│  │ Handler         │  │ Processor        │  │ Manager     │ │
│  │                 │  │                  │  │             │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### MLflow Integration

```
┌─────────────────────────────────────────────────────────────┐
│                      MLflow Tracking                       │
├─────────────────────────────────────────────────────────────┤
│  Experiments:                                              │
│  ├── TimeSeries-Clustering-Training                        │
│  ├── TimeSeries-Clustering-Batch-Processing                │
│  └── TimeSeries-Clustering-Retraining                      │
├─────────────────────────────────────────────────────────────┤
│  Runs Hierarchy:                                           │
│  ├── Parent Run: Pipeline                                  │
│  │   ├── Child: Preprocessing                              │
│  │   ├── Child: HDBSCAN                                    │
│  │   ├── Child: kNN-Fallback                               │
│  │   ├── Child: TimeSeries-LGB                             │
│  │   ├── Child: Resource-LGB                               │
│  │   └── Child: Performance-Report                         │
├─────────────────────────────────────────────────────────────┤
│  Model Registry:                                           │
│  ├── TimeSeries-Clustering-Pipeline                        │
│  │   ├── Version 1: Initial Training                       │
│  │   ├── Version 2: After Retraining                       │
│  │   ├── Staging Environment                               │
│  │   └── Production Environment                            │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Examples

### Small Dataset Configuration

```python
small_dataset_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 8,
        'min_samples': 3,
        'cluster_selection_epsilon': 0.1
    },
    lgb_timeseries_params={
        'num_leaves': 15,
        'learning_rate': 0.08,
        'n_estimators': 50
    },
    noise_threshold=0.25,
    batch_size=50,
    expanding_window_size=1000
)
```

### Large Dataset Configuration

```python
large_dataset_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 100,
        'min_samples': 30,
        'cluster_selection_epsilon': 0.0
    },
    lgb_timeseries_params={
        'num_leaves': 127,
        'learning_rate': 0.03,
        'n_estimators': 200
    },
    noise_threshold=0.35,
    batch_size=2000,
    expanding_window_size=30000
)
```

### Medical/Healthcare Configuration

```python
medical_config = PipelineConfig(
    hdbscan_params={
        'min_cluster_size': 25,
        'min_samples': 10,
        'cluster_selection_epsilon': 0.1
    },
    lgb_timeseries_params={
        'num_leaves': 50,
        'learning_rate': 0.04,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'n_estimators': 150
    },
    noise_threshold=0.25,
    batch_size=200,
    expanding_window_size=8000,
    experiment_name="medical-patient-clustering",
    hospital_profile_id="hospital_001"
)
```

## Performance Monitoring

The pipeline includes comprehensive performance monitoring:

### Metrics Tracked

- **Clustering Metrics**: Silhouette score, cluster count, noise ratio
- **Model Metrics**: RMSE, MAE, R², feature importance
- **Processing Metrics**: Batch processing time, memory usage
- **Drift Metrics**: Noise ratio trends, performance degradation

### Alerting

```python
# Configure performance thresholds
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

# Get alerts
alerts = pipeline.get_alert_summary()
health = pipeline.perform_health_check()
```

## Adaptive Retraining

The pipeline supports multiple retraining strategies:

### Automatic Retraining

Triggered when noise ratio exceeds threshold:

```python
config = PipelineConfig(noise_threshold=0.25)  # Retrain at 25% noise
```

### Manual Retraining

```python
pipeline.retrain("High noise ratio detected in recent batches")
```

### Scheduled Retraining

```python
from datetime import timedelta

# Check for scheduled retraining every 24 hours
retraining_manager.check_scheduled_retraining(
    schedule_interval=timedelta(hours=24)
)
```

## Error Handling

Comprehensive error handling with graceful degradation:

```python
# Enable graceful degradation
pipeline.enable_graceful_degradation(True)

# Get error statistics
error_stats = pipeline.get_error_statistics()
recent_errors = pipeline.get_recent_errors(limit=5)

# Perform health diagnosis
health_report = pipeline.diagnose_pipeline_health()
```

## Testing

Run the example scripts to test your installation:

```bash
# Test basic functionality
python mlflow_timeseries_clustering/examples/initial_training_example.py

# Test batch processing
python mlflow_timeseries_clustering/examples/batch_processing_example.py

# Test retraining
python mlflow_timeseries_clustering/examples/retraining_configuration_example.py
```

## Requirements

- Python 3.8+
- MLflow 2.0+
- HDBSCAN 0.8+
- LightGBM 3.0+
- scikit-learn 1.0+
- NumPy 1.20+
- Pandas 1.3+
- Seaborn 0.11+
- Matplotlib 3.3+

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For issues and questions:

1. Check the [Troubleshooting Guide](docs/troubleshooting_guide.md)
2. Review the [API Documentation](docs/api_documentation.md)
3. Run the diagnostic tools provided in the examples
4. Create an issue with detailed error information

## Changelog

### Version 1.0.0

- Initial release with complete pipeline functionality
- HDBSCAN clustering with kNN fallback
- LightGBM time series and resource models
- MLflow integration and model registry
- Adaptive retraining mechanism
- Performance monitoring and alerting
- Comprehensive documentation and examples