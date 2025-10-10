"""
Complete pipeline example demonstrating the TimeSeriesClusteringPipeline usage.

This example shows how to use the main pipeline controller for:
1. Initial training with HDBSCAN clustering and LightGBM models
2. Batch processing with error handling and performance monitoring
3. Adaptive retraining with drift detection
4. Comprehensive performance reporting
"""

import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any

from mlflow_timeseries_clustering.core.config import PipelineConfig
from mlflow_timeseries_clustering.pipeline.timeseries_clustering_pipeline import TimeSeriesClusteringPipeline
from mlflow_timeseries_clustering.pipeline.performance_monitor import PerformanceAlert


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 1000, n_features: int = 10,
                      n_clusters: int = 3, noise_ratio: float = 0.1) -> tuple:
    """
    Create sample time series data for demonstration.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        n_clusters: Number of underlying clusters
        noise_ratio: Ratio of noise points to add

    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(42)

    # Generate cluster centers
    cluster_centers = np.random.randn(n_clusters, n_features) * 3

    # Generate samples around cluster centers
    samples_per_cluster = n_samples // n_clusters
    X = []
    y = []

    for i, center in enumerate(cluster_centers):
        # Generate samples around this cluster center
        cluster_samples = np.random.randn(samples_per_cluster, n_features) + center
        cluster_targets = np.random.randn(samples_per_cluster) * 0.5 + i * 2

        X.append(cluster_samples)
        y.append(cluster_targets)

    # Combine all samples
    X = np.vstack(X)
    y = np.concatenate(y)

    # Add noise points
    n_noise = int(n_samples * noise_ratio)
    if n_noise > 0:
        noise_X = np.random.randn(n_noise, n_features) * 5
        noise_y = np.random.randn(n_noise) * 2

        X = np.vstack([X, noise_X])
        y = np.concatenate([y, noise_y])

    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def performance_alert_handler(alert: PerformanceAlert) -> None:
    """
    Handle performance alerts from the pipeline.

    Args:
        alert: Performance alert to handle
    """
    print(f"\n🚨 PERFORMANCE ALERT [{alert.level.value.upper()}]")
    print(f"   Component: {alert.component}")
    print(f"   Metric: {alert.metric}")
    print(f"   Message: {alert.message}")
    print(f"   Current Value: {alert.current_value:.3f}")
    print(f"   Threshold: {alert.threshold_value:.3f}")
    print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main example function demonstrating complete pipeline usage."""

    print("🚀 MLflow Time Series Clustering Pipeline - Complete Example")
    print("=" * 60)

    # Step 1: Create pipeline configuration
    print("\n📋 Step 1: Creating pipeline configuration...")

    config = PipelineConfig(
        # HDBSCAN parameters
        hdbscan_params={
            'min_cluster_size': 15,
            'min_samples': 5,
            'cluster_selection_epsilon': 0.0,
            'prediction_data': True
        },

        # LightGBM parameters for time series models
        lgb_timeseries_params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1
        },

        # LightGBM parameters for resource models
        lgb_resource_params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.1,
            'verbose': -1
        },

        # kNN fallback parameters
        knn_params={
            'n_neighbors': 5,
            'weights': 'distance'
        },

        # Pipeline parameters
        noise_threshold=0.3,
        batch_size=100,
        expanding_window_size=5000,

        # MLflow configuration
        experiment_name="Complete-Pipeline-Example",
        model_registry_prefix="example-pipeline",
        hospital_profile_id="demo_hospital"
    )

    print(f"   ✅ Configuration created with noise threshold: {config.noise_threshold}")

    # Step 2: Initialize pipeline
    print("\n🏗️  Step 2: Initializing pipeline...")

    pipeline = TimeSeriesClusteringPipeline(config)

    # Set up callbacks for demonstration
    def pre_training_callback():
        print("   🔄 Pre-training callback: Preparing for initial training...")

    def post_training_callback(clustering_result):
        print(f"   ✅ Post-training callback: Found {clustering_result.metrics['n_clusters']} clusters")

    def pre_batch_callback(X_batch):
        print(f"   🔄 Pre-batch callback: Processing batch with {len(X_batch)} samples...")

    def post_batch_callback(batch_result):
        print(f"   ✅ Post-batch callback: Noise ratio {batch_result.noise_ratio:.3f}")

    pipeline.set_callbacks(
        pre_training=pre_training_callback,
        post_training=post_training_callback,
        pre_batch=pre_batch_callback,
        post_batch=post_batch_callback
    )

    print("   ✅ Pipeline initialized with callbacks")

    # Step 3: Generate training data
    print("\n📊 Step 3: Generating training data...")

    X_train, y_train = create_sample_data(n_samples=2000, n_features=8, n_clusters=4)
    print(f"   ✅ Generated training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Step 4: Initial training
    print("\n🎯 Step 4: Performing initial training...")

    try:
        clustering_result = pipeline.fit_initial(X_train, y_train)

        print(f"   ✅ Initial training completed successfully!")
        print(f"   📈 Clusters found: {clustering_result.metrics['n_clusters']}")
        print(f"   📈 Noise ratio: {clustering_result.noise_ratio:.3f}")
        print(f"   📈 Silhouette score: {clustering_result.metrics.get('silhouette_score', 'N/A')}")

    except Exception as e:
        print(f"   ❌ Initial training failed: {str(e)}")
        return

    # Step 5: Generate performance report
    print("\n📊 Step 5: Generating initial performance report...")

    try:
        performance_report = pipeline.get_performance_report()
        print(f"   ✅ Performance report generated at {performance_report.timestamp}")
        print(f"   📈 Model version: {performance_report.model_version}")

    except Exception as e:
        print(f"   ⚠️  Performance report generation failed: {str(e)}")

    # Step 6: Batch processing demonstration
    print("\n🔄 Step 6: Demonstrating batch processing...")

    # Generate several batches for processing
    n_batches = 5
    batch_results = []

    for i in range(n_batches):
        print(f"\n   Processing batch {i+1}/{n_batches}...")

        # Generate batch data (with some variation to simulate real data)
        X_batch, y_batch = create_sample_data(
            n_samples=150,
            n_features=8,
            n_clusters=4,
            noise_ratio=0.1 + i * 0.05  # Gradually increase noise
        )

        try:
            batch_result = pipeline.process_batch(X_batch, y_batch)
            batch_results.append(batch_result)

            print(f"     ✅ Batch processed: noise_ratio={batch_result.noise_ratio:.3f}, "
                  f"time={batch_result.processing_time:.3f}s")

        except Exception as e:
            print(f"     ❌ Batch processing failed: {str(e)}")

    # Step 7: Performance monitoring demonstration
    print("\n📊 Step 7: Checking performance monitoring...")

    try:
        # Get performance metrics
        metrics = pipeline.get_performance_metrics()
        print(f"   ✅ Retrieved metrics for {len(metrics)} metric types")

        # Get alert summary
        alert_summary = pipeline.get_alert_summary()
        print(f"   📊 Alert summary: {alert_summary['total_alerts']} total alerts")
        print(f"     - Critical: {alert_summary['critical_alerts']}")
        print(f"     - Warning: {alert_summary['warning_alerts']}")

        # Perform health check
        health_report = pipeline.perform_health_check()
        print(f"   🏥 Health status: {health_report['overall_health']}")

        if health_report['recommendations']:
            print("   💡 Recommendations:")
            for rec in health_report['recommendations']:
                print(f"     - {rec}")

    except Exception as e:
        print(f"   ⚠️  Performance monitoring check failed: {str(e)}")

    # Step 8: Error handling demonstration
    print("\n🛡️  Step 8: Demonstrating error handling...")

    try:
        # Get error statistics
        error_stats = pipeline.get_error_statistics()
        print(f"   📊 Error statistics: {error_stats['total_errors']} total errors")

        # Get recent errors
        recent_errors = pipeline.get_recent_errors(limit=3)
        if recent_errors:
            print(f"   📋 Recent errors: {len(recent_errors)} found")
        else:
            print("   ✅ No recent errors found")

        # Demonstrate graceful degradation
        pipeline.enable_graceful_degradation(True)
        print("   ✅ Graceful degradation enabled")

    except Exception as e:
        print(f"   ⚠️  Error handling demonstration failed: {str(e)}")

    # Step 9: Pipeline status and diagnostics
    print("\n🔍 Step 9: Pipeline status and diagnostics...")

    try:
        # Get pipeline status
        status = pipeline.get_pipeline_status()
        print(f"   📊 Pipeline Status:")
        print(f"     - Fitted: {status['pipeline_fitted']}")
        print(f"     - Batch processing enabled: {status['batch_processing_enabled']}")
        print(f"     - Model version: {status['current_model_version']}")
        print(f"     - Health status: {pipeline.current_health_status}")

        # Get model information
        model_info = pipeline.get_model_info()
        if 'clustering' in model_info:
            clustering_info = model_info['clustering']
            print(f"     - Clusters: {clustering_info['n_clusters']}")
            print(f"     - Noise ratio: {clustering_info['noise_ratio']:.3f}")

        # Perform comprehensive health diagnosis
        health_diagnosis = pipeline.diagnose_pipeline_health()
        print(f"   🏥 Health Diagnosis:")
        print(f"     - Overall status: {health_diagnosis['pipeline_status']['pipeline_fitted']}")
        print(f"     - Component health checks: {len(health_diagnosis['component_health'])}")

    except Exception as e:
        print(f"   ⚠️  Status check failed: {str(e)}")

    # Step 10: Manual retraining demonstration
    print("\n🔄 Step 10: Demonstrating manual retraining...")

    try:
        # Trigger manual retraining
        pipeline.retrain("Manual demonstration retraining")
        print("   ✅ Manual retraining triggered successfully")

    except Exception as e:
        print(f"   ⚠️  Manual retraining failed: {str(e)}")

    # Step 11: Final summary
    print("\n📋 Step 11: Final Summary")
    print("=" * 40)

    try:
        final_status = pipeline.get_pipeline_status()
        monitoring_stats = pipeline.monitoring_statistics

        print(f"Pipeline Status: {'✅ Healthy' if final_status['pipeline_fitted'] else '❌ Not Ready'}")
        print(f"Model Version: {final_status['current_model_version']}")
        print(f"Batches Processed: {monitoring_stats.get('total_batches_monitored', 0)}")
        print(f"Health Checks: {monitoring_stats.get('health_checks_performed', 0)}")
        print(f"Drift Detections: {monitoring_stats.get('drift_detections', 0)}")

        print("\n🎉 Complete pipeline example finished successfully!")

    except Exception as e:
        print(f"❌ Final summary failed: {str(e)}")

    print("\n" + "=" * 60)
    print("Example completed. Check MLflow UI for detailed experiment tracking.")


if __name__ == "__main__":
    main()