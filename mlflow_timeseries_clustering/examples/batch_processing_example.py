"""
Batch Processing Example with Real Data

This example demonstrates how to use the trained MLflow Time Series Clustering
Pipeline for processing new data batches in production. It covers:

1. Loading a pre-trained pipeline
2. Processing multiple data batches
3. Monitoring performance and drift detection
4. Handling errors and edge cases
5. Performance optimization strategies

This example simulates a production environment where new patient data
arrives in batches and needs to be processed continuously.
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from mlflow_timeseries_clustering.core.config import PipelineConfig
from mlflow_timeseries_clustering.pipeline.timeseries_clustering_pipeline import TimeSeriesClusteringPipeline
from mlflow_timeseries_clustering.core.data_models import BatchResult
from mlflow_timeseries_clustering.pipeline.performance_monitor import PerformanceAlert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchDataSimulator:
    """Simulates realistic batch data for demonstration purposes."""

    def __init__(self, base_config: Dict[str, Any], random_seed: int = 42):
        """
        Initialize the batch data simulator.

        Args:
            base_config: Base configuration for data generation
            random_seed: Random seed for reproducibility
        """
        self.base_config = base_config
        self.random_seed = random_seed
        self.batch_counter = 0
        np.random.seed(random_seed)

        # Simulate data drift over time
        self.drift_factor = 0.0
        self.noise_trend = 0.1

    def generate_batch(self, batch_size: int = 200,
                      simulate_drift: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a realistic data batch.

        Args:
            batch_size: Number of samples in the batch
            simulate_drift: Whether to simulate data drift

        Returns:
            Tuple of (X, y) batch data
        """
        self.batch_counter += 1

        # Simulate gradual data drift
        if simulate_drift:
            self.drift_factor += 0.02  # Gradual drift
            self.noise_trend += 0.01   # Increasing noise over time

        # Generate base patient data similar to training
        n_features = self.base_config.get('n_features', 12)

        # Create patient clusters with drift
        cluster_centers = [
            np.array([25, 0.2, 1.1, 0.8, 0.3, 0.1, 0.2, 0.9, 0.1, 0.2, 0.3, 0.1]),  # Young
            np.array([45, 0.4, 1.3, 1.2, 0.5, 0.3, 0.4, 0.7, 0.3, 0.4, 0.5, 0.3]),  # Middle-aged
            np.array([70, 0.7, 1.8, 1.8, 0.8, 0.6, 0.7, 0.4, 0.6, 0.7, 0.8, 0.6]),  # Elderly
            np.array([55, 0.9, 2.1, 2.2, 1.0, 0.8, 0.9, 0.3, 0.8, 0.9, 1.0, 0.8]),  # Chronic
        ]

        # Apply drift to cluster centers
        drifted_centers = []
        for center in cluster_centers:
            drifted_center = center + np.random.normal(0, self.drift_factor, len(center))
            drifted_centers.append(drifted_center)

        # Generate samples
        samples_per_cluster = batch_size // len(drifted_centers)
        X_batches = []
        y_batches = []

        for i, center in enumerate(drifted_centers):
            # Generate cluster samples
            cluster_X = np.random.multivariate_normal(
                center,
                np.eye(n_features) * (0.1 + self.noise_trend * 0.1),
                samples_per_cluster
            )

            # Generate target values with drift
            risk_factor = center[1] + self.drift_factor * 0.1
            cluster_y = np.random.normal(risk_factor * 10, 1.0 + self.drift_factor, samples_per_cluster)

            X_batches.append(cluster_X)
            y_batches.append(cluster_y)

        # Combine clusters
        X = np.vstack(X_batches)
        y = np.concatenate(y_batches)

        # Add noise patients (increasing over time)
        noise_ratio = min(0.1 + self.noise_trend * 0.1, 0.4)  # Cap at 40%
        n_noise = int(batch_size * noise_ratio)

        if n_noise > 0:
            noise_X = np.random.uniform(-2, 4, (n_noise, n_features))
            noise_y = np.random.uniform(-5, 20, n_noise)

            X = np.vstack([X, noise_X])
            y = np.concatenate([y, noise_y])

        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Trim to exact batch size
        X = X[:batch_size]
        y = y[:batch_size]

        return X, y

    def get_batch_metadata(self) -> Dict[str, Any]:
        """Get metadata about the current batch state."""
        return {
            'batch_number': self.batch_counter,
            'drift_factor': self.drift_factor,
            'noise_trend': self.noise_trend,
            'expected_noise_ratio': min(0.1 + self.noise_trend * 0.1, 0.4)
        }


def setup_trained_pipeline() -> TimeSeriesClusteringPipeline:
    """
    Set up a pre-trained pipeline for batch processing.

    In a real scenario, this would load an existing trained pipeline
    from MLflow model registry or artifact storage.

    Returns:
        Configured and trained pipeline
    """
    logger.info("Setting up pre-trained pipeline...")

    # Create configuration (same as training)
    config = PipelineConfig(
        hdbscan_params={
            'min_cluster_size': 20,
            'min_samples': 10,
            'cluster_selection_epsilon': 0.1,
            'prediction_data': True,
            'cluster_selection_method': 'eom'
        },
        lgb_timeseries_params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 50,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'verbose': -1
        },
        lgb_resource_params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 25,
            'learning_rate': 0.1,
            'n_estimators': 50,
            'verbose': -1
        },
        knn_params={
            'n_neighbors': 7,
            'weights': 'distance'
        },
        noise_threshold=0.25,
        batch_size=200,
        expanding_window_size=8000,
        experiment_name="Batch-Processing-Demo",
        model_registry_prefix="batch-processing",
        hospital_profile_id="demo_hospital_batch"
    )

    # Initialize pipeline
    pipeline = TimeSeriesClusteringPipeline(config)

    # Simulate initial training (in real scenario, load from MLflow)
    logger.info("Performing initial training for demonstration...")

    # Generate training data
    np.random.seed(42)
    X_train = np.random.randn(2000, 12)
    y_train = np.random.randn(2000)

    # Add cluster structure to training data
    cluster_centers = np.random.randn(4, 12) * 2
    cluster_assignments = np.random.choice(4, 2000)

    for i, center in enumerate(cluster_centers):
        mask = cluster_assignments == i
        X_train[mask] += center

    # Train the pipeline
    clustering_result = pipeline.fit_initial(X_train, y_train)

    logger.info("Pipeline training completed: %d clusters, %.3f noise ratio",
                clustering_result.metrics['n_clusters'], clustering_result.noise_ratio)

    return pipeline


def setup_performance_monitoring(pipeline: TimeSeriesClusteringPipeline) -> None:
    """
    Set up performance monitoring and alerting for batch processing.

    Args:
        pipeline: Pipeline to monitor
    """
    logger.info("Setting up performance monitoring...")

    # Configure performance thresholds
    alert_thresholds = {
        'noise_ratio': 0.3,           # Alert if noise ratio > 30%
        'processing_time': 5.0,       # Alert if processing > 5 seconds
        'prediction_error': 2.0,      # Alert if prediction error > 2.0
        'memory_usage': 1000.0        # Alert if memory usage > 1GB
    }

    drift_thresholds = {
        'noise_ratio_trend': 0.05,    # Alert if noise increasing by 5%
        'performance_degradation': 0.1 # Alert if performance drops by 10%
    }

    pipeline.update_performance_thresholds(
        alert_thresholds=alert_thresholds,
        drift_thresholds=drift_thresholds
    )

    logger.info("Performance monitoring configured with %d alert thresholds",
                len(alert_thresholds))


def process_batch_with_monitoring(pipeline: TimeSeriesClusteringPipeline,
                                X_batch: np.ndarray,
                                y_batch: np.ndarray,
                                batch_metadata: Dict[str, Any]) -> BatchResult:
    """
    Process a single batch with comprehensive monitoring.

    Args:
        pipeline: Trained pipeline
        X_batch: Batch features
        y_batch: Batch targets
        batch_metadata: Metadata about the batch

    Returns:
        Batch processing result
    """
    batch_num = batch_metadata['batch_number']
    logger.info("Processing batch %d with %d samples", batch_num, len(X_batch))

    start_time = time.time()

    try:
        # Process the batch
        batch_result = pipeline.process_batch(X_batch, y_batch)

        processing_time = time.time() - start_time

        # Log batch results
        logger.info("Batch %d processed successfully:", batch_num)
        logger.info("  - Processing time: %.3f seconds", processing_time)
        logger.info("  - Noise ratio: %.3f", batch_result.noise_ratio)
        logger.info("  - Cluster assignments: %d unique clusters",
                   len(np.unique(batch_result.cluster_assignments)))

        # Check for performance issues
        if processing_time > 3.0:
            logger.warning("Batch %d processing time %.3f exceeds threshold",
                          batch_num, processing_time)

        if batch_result.noise_ratio > 0.3:
            logger.warning("Batch %d noise ratio %.3f exceeds threshold",
                          batch_num, batch_result.noise_ratio)

        # Compare with expected values
        expected_noise = batch_metadata.get('expected_noise_ratio', 0.1)
        noise_diff = abs(batch_result.noise_ratio - expected_noise)

        if noise_diff > 0.1:
            logger.info("Batch %d noise ratio differs from expected by %.3f",
                       batch_num, noise_diff)

        return batch_result

    except Exception as e:
        logger.error("Batch %d processing failed: %s", batch_num, str(e))
        raise


def analyze_batch_trends(batch_results: List[BatchResult],
                        batch_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze trends across multiple batches.

    Args:
        batch_results: List of batch processing results
        batch_metadata_list: List of batch metadata

    Returns:
        Trend analysis results
    """
    logger.info("Analyzing trends across %d batches", len(batch_results))

    # Extract metrics
    noise_ratios = [result.noise_ratio for result in batch_results]
    processing_times = [result.processing_time for result in batch_results]

    # Calculate trends
    noise_trend = np.polyfit(range(len(noise_ratios)), noise_ratios, 1)[0]
    time_trend = np.polyfit(range(len(processing_times)), processing_times, 1)[0]

    # Calculate statistics
    analysis = {
        'batch_count': len(batch_results),
        'noise_ratio_stats': {
            'mean': float(np.mean(noise_ratios)),
            'std': float(np.std(noise_ratios)),
            'min': float(np.min(noise_ratios)),
            'max': float(np.max(noise_ratios)),
            'trend': float(noise_trend)
        },
        'processing_time_stats': {
            'mean': float(np.mean(processing_times)),
            'std': float(np.std(processing_times)),
            'min': float(np.min(processing_times)),
            'max': float(np.max(processing_times)),
            'trend': float(time_trend)
        },
        'drift_indicators': {
            'noise_increasing': noise_trend > 0.01,
            'performance_degrading': time_trend > 0.1,
            'retraining_recommended': max(noise_ratios[-3:]) > 0.25 if len(noise_ratios) >= 3 else False
        }
    }

    # Log analysis results
    logger.info("Batch trend analysis:")
    logger.info("  - Average noise ratio: %.3f (trend: %+.4f)",
               analysis['noise_ratio_stats']['mean'], noise_trend)
    logger.info("  - Average processing time: %.3f (trend: %+.4f)",
               analysis['processing_time_stats']['mean'], time_trend)
    logger.info("  - Drift indicators: %s", analysis['drift_indicators'])

    return analysis


def demonstrate_error_handling(pipeline: TimeSeriesClusteringPipeline) -> None:
    """
    Demonstrate error handling capabilities during batch processing.

    Args:
        pipeline: Pipeline to test
    """
    logger.info("Demonstrating error handling capabilities...")

    # Test 1: Invalid data shape
    try:
        invalid_X = np.random.randn(10, 5)  # Wrong number of features
        invalid_y = np.random.randn(10)
        pipeline.process_batch(invalid_X, invalid_y)
    except Exception as e:
        logger.info("Test 1 - Invalid data shape handled: %s", str(e)[:100])

    # Test 2: Empty batch
    try:
        empty_X = np.empty((0, 12))
        empty_y = np.empty(0)
        pipeline.process_batch(empty_X, empty_y)
    except Exception as e:
        logger.info("Test 2 - Empty batch handled: %s", str(e)[:100])

    # Test 3: NaN values
    try:
        nan_X = np.random.randn(50, 12)
        nan_X[0, 0] = np.nan
        nan_y = np.random.randn(50)
        pipeline.process_batch(nan_X, nan_y)
    except Exception as e:
        logger.info("Test 3 - NaN values handled: %s", str(e)[:100])

    # Get error statistics
    error_stats = pipeline.get_error_statistics()
    logger.info("Error handling test completed: %d total errors recorded",
               error_stats.get('total_errors', 0))


def optimize_batch_processing(pipeline: TimeSeriesClusteringPipeline,
                            batch_results: List[BatchResult]) -> Dict[str, Any]:
    """
    Analyze batch processing performance and provide optimization recommendations.

    Args:
        pipeline: Pipeline to optimize
        batch_results: Historical batch results

    Returns:
        Optimization recommendations
    """
    logger.info("Analyzing batch processing performance for optimization...")

    # Analyze processing times
    processing_times = [result.processing_time for result in batch_results]
    avg_time = np.mean(processing_times)

    # Analyze noise ratios
    noise_ratios = [result.noise_ratio for result in batch_results]
    avg_noise = np.mean(noise_ratios)

    # Get pipeline status
    status = pipeline.get_pipeline_status()
    model_info = pipeline.get_model_info()

    recommendations = {
        'performance_analysis': {
            'average_processing_time': float(avg_time),
            'processing_time_std': float(np.std(processing_times)),
            'average_noise_ratio': float(avg_noise),
            'noise_ratio_std': float(np.std(noise_ratios))
        },
        'optimization_recommendations': []
    }

    # Generate recommendations
    if avg_time > 2.0:
        recommendations['optimization_recommendations'].append(
            "Consider reducing batch size or optimizing model complexity"
        )

    if avg_noise > 0.2:
        recommendations['optimization_recommendations'].append(
            "High noise ratio detected - consider retraining or adjusting clustering parameters"
        )

    if np.std(processing_times) > avg_time * 0.5:
        recommendations['optimization_recommendations'].append(
            "High processing time variance - investigate batch size consistency"
        )

    # Check model complexity
    if 'clustering' in model_info:
        n_clusters = model_info['clustering'].get('n_clusters', 0)
        if n_clusters > 10:
            recommendations['optimization_recommendations'].append(
                f"High cluster count ({n_clusters}) may impact performance"
            )

    logger.info("Optimization analysis completed with %d recommendations",
                len(recommendations['optimization_recommendations']))

    return recommendations


def main():
    """Main function demonstrating batch processing workflow."""

    print("=" * 80)
    print("MLflow Time Series Clustering Pipeline - Batch Processing Example")
    print("=" * 80)

    try:
        # Step 1: Set up trained pipeline
        print("\n1. Setting up pre-trained pipeline...")
        pipeline = setup_trained_pipeline()
        setup_performance_monitoring(pipeline)
        print(f"   Pipeline ready: {pipeline.current_model_version}")
        print(f"   Health status: {pipeline.current_health_status}")

        # Step 2: Initialize batch data simulator
        print("\n2. Initializing batch data simulator...")
        simulator = BatchDataSimulator({
            'n_features': 12,
            'base_noise_ratio': 0.1
        })
        print("   Batch simulator ready")

        # Step 3: Process multiple batches
        print("\n3. Processing multiple data batches...")
        n_batches = 15
        batch_results = []
        batch_metadata_list = []

        for i in range(n_batches):
            # Generate batch data
            X_batch, y_batch = simulator.generate_batch(batch_size=200)
            batch_metadata = simulator.get_batch_metadata()
            batch_metadata_list.append(batch_metadata)

            # Process batch
            batch_result = process_batch_with_monitoring(
                pipeline, X_batch, y_batch, batch_metadata
            )
            batch_results.append(batch_result)

            # Brief pause to simulate real-time processing
            time.sleep(0.1)

            # Print progress
            if (i + 1) % 5 == 0:
                print(f"   Processed {i + 1}/{n_batches} batches")

        print(f"   All {n_batches} batches processed successfully")

        # Step 4: Analyze batch trends
        print("\n4. Analyzing batch processing trends...")
        trend_analysis = analyze_batch_trends(batch_results, batch_metadata_list)

        print(f"   Average noise ratio: {trend_analysis['noise_ratio_stats']['mean']:.3f}")
        print(f"   Average processing time: {trend_analysis['processing_time_stats']['mean']:.3f}s")
        print(f"   Retraining recommended: {trend_analysis['drift_indicators']['retraining_recommended']}")

        # Step 5: Demonstrate error handling
        print("\n5. Testing error handling capabilities...")
        demonstrate_error_handling(pipeline)
        print("   Error handling tests completed")

        # Step 6: Performance optimization analysis
        print("\n6. Analyzing performance optimization opportunities...")
        optimization_results = optimize_batch_processing(pipeline, batch_results)

        print(f"   Performance analysis completed")
        print(f"   Optimization recommendations: {len(optimization_results['optimization_recommendations'])}")

        for i, rec in enumerate(optimization_results['optimization_recommendations'], 1):
            print(f"     {i}. {rec}")

        # Step 7: Final monitoring summary
        print("\n7. Final monitoring summary...")

        # Get performance metrics
        metrics = pipeline.get_performance_metrics()
        alert_summary = pipeline.get_alert_summary()
        health_report = pipeline.perform_health_check()

        print(f"   Performance metrics tracked: {len(metrics)}")
        print(f"   Total alerts: {alert_summary.get('total_alerts', 0)}")
        print(f"   Pipeline health: {health_report['overall_health']}")

        # Check if retraining is needed
        if trend_analysis['drift_indicators']['retraining_recommended']:
            print("\n   🔄 RETRAINING RECOMMENDED")
            print("   High noise ratios detected in recent batches")
            print("   Consider triggering pipeline retraining")

        print("\n" + "=" * 80)
        print("Batch processing example completed successfully!")
        print("Key insights:")
        print(f"  - Processed {n_batches} batches with average noise ratio {trend_analysis['noise_ratio_stats']['mean']:.3f}")
        print(f"  - Average processing time: {trend_analysis['processing_time_stats']['mean']:.3f} seconds")
        print(f"  - Pipeline health: {health_report['overall_health']}")
        print("=" * 80)

    except Exception as e:
        logger.error("Batch processing example failed: %s", str(e))
        print(f"\nBatch processing failed: {str(e)}")
        print("Check logs for detailed error information.")
        raise


if __name__ == "__main__":
    main()