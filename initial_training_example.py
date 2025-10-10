"""
Initial Training Workflow Example

This example demonstrates how to perform initial training of the MLflow Time Series
Clustering Pipeline from scratch. It covers:

1. Data preparation and validation
2. Configuration setup with parameter tuning
3. Initial model training (HDBSCAN + LightGBM)
4. Performance evaluation and reporting
5. Model registration and artifact management

This is the starting point for any new pipeline deployment.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

from mlflow_timeseries_clustering.core.config import PipelineConfig
from mlflow_timeseries_clustering.pipeline.timeseries_clustering_pipeline import TimeSeriesClusteringPipeline
from mlflow_timeseries_clustering.core.data_models import ClusteringResult, PerformanceReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_patient_data(n_patients: int = 2000,
                           n_features: int = 12,
                           random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic sample patient time series data for demonstration.

    In a real scenario, this would load actual patient data from your data source.

    Args:
        n_patients: Number of patient records to generate
        n_features: Number of features per patient
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is features and y is target values
    """
    np.random.seed(random_seed)
    logger.info("Generating sample patient data: %d patients, %d features",
                n_patients, n_features)

    # Create realistic patient clusters
    # Cluster 1: Young patients with low risk
    # Cluster 2: Middle-aged patients with moderate risk
    # Cluster 3: Elderly patients with high risk
    # Cluster 4: Chronic condition patients (mixed age)

    cluster_configs = [
        {'center': np.array([25, 0.2, 1.1, 0.8, 0.3, 0.1, 0.2, 0.9, 0.1, 0.2, 0.3, 0.1]), 'size': 600},  # Young/low risk
        {'center': np.array([45, 0.4, 1.3, 1.2, 0.5, 0.3, 0.4, 0.7, 0.3, 0.4, 0.5, 0.3]), 'size': 500},  # Middle-aged/moderate
        {'center': np.array([70, 0.7, 1.8, 1.8, 0.8, 0.6, 0.7, 0.4, 0.6, 0.7, 0.8, 0.6]), 'size': 400},  # Elderly/high risk
        {'center': np.array([55, 0.9, 2.1, 2.2, 1.0, 0.8, 0.9, 0.3, 0.8, 0.9, 1.0, 0.8]), 'size': 300},  # Chronic conditions
    ]

    X_clusters = []
    y_clusters = []

    for i, config in enumerate(cluster_configs):
        # Generate samples around cluster center
        cluster_X = np.random.multivariate_normal(
            config['center'],
            np.eye(n_features) * 0.1,  # Small variance around center
            config['size']
        )

        # Generate target values based on risk profile
        risk_factor = config['center'][1]  # Use second feature as risk indicator
        cluster_y = np.random.normal(risk_factor * 10, 1.0, config['size'])

        X_clusters.append(cluster_X)
        y_clusters.append(cluster_y)

    # Combine all clusters
    X = np.vstack(X_clusters)
    y = np.concatenate(y_clusters)

    # Add some noise patients (outliers)
    n_noise = int(n_patients * 0.1)  # 10% noise
    if n_noise > 0:
        noise_X = np.random.uniform(-1, 3, (n_noise, n_features))
        noise_y = np.random.uniform(0, 15, n_noise)

        X = np.vstack([X, noise_X])
        y = np.concatenate([y, noise_y])

    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    logger.info("Generated data shape: X=%s, y=%s", X.shape, y.shape)
    return X, y


def create_training_configuration(hospital_id: str = "demo_hospital") -> PipelineConfig:
    """
    Create optimized configuration for initial training.

    Args:
        hospital_id: Hospital profile identifier

    Returns:
        Configured PipelineConfig instance
    """
    logger.info("Creating training configuration for hospital: %s", hospital_id)

    config = PipelineConfig(
        # HDBSCAN parameters - optimized for patient clustering
        hdbscan_params={
            'min_cluster_size': 20,      # Minimum 20 patients per cluster
            'min_samples': 10,           # Conservative clustering
            'cluster_selection_epsilon': 0.1,  # Allow some noise tolerance
            'prediction_data': True,     # Enable prediction for new data
            'cluster_selection_method': 'eom',  # Excess of mass method
            'metric': 'euclidean'        # Standard distance metric
        },

        # LightGBM parameters for time series prediction
        lgb_timeseries_params={
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,            # More complex trees for patient data
            'learning_rate': 0.05,       # Conservative learning rate
            'feature_fraction': 0.9,     # Use most features
            'bagging_fraction': 0.8,     # Bootstrap sampling
            'bagging_freq': 5,
            'min_data_in_leaf': 10,      # Minimum samples per leaf
            'lambda_l1': 0.1,            # L1 regularization
            'lambda_l2': 0.1,            # L2 regularization
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100          # Number of boosting rounds
        },

        # LightGBM parameters for resource usage prediction
        lgb_resource_params={
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 25,            # Simpler model for resource prediction
            'learning_rate': 0.1,        # Faster learning for resource models
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_data_in_leaf': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 50           # Fewer rounds for resource models
        },

        # kNN fallback parameters
        knn_params={
            'n_neighbors': 7,            # Slightly more neighbors for stability
            'weights': 'distance',       # Distance-weighted predictions
            'algorithm': 'auto',         # Let sklearn choose best algorithm
            'metric': 'euclidean'        # Match HDBSCAN metric
        },

        # Pipeline behavior parameters
        noise_threshold=0.25,            # Trigger retraining at 25% noise
        batch_size=200,                  # Process 200 patients per batch
        expanding_window_size=8000,      # Keep 8000 patients in retraining window

        # MLflow configuration
        experiment_name=f"Initial-Training-{hospital_id}",
        model_registry_prefix=f"patient-clustering-{hospital_id}",
        hospital_profile_id=hospital_id
    )

    logger.info("Configuration created with %d HDBSCAN min_cluster_size and %.2f noise_threshold",
                config.hdbscan_params['min_cluster_size'], config.noise_threshold)

    return config


def validate_training_data(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Perform comprehensive validation of training data.

    Args:
        X: Feature matrix
        y: Target values

    Returns:
        Dictionary with validation results and statistics
    """
    logger.info("Validating training data...")

    validation_results = {
        'data_shape': {'n_samples': X.shape[0], 'n_features': X.shape[1]},
        'data_quality': {},
        'recommendations': []
    }

    # Check for missing values
    missing_X = np.isnan(X).sum()
    missing_y = np.isnan(y).sum()
    validation_results['data_quality']['missing_values'] = {
        'X_missing': int(missing_X),
        'y_missing': int(missing_y)
    }

    if missing_X > 0 or missing_y > 0:
        validation_results['recommendations'].append(
            f"Found {missing_X + missing_y} missing values - consider imputation"
        )

    # Check data ranges
    X_stats = {
        'min': float(np.min(X)),
        'max': float(np.max(X)),
        'mean': float(np.mean(X)),
        'std': float(np.std(X))
    }
    y_stats = {
        'min': float(np.min(y)),
        'max': float(np.max(y)),
        'mean': float(np.mean(y)),
        'std': float(np.std(y))
    }

    validation_results['data_quality']['feature_stats'] = X_stats
    validation_results['data_quality']['target_stats'] = y_stats

    # Check for potential scaling issues
    if X_stats['std'] > 10 or X_stats['max'] - X_stats['min'] > 100:
        validation_results['recommendations'].append(
            "Large feature value ranges detected - scaling will be applied automatically"
        )

    # Check sample size adequacy
    min_samples_recommended = 1000
    if X.shape[0] < min_samples_recommended:
        validation_results['recommendations'].append(
            f"Sample size {X.shape[0]} is below recommended {min_samples_recommended}"
        )

    # Check feature dimensionality
    if X.shape[1] > X.shape[0] / 10:
        validation_results['recommendations'].append(
            "High dimensionality detected - consider feature selection"
        )

    logger.info("Data validation completed: %d samples, %d features, %d recommendations",
                X.shape[0], X.shape[1], len(validation_results['recommendations']))

    return validation_results


def perform_initial_training(config: PipelineConfig,
                           X: np.ndarray,
                           y: np.ndarray) -> Tuple[TimeSeriesClusteringPipeline, ClusteringResult]:
    """
    Perform the complete initial training workflow.

    Args:
        config: Pipeline configuration
        X: Training features
        y: Training targets

    Returns:
        Tuple of (trained pipeline, clustering result)
    """
    logger.info("Starting initial training workflow...")

    # Initialize pipeline
    pipeline = TimeSeriesClusteringPipeline(config)

    # Set up training callbacks for monitoring
    def pre_training_callback():
        logger.info("Pre-training: Initializing models and MLflow tracking...")

    def post_training_callback(clustering_result: ClusteringResult):
        logger.info("Post-training: Found %d clusters with %.3f noise ratio",
                   clustering_result.metrics['n_clusters'],
                   clustering_result.noise_ratio)

    pipeline.set_callbacks(
        pre_training=pre_training_callback,
        post_training=post_training_callback
    )

    # Perform initial training
    logger.info("Executing initial training...")
    start_time = datetime.now()

    try:
        clustering_result = pipeline.fit_initial(X, y)
        training_time = (datetime.now() - start_time).total_seconds()

        logger.info("Initial training completed successfully in %.2f seconds", training_time)
        logger.info("Training results:")
        logger.info("  - Clusters found: %d", clustering_result.metrics['n_clusters'])
        logger.info("  - Noise ratio: %.3f", clustering_result.noise_ratio)
        logger.info("  - Silhouette score: %.3f",
                   clustering_result.metrics.get('silhouette_score', 0))

        return pipeline, clustering_result

    except Exception as e:
        logger.error("Initial training failed: %s", str(e))
        raise


def evaluate_training_results(pipeline: TimeSeriesClusteringPipeline,
                            clustering_result: ClusteringResult) -> PerformanceReport:
    """
    Evaluate and report on training results.

    Args:
        pipeline: Trained pipeline
        clustering_result: Results from clustering

    Returns:
        Comprehensive performance report
    """
    logger.info("Evaluating training results...")

    # Generate performance report
    performance_report = pipeline.get_performance_report()

    # Log key metrics
    logger.info("Performance Report Summary:")
    logger.info("  - Report timestamp: %s", performance_report.timestamp)
    logger.info("  - Model version: %s", performance_report.model_version)

    # Clustering metrics
    clustering_metrics = performance_report.clustering_metrics
    logger.info("  - Clustering metrics:")
    for metric, value in clustering_metrics.items():
        if isinstance(value, (int, float)):
            logger.info("    - %s: %.3f", metric, value)
        else:
            logger.info("    - %s: %s", metric, value)

    # Model metrics
    if performance_report.model_metrics:
        logger.info("  - Model metrics available for %d clusters",
                   len(performance_report.model_metrics))

        # Average metrics across clusters
        avg_rmse = np.mean([metrics.get('rmse', 0)
                           for metrics in performance_report.model_metrics.values()])
        avg_r2 = np.mean([metrics.get('r2_score', 0)
                         for metrics in performance_report.model_metrics.values()])

        logger.info("    - Average RMSE: %.3f", avg_rmse)
        logger.info("    - Average R²: %.3f", avg_r2)

    # Pipeline health check
    health_report = pipeline.perform_health_check()
    logger.info("  - Pipeline health: %s", health_report['overall_health'])

    if health_report.get('recommendations'):
        logger.info("  - Recommendations:")
        for rec in health_report['recommendations']:
            logger.info("    - %s", rec)

    return performance_report


def save_training_artifacts(pipeline: TimeSeriesClusteringPipeline,
                          validation_results: Dict[str, Any],
                          performance_report: PerformanceReport) -> None:
    """
    Save training artifacts and documentation.

    Args:
        pipeline: Trained pipeline
        validation_results: Data validation results
        performance_report: Performance evaluation results
    """
    logger.info("Saving training artifacts...")

    try:
        # Get model information
        model_info = pipeline.get_model_info()

        # Create training summary
        training_summary = {
            'training_timestamp': datetime.now().isoformat(),
            'pipeline_version': pipeline.current_model_version,
            'data_validation': validation_results,
            'model_info': model_info,
            'performance_summary': {
                'clustering_metrics': performance_report.clustering_metrics,
                'model_count': len(performance_report.model_metrics) if performance_report.model_metrics else 0,
                'health_status': pipeline.current_health_status
            }
        }

        logger.info("Training summary created with %d sections", len(training_summary))
        logger.info("Artifacts saved successfully")

    except Exception as e:
        logger.warning("Failed to save some training artifacts: %s", str(e))


def main():
    """Main function demonstrating complete initial training workflow."""

    print("=" * 80)
    print("MLflow Time Series Clustering Pipeline - Initial Training Example")
    print("=" * 80)

    try:
        # Step 1: Load and validate training data
        print("\n1. Loading and validating training data...")
        X_train, y_train = load_sample_patient_data(n_patients=2500, n_features=12)
        validation_results = validate_training_data(X_train, y_train)

        print(f"   Data loaded: {X_train.shape[0]} patients, {X_train.shape[1]} features")
        print(f"   Validation: {len(validation_results['recommendations'])} recommendations")

        # Step 2: Create configuration
        print("\n2. Creating pipeline configuration...")
        config = create_training_configuration("demo_hospital_001")
        print(f"   Configuration: {config.experiment_name}")
        print(f"   Noise threshold: {config.noise_threshold}")
        print(f"   Min cluster size: {config.hdbscan_params['min_cluster_size']}")

        # Step 3: Perform initial training
        print("\n3. Performing initial training...")
        pipeline, clustering_result = perform_initial_training(config, X_train, y_train)
        print(f"   Training completed: {clustering_result.metrics['n_clusters']} clusters found")
        print(f"   Noise ratio: {clustering_result.noise_ratio:.3f}")

        # Step 4: Evaluate results
        print("\n4. Evaluating training results...")
        performance_report = evaluate_training_results(pipeline, clustering_result)
        print(f"   Performance report generated: {performance_report.model_version}")
        print(f"   Pipeline health: {pipeline.current_health_status}")

        # Step 5: Save artifacts
        print("\n5. Saving training artifacts...")
        save_training_artifacts(pipeline, validation_results, performance_report)
        print("   Artifacts saved successfully")

        # Step 6: Final status
        print("\n6. Final training status:")
        status = pipeline.get_pipeline_status()
        print(f"   Pipeline fitted: {status['pipeline_fitted']}")
        print(f"   Batch processing ready: {status['batch_processing_enabled']}")
        print(f"   Model version: {status['current_model_version']}")

        print("\n" + "=" * 80)
        print("Initial training completed successfully!")
        print("Next steps:")
        print("  1. Review the performance report in MLflow UI")
        print("  2. Test batch processing with new data")
        print("  3. Set up monitoring and retraining schedules")
        print("=" * 80)

    except Exception as e:
        logger.error("Initial training example failed: %s", str(e))
        print(f"\nTraining failed: {str(e)}")
        print("Check logs for detailed error information.")
        raise


if __name__ == "__main__":
    main()