"""
Example usage of the Adaptive Retraining Mechanism.

This script demonstrates how to use the adaptive retraining components
including trigger monitoring, expanded window retraining, and reporting.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline components
from mlflow_timeseries_clustering.core.config import PipelineConfig
from mlflow_timeseries_clustering.core.data_models import BatchResult
from mlflow_timeseries_clustering.clustering.adaptive_clustering_engine import AdaptiveClusteringEngine
from mlflow_timeseries_clustering.models.cluster_specific_model_manager import ClusterSpecificModelManager
from mlflow_timeseries_clustering.mlflow_integration.experiment_manager import ExperimentManager
from mlflow_timeseries_clustering.mlflow_integration.artifact_manager import ArtifactManager
from mlflow_timeseries_clustering.monitoring import (
    AdaptiveRetrainingManager,
    RetrainingTrigger,
    TriggerReason
)


def generate_sample_data(n_samples: int = 1000, n_features: int = 10) -> tuple:
    """Generate sample time series data for demonstration."""
    np.random.seed(42)

    # Generate features with some structure
    X = np.random.randn(n_samples, n_features)

    # Add some cluster structure
    cluster_centers = np.random.randn(3, n_features) * 2
    cluster_assignments = np.random.choice(3, n_samples)

    for i, center in enumerate(cluster_centers):
        mask = cluster_assignments == i
        X[mask] += center + np.random.randn(np.sum(mask), n_features) * 0.5

    # Generate target values
    y = np.random.randn(n_samples) + np.sum(X[:, :3], axis=1) * 0.1

    return X, y


def simulate_batch_processing(retraining_manager: AdaptiveRetrainingManager,
                            n_batches: int = 10,
                            batch_size: int = 100) -> None:
    """Simulate batch processing with potential retraining triggers."""
    logger.info("Starting batch processing simulation with %d batches", n_batches)

    for batch_idx in range(n_batches):
        # Generate batch data
        X_batch, y_batch = generate_sample_data(batch_size, 10)

        # Simulate varying noise ratios (some batches will trigger retraining)
        if batch_idx > 5:
            # Introduce higher noise ratios in later batches
            noise_ratio = 0.2 + (batch_idx - 5) * 0.1
        else:
            noise_ratio = 0.1 + np.random.random() * 0.1

        # Create batch result
        batch_result = BatchResult(
            cluster_assignments=np.random.choice(3, batch_size),
            timeseries_predictions=np.random.randn(batch_size),
            resource_predictions={0: {'cpu': 1.2, 'memory': 512}, 1: {'cpu': 1.5, 'memory': 768}},
            noise_ratio=noise_ratio,
            processing_time=np.random.uniform(0.5, 2.0)
        )

        logger.info("Processing batch %d with noise ratio: %.3f", batch_idx, noise_ratio)

        # Monitor batch result (may trigger retraining)
        retraining_triggered = retraining_manager.monitor_batch_result(
            batch_result,
            {'X': X_batch, 'y': y_batch}
        )

        if retraining_triggered:
            logger.info("Retraining was triggered for batch %d", batch_idx)

        # Simulate some processing delay
        import time
        time.sleep(0.1)


def demonstrate_manual_retraining(retraining_manager: AdaptiveRetrainingManager) -> None:
    """Demonstrate manual retraining functionality."""
    logger.info("Demonstrating manual retraining")

    # Request manual retraining
    success = retraining_manager.request_manual_retraining(
        "Demonstration of manual retraining capability"
    )

    if success:
        logger.info("Manual retraining was successfully initiated")
    else:
        logger.warning("Manual retraining failed to initiate")


def demonstrate_scheduled_retraining(retraining_manager: AdaptiveRetrainingManager) -> None:
    """Demonstrate scheduled retraining functionality."""
    logger.info("Demonstrating scheduled retraining")

    # Check for scheduled retraining (with a very short interval for demo)
    success = retraining_manager.check_scheduled_retraining(
        schedule_interval=timedelta(seconds=1)  # Very short for demo
    )

    if success:
        logger.info("Scheduled retraining was triggered")
    else:
        logger.info("No scheduled retraining needed")


def main():
    """Main demonstration function."""
    logger.info("Starting Adaptive Retraining Mechanism Demonstration")

    try:
        # 1. Create configuration
        config = PipelineConfig(
            noise_threshold=0.25,  # Lower threshold for demo
            expanding_window_size=5000,
            experiment_name="adaptive-retraining-demo"
        )

        # 2. Initialize MLflow components (mock for demo)
        experiment_manager = ExperimentManager(config)
        artifact_manager = ArtifactManager(config)

        # 3. Create adaptive retraining manager
        retraining_manager = AdaptiveRetrainingManager(
            config=config,
            experiment_manager=experiment_manager,
            artifact_manager=artifact_manager
        )

        # 4. Initialize models for demonstration
        # Generate initial training data
        X_initial, y_initial = generate_sample_data(2000, 10)

        # Create and fit clustering engine
        clustering_engine = AdaptiveClusteringEngine(
            hdbscan_params=config.hdbscan_params,
            knn_params=config.knn_params
        )
        clustering_result = clustering_engine.fit(X_initial)
        logger.info("Initial clustering completed with %d clusters",
                   clustering_result.metrics['n_clusters'])

        # Create model manager
        model_manager = ClusterSpecificModelManager(
            lgb_timeseries_params=config.lgb_timeseries_params,
            lgb_resource_params=config.lgb_resource_params
        )

        # Set models in retraining manager
        retraining_manager.set_current_models(clustering_engine, model_manager)

        # 5. Set up retraining callbacks
        def pre_retraining_callback():
            logger.info("Pre-retraining callback: Preparing for retraining...")

        def post_retraining_callback(result):
            logger.info("Post-retraining callback: Retraining completed with improvement ratio: %.3f",
                       result.model_comparison.get('overall_improvement_ratio', 0))

        retraining_manager.set_retraining_callbacks(
            pre_callback=pre_retraining_callback,
            post_callback=post_retraining_callback
        )

        # 6. Demonstrate batch processing with automatic retraining
        simulate_batch_processing(retraining_manager, n_batches=8, batch_size=200)

        # 7. Demonstrate manual retraining
        demonstrate_manual_retraining(retraining_manager)

        # 8. Demonstrate scheduled retraining
        demonstrate_scheduled_retraining(retraining_manager)

        # 9. Get retraining status and recommendations
        status = retraining_manager.get_retraining_status()
        logger.info("Retraining status: %d total retrainings completed",
                   status['trigger_statistics']['total_retrainings'])

        recommendations = retraining_manager.get_retraining_recommendations()
        logger.info("Retraining recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info("  %d. %s", i, rec)

        logger.info("Adaptive Retraining Mechanism Demonstration completed successfully!")

    except Exception as e:
        logger.error("Demonstration failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()