"""
Retraining Configuration Example

This example demonstrates various retraining scenarios and configuration options
for the MLflow Time Series Clustering Pipeline. It covers:

1. Automatic retraining triggered by noise thresholds
2. Manual retraining with custom configurations
3. Scheduled retraining workflows
4. Performance comparison between model versions
5. Advanced retraining strategies and optimization

This example shows how to configure and manage the adaptive retraining
mechanism for different production scenarios.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import time

from mlflow_timeseries_clustering.core.config import PipelineConfig
from mlflow_timeseries_clustering.pipeline.timeseries_clustering_pipeline import TimeSeriesClusteringPipeline
from mlflow_timeseries_clustering.core.data_models import BatchResult
from mlflow_timeseries_clustering.monitoring.adaptive_retraining_manager import AdaptiveRetrainingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainingScenarioManager:
    """Manages different retraining scenarios for demonstration."""

    def __init__(self, base_config: PipelineConfig):
        """
        Initialize the retraining scenario manager.

        Args:
            base_config: Base pipeline configuration
        """
        self.base_config = base_config
        self.scenario_configs = {}
        self.scenario_results = {}

    def create_conservative_config(self) -> PipelineConfig:
        """Create configuration for conservative retraining strategy."""
        config = PipelineConfig(
            # Conservative HDBSCAN parameters
            hdbscan_params={
                'min_cluster_size': 30,      # Larger clusters for stability
                'min_samples': 15,           # More conservative clustering
                'cluster_selection_epsilon': 0.05,  # Less noise tolerance
                'prediction_data': True,
                'cluster_selection_method': 'eom'
            },

            # Stable LightGBM parameters
            lgb_timeseries_params={
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.03,      # Slower learning
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'lambda_l1': 0.2,           # More regularization
                'lambda_l2': 0.2,
                'verbose': -1,
                'n_estimators': 150         # More trees for stability
            },

            lgb_resource_params={
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 20,
                'learning_rate': 0.05,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'n_estimators': 100
            },

            knn_params={
                'n_neighbors': 10,           # More neighbors for stability
                'weights': 'distance'
            },

            # Conservative retraining thresholds
            noise_threshold=0.35,            # Higher threshold
            batch_size=300,                  # Larger batches
            expanding_window_size=12000,     # Larger window

            experiment_name="Conservative-Retraining",
            model_registry_prefix="conservative-model",
            hospital_profile_id="conservative_hospital"
        )

        self.scenario_configs['conservative'] = config
        return config

    def create_aggressive_config(self) -> PipelineConfig:
        """Create configuration for aggressive retraining strategy."""
        config = PipelineConfig(
            # Aggressive HDBSCAN parameters
            hdbscan_params={
                'min_cluster_size': 10,      # Smaller clusters
                'min_samples': 5,            # Less conservative
                'cluster_selection_epsilon': 0.2,  # More noise tolerance
                'prediction_data': True,
                'cluster_selection_method': 'leaf'  # More sensitive to outliers
            },

            # Adaptive LightGBM parameters
            lgb_timeseries_params={
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 63,           # More complex trees
                'learning_rate': 0.08,      # Faster learning
                'feature_fraction': 0.95,
                'bagging_fraction': 0.9,
                'lambda_l1': 0.05,          # Less regularization
                'lambda_l2': 0.05,
                'verbose': -1,
                'n_estimators': 80          # Fewer trees for speed
            },

            lgb_resource_params={
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.12,
                'lambda_l1': 0.02,
                'lambda_l2': 0.02,
                'verbose': -1,
                'n_estimators': 60
            },

            knn_params={
                'n_neighbors': 3,            # Fewer neighbors for sensitivity
                'weights': 'distance'
            },

            # Aggressive retraining thresholds
            noise_threshold=0.15,            # Lower threshold
            batch_size=150,                  # Smaller batches
            expanding_window_size=6000,      # Smaller window

            experiment_name="Aggressive-Retraining",
            model_registry_prefix="aggressive-model",
            hospital_profile_id="aggressive_hospital"
        )

        self.scenario_configs['aggressive'] = config
        return config

    def create_balanced_config(self) -> PipelineConfig:
        """Create configuration for balanced retraining strategy."""
        config = PipelineConfig(
            # Balanced HDBSCAN parameters
            hdbscan_params={
                'min_cluster_size': 20,
                'min_samples': 8,
                'cluster_selection_epsilon': 0.1,
                'prediction_data': True,
                'cluster_selection_method': 'eom'
            },

            # Balanced LightGBM parameters
            lgb_timeseries_params={
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 45,
                'learning_rate': 0.06,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.8,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'n_estimators': 100
            },

            lgb_resource_params={
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 25,
                'learning_rate': 0.08,
                'lambda_l1': 0.05,
                'lambda_l2': 0.05,
                'verbose': -1,
                'n_estimators': 75
            },

            knn_params={
                'n_neighbors': 6,
                'weights': 'distance'
            },

            # Balanced retraining thresholds
            noise_threshold=0.25,
            batch_size=200,
            expanding_window_size=8000,

            experiment_name="Balanced-Retraining",
            model_registry_prefix="balanced-model",
            hospital_profile_id="balanced_hospital"
        )

        self.scenario_configs['balanced'] = config
        return config


def setup_pipeline_for_retraining(config: PipelineConfig) -> TimeSeriesClusteringPipeline:
    """
    Set up and train a pipeline for retraining demonstration.

    Args:
        config: Pipeline configuration

    Returns:
        Trained pipeline ready for retraining scenarios
    """
    logger.info("Setting up pipeline for retraining: %s", config.experiment_name)

    # Initialize pipeline
    pipeline = TimeSeriesClusteringPipeline(config)

    # Generate initial training data
    np.random.seed(42)
    n_samples = 2500
    n_features = 12

    # Create structured training data
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples)

    # Add cluster structure
    cluster_centers = np.random.randn(4, n_features) * 2
    cluster_assignments = np.random.choice(4, n_samples)

    for i, center in enumerate(cluster_centers):
        mask = cluster_assignments == i
        X_train[mask] += center + np.random.randn(np.sum(mask), n_features) * 0.3
        y_train[mask] += i * 2 + np.random.randn(np.sum(mask)) * 0.5

    # Perform initial training
    clustering_result = pipeline.fit_initial(X_train, y_train)

    logger.info("Pipeline trained: %d clusters, %.3f noise ratio",
                clustering_result.metrics['n_clusters'], clustering_result.noise_ratio)

    return pipeline


def simulate_drift_batches(n_batches: int = 10,
                          batch_size: int = 200,
                          drift_intensity: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate batches with increasing data drift to trigger retraining.

    Args:
        n_batches: Number of batches to generate
        batch_size: Size of each batch
        drift_intensity: Intensity of drift (0.0 to 1.0)

    Returns:
        List of (X, y) batch tuples
    """
    logger.info("Generating %d drift batches with intensity %.2f", n_batches, drift_intensity)

    batches = []
    np.random.seed(123)  # Different seed for drift data

    for batch_idx in range(n_batches):
        # Increase drift over time
        current_drift = drift_intensity * (batch_idx / n_batches)

        # Generate base data
        X_batch = np.random.randn(batch_size, 12)
        y_batch = np.random.randn(batch_size)

        # Apply drift transformation
        drift_matrix = np.random.randn(12, 12) * current_drift
        X_batch = X_batch @ (np.eye(12) + drift_matrix)
        y_batch = y_batch + np.sum(X_batch[:, :3], axis=1) * current_drift

        # Add increasing noise
        noise_ratio = 0.1 + current_drift * 0.3
        n_noise = int(batch_size * noise_ratio)

        if n_noise > 0:
            noise_indices = np.random.choice(batch_size, n_noise, replace=False)
            X_batch[noise_indices] = np.random.uniform(-3, 3, (n_noise, 12))
            y_batch[noise_indices] = np.random.uniform(-10, 10, n_noise)

        batches.append((X_batch, y_batch))

    return batches


def demonstrate_automatic_retraining(pipeline: TimeSeriesClusteringPipeline,
                                   drift_batches: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """
    Demonstrate automatic retraining triggered by noise thresholds.

    Args:
        pipeline: Pipeline to test
        drift_batches: Batches with increasing drift

    Returns:
        Results of automatic retraining demonstration
    """
    logger.info("Demonstrating automatic retraining...")

    results = {
        'batch_results': [],
        'retraining_events': [],
        'performance_history': []
    }

    # Set up retraining callbacks
    def retraining_callback():
        logger.info("Automatic retraining triggered!")
        results['retraining_events'].append({
            'timestamp': datetime.now(),
            'trigger': 'automatic',
            'batch_number': len(results['batch_results'])
        })

    # Process batches and monitor for retraining
    for i, (X_batch, y_batch) in enumerate(drift_batches):
        logger.info("Processing drift batch %d/%d", i + 1, len(drift_batches))

        try:
            # Process batch
            batch_result = pipeline.process_batch(X_batch, y_batch)
            results['batch_results'].append(batch_result)

            logger.info("  Batch %d: noise_ratio=%.3f, processing_time=%.3f",
                       i + 1, batch_result.noise_ratio, batch_result.processing_time)

            # Check if retraining was triggered
            if batch_result.noise_ratio > pipeline.config.noise_threshold:
                logger.warning("  Noise threshold exceeded: %.3f > %.3f",
                              batch_result.noise_ratio, pipeline.config.noise_threshold)

                # Trigger retraining
                pipeline.retrain(f"Automatic retraining - batch {i + 1}")
                retraining_callback()

            # Record performance metrics
            performance_report = pipeline.get_performance_report()
            results['performance_history'].append({
                'batch_number': i + 1,
                'noise_ratio': batch_result.noise_ratio,
                'model_version': performance_report.model_version,
                'timestamp': datetime.now()
            })

        except Exception as e:
            logger.error("Batch %d processing failed: %s", i + 1, str(e))

    logger.info("Automatic retraining demonstration completed: %d retraining events",
                len(results['retraining_events']))

    return results


def demonstrate_manual_retraining(pipeline: TimeSeriesClusteringPipeline) -> Dict[str, Any]:
    """
    Demonstrate manual retraining with custom configurations.

    Args:
        pipeline: Pipeline to retrain

    Returns:
        Results of manual retraining
    """
    logger.info("Demonstrating manual retraining...")

    # Get initial performance baseline
    initial_report = pipeline.get_performance_report()
    initial_version = initial_report.model_version

    logger.info("Initial model version: %s", initial_version)

    # Trigger manual retraining
    logger.info("Triggering manual retraining...")
    start_time = datetime.now()

    try:
        pipeline.retrain("Manual retraining demonstration")
        retraining_time = (datetime.now() - start_time).total_seconds()

        # Get post-retraining performance
        updated_report = pipeline.get_performance_report()
        updated_version = updated_report.model_version

        logger.info("Manual retraining completed in %.2f seconds", retraining_time)
        logger.info("Model version updated: %s -> %s", initial_version, updated_version)

        # Compare performance
        performance_comparison = {
            'retraining_time': retraining_time,
            'version_change': {
                'before': initial_version,
                'after': updated_version
            },
            'clustering_comparison': {
                'before': initial_report.clustering_metrics,
                'after': updated_report.clustering_metrics
            }
        }

        # Log comparison results
        logger.info("Performance comparison:")
        if 'n_clusters' in initial_report.clustering_metrics and 'n_clusters' in updated_report.clustering_metrics:
            logger.info("  Clusters: %d -> %d",
                       initial_report.clustering_metrics['n_clusters'],
                       updated_report.clustering_metrics['n_clusters'])

        return performance_comparison

    except Exception as e:
        logger.error("Manual retraining failed: %s", str(e))
        raise


def demonstrate_scheduled_retraining(pipeline: TimeSeriesClusteringPipeline,
                                   schedule_interval: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
    """
    Demonstrate scheduled retraining workflow.

    Args:
        pipeline: Pipeline to schedule
        schedule_interval: Interval between scheduled retrainings

    Returns:
        Results of scheduled retraining demonstration
    """
    logger.info("Demonstrating scheduled retraining (simulated)...")

    # Simulate scheduled retraining check
    last_retraining = datetime.now() - timedelta(hours=25)  # Simulate overdue
    current_time = datetime.now()

    time_since_last = current_time - last_retraining

    logger.info("Time since last retraining: %s", time_since_last)
    logger.info("Schedule interval: %s", schedule_interval)

    results = {
        'schedule_check': {
            'last_retraining': last_retraining,
            'current_time': current_time,
            'time_since_last': time_since_last,
            'schedule_interval': schedule_interval,
            'retraining_due': time_since_last > schedule_interval
        }
    }

    if results['schedule_check']['retraining_due']:
        logger.info("Scheduled retraining is due - triggering retraining...")

        try:
            pipeline.retrain("Scheduled retraining")
            results['retraining_executed'] = True
            results['retraining_timestamp'] = datetime.now()

            logger.info("Scheduled retraining completed successfully")

        except Exception as e:
            logger.error("Scheduled retraining failed: %s", str(e))
            results['retraining_executed'] = False
            results['error'] = str(e)
    else:
        logger.info("Scheduled retraining not due yet")
        results['retraining_executed'] = False

    return results


def compare_retraining_strategies(scenario_manager: RetrainingScenarioManager) -> Dict[str, Any]:
    """
    Compare different retraining strategies across scenarios.

    Args:
        scenario_manager: Manager with different scenario configurations

    Returns:
        Comparison results across strategies
    """
    logger.info("Comparing retraining strategies...")

    strategies = ['conservative', 'aggressive', 'balanced']
    comparison_results = {}

    for strategy in strategies:
        logger.info("Testing %s strategy...", strategy)

        # Get configuration for this strategy
        if strategy == 'conservative':
            config = scenario_manager.create_conservative_config()
        elif strategy == 'aggressive':
            config = scenario_manager.create_aggressive_config()
        else:
            config = scenario_manager.create_balanced_config()

        # Set up pipeline
        pipeline = setup_pipeline_for_retraining(config)

        # Generate test batches
        test_batches = simulate_drift_batches(n_batches=5, drift_intensity=0.3)

        # Process batches and measure performance
        strategy_results = {
            'config': config.to_dict(),
            'batch_results': [],
            'retraining_triggered': False,
            'average_noise_ratio': 0.0,
            'average_processing_time': 0.0
        }

        noise_ratios = []
        processing_times = []

        for X_batch, y_batch in test_batches:
            try:
                batch_result = pipeline.process_batch(X_batch, y_batch)
                strategy_results['batch_results'].append({
                    'noise_ratio': batch_result.noise_ratio,
                    'processing_time': batch_result.processing_time
                })

                noise_ratios.append(batch_result.noise_ratio)
                processing_times.append(batch_result.processing_time)

                # Check if retraining would be triggered
                if batch_result.noise_ratio > config.noise_threshold:
                    strategy_results['retraining_triggered'] = True

            except Exception as e:
                logger.warning("Batch processing failed for %s strategy: %s", strategy, str(e))

        # Calculate averages
        if noise_ratios:
            strategy_results['average_noise_ratio'] = float(np.mean(noise_ratios))
            strategy_results['average_processing_time'] = float(np.mean(processing_times))

        comparison_results[strategy] = strategy_results

        logger.info("%s strategy results:", strategy.capitalize())
        logger.info("  Average noise ratio: %.3f", strategy_results['average_noise_ratio'])
        logger.info("  Average processing time: %.3f", strategy_results['average_processing_time'])
        logger.info("  Retraining triggered: %s", strategy_results['retraining_triggered'])

    # Generate comparison summary
    summary = {
        'strategy_comparison': comparison_results,
        'recommendations': []
    }

    # Analyze results and generate recommendations
    noise_ratios = {s: r['average_noise_ratio'] for s, r in comparison_results.items()}
    processing_times = {s: r['average_processing_time'] for s, r in comparison_results.items()}

    best_noise = min(noise_ratios, key=noise_ratios.get)
    fastest = min(processing_times, key=processing_times.get)

    summary['recommendations'].append(f"Best noise handling: {best_noise} strategy")
    summary['recommendations'].append(f"Fastest processing: {fastest} strategy")

    logger.info("Strategy comparison completed with %d recommendations",
                len(summary['recommendations']))

    return summary


def main():
    """Main function demonstrating retraining configuration scenarios."""

    print("=" * 80)
    print("MLflow Time Series Clustering Pipeline - Retraining Configuration Example")
    print("=" * 80)

    try:
        # Step 1: Set up scenario manager
        print("\n1. Setting up retraining scenario manager...")
        base_config = PipelineConfig(experiment_name="Retraining-Demo")
        scenario_manager = RetrainingScenarioManager(base_config)
        print("   Scenario manager initialized")

        # Step 2: Create and test balanced configuration
        print("\n2. Setting up balanced retraining configuration...")
        balanced_config = scenario_manager.create_balanced_config()
        balanced_pipeline = setup_pipeline_for_retraining(balanced_config)
        print(f"   Balanced pipeline ready: {balanced_pipeline.current_model_version}")

        # Step 3: Demonstrate automatic retraining
        print("\n3. Demonstrating automatic retraining...")
        drift_batches = simulate_drift_batches(n_batches=8, drift_intensity=0.6)
        auto_results = demonstrate_automatic_retraining(balanced_pipeline, drift_batches)
        print(f"   Processed {len(auto_results['batch_results'])} batches")
        print(f"   Retraining events: {len(auto_results['retraining_events'])}")

        # Step 4: Demonstrate manual retraining
        print("\n4. Demonstrating manual retraining...")
        manual_results = demonstrate_manual_retraining(balanced_pipeline)
        print(f"   Manual retraining completed in {manual_results['retraining_time']:.2f}s")
        print(f"   Version: {manual_results['version_change']['before']} -> {manual_results['version_change']['after']}")

        # Step 5: Demonstrate scheduled retraining
        print("\n5. Demonstrating scheduled retraining...")
        scheduled_results = demonstrate_scheduled_retraining(
            balanced_pipeline,
            schedule_interval=timedelta(hours=1)  # Short interval for demo
        )
        print(f"   Scheduled retraining due: {scheduled_results['schedule_check']['retraining_due']}")
        print(f"   Retraining executed: {scheduled_results.get('retraining_executed', False)}")

        # Step 6: Compare retraining strategies
        print("\n6. Comparing retraining strategies...")
        strategy_comparison = compare_retraining_strategies(scenario_manager)

        print("   Strategy comparison results:")
        for strategy, results in strategy_comparison['strategy_comparison'].items():
            print(f"     {strategy.capitalize()}:")
            print(f"       - Avg noise ratio: {results['average_noise_ratio']:.3f}")
            print(f"       - Avg processing time: {results['average_processing_time']:.3f}s")
            print(f"       - Retraining triggered: {results['retraining_triggered']}")

        print("\n   Recommendations:")
        for i, rec in enumerate(strategy_comparison['recommendations'], 1):
            print(f"     {i}. {rec}")

        # Step 7: Final summary and best practices
        print("\n7. Retraining configuration best practices:")
        print("   ✓ Conservative strategy: Higher thresholds, more stable models")
        print("   ✓ Aggressive strategy: Lower thresholds, faster adaptation")
        print("   ✓ Balanced strategy: Moderate thresholds, good trade-offs")
        print("   ✓ Monitor noise ratios and processing times continuously")
        print("   ✓ Set up both automatic and scheduled retraining")
        print("   ✓ Compare model versions after retraining")

        print("\n" + "=" * 80)
        print("Retraining configuration example completed successfully!")
        print("Key takeaways:")
        print(f"  - Tested {len(strategy_comparison['strategy_comparison'])} retraining strategies")
        print(f"  - Automatic retraining triggered {len(auto_results['retraining_events'])} times")
        print(f"  - Manual retraining completed in {manual_results['retraining_time']:.2f} seconds")
        print("  - Choose strategy based on your stability vs. adaptability requirements")
        print("=" * 80)

    except Exception as e:
        logger.error("Retraining configuration example failed: %s", str(e))
        print(f"\nRetraining example failed: {str(e)}")
        print("Check logs for detailed error information.")
        raise


if __name__ == "__main__":
    main()