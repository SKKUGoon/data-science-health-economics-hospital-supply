"""
Expanded window retraining for the MLflow Time Series Clustering Pipeline.

This module implements the ExpandedWindowRetrainer class that handles
HDBSCAN refitting with expanded historical data, cluster reassignment,
and kNN fallback model updates during retraining events.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass

from ..core.config import PipelineConfig
from ..core.data_models import ClusteringResult
from ..core.exceptions import ModelTrainingError, DataValidationError
from ..clustering.adaptive_clustering_engine import AdaptiveClusteringEngine
from ..models.cluster_specific_model_manager import ClusterSpecificModelManager
from .retraining_trigger import RetrainingEvent


logger = logging.getLogger(__name__)


@dataclass
class RetrainingResult:
    """
    Result of a retraining operation.

    Attributes:
        clustering_result: New clustering result after retraining
        model_comparison: Comparison metrics between old and new models
        reassignment_summary: Summary of cluster reassignments
        retraining_metrics: Performance metrics from retraining process
        timestamp: When retraining was completed
        trigger_event: The event that triggered this retraining
    """
    clustering_result: ClusteringResult
    model_comparison: Dict[str, Any]
    reassignment_summary: Dict[str, Any]
    retraining_metrics: Dict[str, Any]
    timestamp: datetime
    trigger_event: RetrainingEvent


class ExpandedWindowRetrainer:
    """
    Handles expanded window retraining for the clustering pipeline.

    This class manages the retraining process including HDBSCAN refitting
    with expanded historical data, cluster reassignment for all data points,
    and updating the kNN fallback model with new cluster assignments.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the expanded window retrainer.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.expanding_window_size = config.expanding_window_size

        # Data storage for expanded window
        self._historical_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self._max_history_size = max(config.expanding_window_size * 2, 20000)  # Keep extra for safety

        # Current models (to be updated during retraining)
        self._current_clustering_engine: Optional[AdaptiveClusteringEngine] = None
        self._current_model_manager: Optional[ClusterSpecificModelManager] = None

        # Retraining history
        self._retraining_results: List[RetrainingResult] = []

        logger.info("ExpandedWindowRetrainer initialized with window size: %d",
                   self.expanding_window_size)

    def set_current_models(self,
                          clustering_engine: AdaptiveClusteringEngine,
                          model_manager: ClusterSpecificModelManager) -> None:
        """
        Set the current models that will be updated during retraining.

        Args:
            clustering_engine: Current clustering engine
            model_manager: Current model manager
        """
        self._current_clustering_engine = clustering_engine
        self._current_model_manager = model_manager
        logger.info("Current models set for retraining")

    def add_historical_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Add new data to the historical data buffer.

        Args:
            X: Input features
            y: Target values
        """
        # Validate input data
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise DataValidationError("Historical data must be numpy arrays")

        if X.shape[0] != y.shape[0]:
            raise DataValidationError("X and y must have the same number of samples")

        # Add to historical data
        self._historical_data.append((X.copy(), y.copy()))

        # Maintain maximum history size
        if len(self._historical_data) > self._max_history_size // 1000:  # Approximate batch count
            # Remove oldest batches to stay within memory limits
            excess = len(self._historical_data) - (self._max_history_size // 1000)
            self._historical_data = self._historical_data[excess:]
            logger.debug("Trimmed historical data to maintain memory limits")

        logger.debug("Added historical data batch: X shape %s, y shape %s", X.shape, y.shape)

    def perform_retraining(self, trigger_event: RetrainingEvent) -> RetrainingResult:
        """
        Perform expanded window retraining.

        Args:
            trigger_event: The event that triggered this retraining

        Returns:
            RetrainingResult containing all retraining information

        Raises:
            ModelTrainingError: If retraining fails
        """
        logger.info("Starting expanded window retraining triggered by: %s",
                   trigger_event.reason.value)

        if self._current_clustering_engine is None:
            raise ModelTrainingError(
                "No current clustering engine set for retraining",
                model_type="clustering_engine"
            )

        start_time = datetime.now()

        try:
            # Step 1: Prepare expanded window data
            X_expanded, y_expanded = self._prepare_expanded_window_data()
            logger.info("Prepared expanded window data: %d samples", X_expanded.shape[0])

            # Step 2: Store old model metrics for comparison
            old_metrics = self._capture_old_model_metrics()

            # Step 3: Refit HDBSCAN with expanded data
            new_clustering_result = self._refit_hdbscan(X_expanded)
            logger.info("HDBSCAN refitting completed with %d clusters",
                       new_clustering_result.metrics['n_clusters'])

            # Step 4: Analyze cluster reassignments
            reassignment_summary = self._analyze_cluster_reassignments(
                X_expanded, new_clustering_result.labels
            )

            # Step 5: Update kNN fallback model
            self._update_knn_fallback(X_expanded, new_clustering_result.labels)

            # Step 6: Retrain cluster-specific models if model manager is available
            if self._current_model_manager is not None:
                self._retrain_cluster_models(X_expanded, y_expanded, new_clustering_result.labels)

            # Step 7: Calculate model comparison metrics
            new_metrics = self._capture_new_model_metrics()
            model_comparison = self._compare_model_performance(old_metrics, new_metrics)

            # Step 8: Calculate retraining performance metrics
            end_time = datetime.now()
            retraining_metrics = {
                'retraining_duration_seconds': (end_time - start_time).total_seconds(),
                'expanded_window_size': X_expanded.shape[0],
                'old_cluster_count': old_metrics.get('n_clusters', 0),
                'new_cluster_count': new_clustering_result.metrics['n_clusters'],
                'old_noise_ratio': old_metrics.get('noise_ratio', 0.0),
                'new_noise_ratio': new_clustering_result.noise_ratio,
                'cluster_stability_score': reassignment_summary.get('stability_score', 0.0)
            }

            # Create retraining result
            result = RetrainingResult(
                clustering_result=new_clustering_result,
                model_comparison=model_comparison,
                reassignment_summary=reassignment_summary,
                retraining_metrics=retraining_metrics,
                timestamp=end_time,
                trigger_event=trigger_event
            )

            # Store result in history
            self._retraining_results.append(result)

            logger.info("Expanded window retraining completed successfully in %.2f seconds",
                       retraining_metrics['retraining_duration_seconds'])

            return result

        except Exception as e:
            logger.error("Expanded window retraining failed: %s", str(e))
            raise ModelTrainingError(
                f"Expanded window retraining failed: {str(e)}",
                model_type="expanded_window_retraining",
                training_data_size=len(self._historical_data)
            ) from e

    def _prepare_expanded_window_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare expanded window data from historical data buffer.

        Returns:
            Tuple of (X_expanded, y_expanded) arrays

        Raises:
            DataValidationError: If insufficient historical data
        """
        if not self._historical_data:
            raise DataValidationError("No historical data available for retraining")

        # Concatenate all historical data
        X_batches = []
        y_batches = []
        total_samples = 0

        # Use data from most recent batches up to expanding window size
        for X_batch, y_batch in reversed(self._historical_data):
            if total_samples + X_batch.shape[0] > self.expanding_window_size:
                # Take only part of this batch to reach exact window size
                remaining = self.expanding_window_size - total_samples
                if remaining > 0:
                    X_batches.insert(0, X_batch[:remaining])
                    y_batches.insert(0, y_batch[:remaining])
                break
            else:
                X_batches.insert(0, X_batch)
                y_batches.insert(0, y_batch)
                total_samples += X_batch.shape[0]

        if not X_batches:
            raise DataValidationError("No valid historical data batches found")

        # Concatenate all batches
        X_expanded = np.vstack(X_batches)
        y_expanded = np.hstack(y_batches)

        logger.info("Prepared expanded window: %d samples from %d batches",
                   X_expanded.shape[0], len(X_batches))

        return X_expanded, y_expanded

    def _capture_old_model_metrics(self) -> Dict[str, Any]:
        """
        Capture metrics from the current model before retraining.

        Returns:
            Dictionary containing old model metrics
        """
        old_metrics = {}

        if self._current_clustering_engine and self._current_clustering_engine.is_fitted:
            try:
                clustering_metrics = self._current_clustering_engine.get_clustering_metrics()
                old_metrics.update(clustering_metrics)

                # Get kNN fallback metrics
                knn_metrics = self._current_clustering_engine.get_knn_fallback_metrics()
                if knn_metrics:
                    old_metrics['knn_fallback'] = knn_metrics

            except Exception as e:
                logger.warning("Failed to capture old clustering metrics: %s", str(e))

        if self._current_model_manager:
            try:
                model_metrics = self._current_model_manager.get_model_metrics()
                old_metrics['model_manager'] = model_metrics
            except Exception as e:
                logger.warning("Failed to capture old model manager metrics: %s", str(e))

        return old_metrics

    def _refit_hdbscan(self, X_expanded: np.ndarray) -> ClusteringResult:
        """
        Refit HDBSCAN clustering with expanded data.

        Args:
            X_expanded: Expanded window data

        Returns:
            New clustering result
        """
        # Create new clustering engine with same parameters
        new_engine = AdaptiveClusteringEngine(
            hdbscan_params=self.config.hdbscan_params,
            knn_params=self.config.knn_params
        )

        # Fit on expanded data
        clustering_result = new_engine.fit(X_expanded)

        # Replace current clustering engine
        self._current_clustering_engine = new_engine

        return clustering_result

    def _analyze_cluster_reassignments(self,
                                     X_expanded: np.ndarray,
                                     new_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze how cluster assignments have changed after retraining.

        Args:
            X_expanded: Expanded window data
            new_labels: New cluster labels

        Returns:
            Dictionary containing reassignment analysis
        """
        unique_labels = np.unique(new_labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(new_labels == -1)

        # Calculate cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = int(np.sum(new_labels == label))

        # Calculate stability metrics
        stability_score = self._calculate_cluster_stability(new_labels)

        reassignment_summary = {
            'total_points_reassigned': X_expanded.shape[0],
            'new_cluster_count': n_clusters,
            'new_noise_points': n_noise,
            'new_noise_ratio': n_noise / len(new_labels),
            'cluster_sizes': cluster_sizes,
            'stability_score': stability_score,
            'largest_cluster_size': max(cluster_sizes.values()) if cluster_sizes else 0,
            'smallest_cluster_size': min([size for label, size in cluster_sizes.items()
                                        if label != -1]) if any(label != -1 for label in cluster_sizes) else 0
        }

        logger.info("Cluster reassignment analysis: %d clusters, %.2f%% noise, stability: %.3f",
                   n_clusters, reassignment_summary['new_noise_ratio'] * 100, stability_score)

        return reassignment_summary

    def _calculate_cluster_stability(self, labels: np.ndarray) -> float:
        """
        Calculate a stability score for the clustering result.

        Args:
            labels: Cluster labels

        Returns:
            Stability score between 0 and 1
        """
        try:
            # Simple stability metric based on cluster size distribution
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) <= 1:
                return 0.0

            cluster_sizes = [np.sum(labels == label) for label in unique_labels]

            # Calculate coefficient of variation (lower is more stable)
            mean_size = np.mean(cluster_sizes)
            std_size = np.std(cluster_sizes)

            if mean_size == 0:
                return 0.0

            cv = std_size / mean_size
            # Convert to stability score (higher is better)
            stability = 1.0 / (1.0 + cv)

            return float(stability)

        except Exception as e:
            logger.warning("Failed to calculate cluster stability: %s", str(e))
            return 0.0

    def _update_knn_fallback(self, X_expanded: np.ndarray, new_labels: np.ndarray) -> None:
        """
        Update kNN fallback model with new cluster assignments.

        Args:
            X_expanded: Expanded window data
            new_labels: New cluster labels
        """
        try:
            if self._current_clustering_engine:
                self._current_clustering_engine.update_fallback_model(X_expanded, new_labels)
                logger.info("kNN fallback model updated with new cluster assignments")
        except Exception as e:
            logger.error("Failed to update kNN fallback model: %s", str(e))

    def _retrain_cluster_models(self,
                               X_expanded: np.ndarray,
                               y_expanded: np.ndarray,
                               new_labels: np.ndarray) -> None:
        """
        Retrain cluster-specific models with new cluster assignments.

        Args:
            X_expanded: Expanded window features
            y_expanded: Expanded window targets
            new_labels: New cluster labels
        """
        try:
            if self._current_model_manager:
                # Prepare cluster data
                cluster_data = {}
                unique_labels = np.unique(new_labels[new_labels != -1])

                for label in unique_labels:
                    mask = new_labels == label
                    cluster_data[int(label)] = {
                        'X': X_expanded[mask],
                        'y': y_expanded[mask]
                    }

                # Retrain time series models
                self._current_model_manager.fit_timeseries_models(cluster_data)

                # Prepare resource data for retraining
                resource_data = self._prepare_resource_data_for_retraining(cluster_data)

                # Retrain resource models
                self._current_model_manager.fit_resource_models(cluster_data, resource_data)

                logger.info("Cluster-specific models retrained for %d clusters", len(unique_labels))

        except Exception as e:
            logger.error("Failed to retrain cluster-specific models: %s", str(e))

    def _prepare_resource_data_for_retraining(self, cluster_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Prepare synthetic resource usage data for retraining.

        Args:
            cluster_data: Dictionary mapping cluster_id to {'X': features, 'y': targets}

        Returns:
            Dictionary mapping cluster_id to resource usage metrics
        """
        resource_data = {}

        for cluster_id, data in cluster_data.items():
            n_samples = len(data['X'])

            # Generate synthetic resource usage metrics based on cluster size and complexity
            # Processing time: base time + complexity factor based on features
            base_processing_time = 0.01  # 10ms base
            complexity_factor = np.mean(np.std(data['X'], axis=0))  # Feature complexity
            processing_time = np.random.normal(
                base_processing_time + complexity_factor * 0.001,
                base_processing_time * 0.1,
                n_samples
            )
            processing_time = np.maximum(processing_time, 0.001)  # Minimum 1ms

            # Memory usage: base memory + data size factor
            base_memory = 50.0  # 50MB base
            memory_per_sample = 0.1  # 0.1MB per sample
            memory_usage = np.random.normal(
                base_memory + n_samples * memory_per_sample,
                base_memory * 0.05,
                n_samples
            )
            memory_usage = np.maximum(memory_usage, 10.0)  # Minimum 10MB

            # Computational complexity: based on feature dimensionality and variance
            feature_complexity = data['X'].shape[1] * np.mean(np.var(data['X'], axis=0))
            complexity = np.random.normal(
                feature_complexity,
                feature_complexity * 0.1,
                n_samples
            )
            complexity = np.maximum(complexity, 1.0)  # Minimum complexity of 1

            resource_data[cluster_id] = {
                'processing_time': processing_time,
                'memory_usage': memory_usage,
                'complexity': complexity
            }

        return resource_data

    def _capture_new_model_metrics(self) -> Dict[str, Any]:
        """
        Capture metrics from the newly trained models.

        Returns:
            Dictionary containing new model metrics
        """
        new_metrics = {}

        if self._current_clustering_engine and self._current_clustering_engine.is_fitted:
            try:
                clustering_metrics = self._current_clustering_engine.get_clustering_metrics()
                new_metrics.update(clustering_metrics)

                # Get kNN fallback metrics
                knn_metrics = self._current_clustering_engine.get_knn_fallback_metrics()
                if knn_metrics:
                    new_metrics['knn_fallback'] = knn_metrics

            except Exception as e:
                logger.warning("Failed to capture new clustering metrics: %s", str(e))

        if self._current_model_manager:
            try:
                model_metrics = self._current_model_manager.get_model_metrics()
                new_metrics['model_manager'] = model_metrics
            except Exception as e:
                logger.warning("Failed to capture new model manager metrics: %s", str(e))

        return new_metrics

    def _compare_model_performance(self,
                                  old_metrics: Dict[str, Any],
                                  new_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare performance between old and new models.

        Args:
            old_metrics: Metrics from old models
            new_metrics: Metrics from new models

        Returns:
            Dictionary containing performance comparison
        """
        comparison = {
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'improvements': {},
            'degradations': {}
        }

        # Compare clustering metrics
        clustering_comparisons = [
            ('n_clusters', 'cluster_count_change'),
            ('noise_ratio', 'noise_ratio_change'),
            ('silhouette_score', 'silhouette_improvement'),
            ('calinski_harabasz_score', 'calinski_harabasz_improvement'),
            ('davies_bouldin_score', 'davies_bouldin_improvement')
        ]

        for metric_name, comparison_name in clustering_comparisons:
            old_val = old_metrics.get(metric_name)
            new_val = new_metrics.get(metric_name)

            if old_val is not None and new_val is not None:
                if metric_name == 'davies_bouldin_score':
                    # Lower is better for Davies-Bouldin
                    change = old_val - new_val
                else:
                    # Higher is better for other metrics
                    change = new_val - old_val

                comparison[comparison_name] = change

                if change > 0:
                    comparison['improvements'][metric_name] = change
                elif change < 0:
                    comparison['degradations'][metric_name] = abs(change)

        # Overall improvement score
        improvement_count = len(comparison['improvements'])
        degradation_count = len(comparison['degradations'])
        total_comparisons = improvement_count + degradation_count

        if total_comparisons > 0:
            comparison['overall_improvement_ratio'] = improvement_count / total_comparisons
        else:
            comparison['overall_improvement_ratio'] = 0.5  # Neutral if no comparisons possible

        return comparison

    def get_retraining_history(self, limit: Optional[int] = None) -> List[RetrainingResult]:
        """
        Get history of retraining results.

        Args:
            limit: Optional limit on number of results to return

        Returns:
            List of retraining results
        """
        if limit is None:
            return self._retraining_results.copy()
        return self._retraining_results[-limit:].copy()

    def get_historical_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of historical data stored for retraining.

        Returns:
            Dictionary containing historical data summary
        """
        if not self._historical_data:
            return {
                'total_batches': 0,
                'total_samples': 0,
                'memory_usage_estimate': 0
            }

        total_samples = sum(X.shape[0] for X, _ in self._historical_data)
        total_features = self._historical_data[0][0].shape[1] if self._historical_data else 0

        # Rough memory estimate (assuming float64)
        memory_estimate = total_samples * total_features * 8 * 2  # X and y arrays

        return {
            'total_batches': len(self._historical_data),
            'total_samples': total_samples,
            'total_features': total_features,
            'memory_usage_estimate_bytes': memory_estimate,
            'expanding_window_size': self.expanding_window_size,
            'max_history_size': self._max_history_size
        }