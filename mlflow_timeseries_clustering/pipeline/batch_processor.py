"""
Batch processing engine for the MLflow Time Series Clustering Pipeline.

This module implements the BatchProcessor class that handles batch data processing
with scaling, cluster assignment, noise detection, and kNN fallback.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging

from ..core.data_models import BatchResult
from ..core.config import PipelineConfig
from ..core.exceptions import PredictionError, DataValidationError
from ..clustering.adaptive_clustering_engine import AdaptiveClusteringEngine
from ..models.cluster_specific_model_manager import ClusterSpecificModelManager
from ..mlflow_integration.logging_utils import PipelineLogger
from ..mlflow_integration.experiment_manager import ExperimentManager


logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processing engine for streaming data with cluster assignment and prediction.

    This class handles batch processing of new data using trained HDBSCAN clustering,
    kNN fallback for noise points, and cluster-specific time series models.
    """

    def __init__(self,
                 config: PipelineConfig,
                 clustering_engine: AdaptiveClusteringEngine,
                 model_manager: ClusterSpecificModelManager,
                 experiment_manager: ExperimentManager):
        """
        Initialize the batch processor.

        Args:
            config: Pipeline configuration
            clustering_engine: Fitted adaptive clustering engine
            model_manager: Fitted cluster-specific model manager
            experiment_manager: MLflow experiment manager
        """
        self.config = config
        self.clustering_engine = clustering_engine
        self.model_manager = model_manager
        self.experiment_manager = experiment_manager
        self.logger = PipelineLogger("BatchProcessor")

        # Batch processing state
        self.batch_count = 0
        self.total_processed_samples = 0
        self.processing_history: List[Dict[str, Any]] = []
        self.noise_ratio_history: List[float] = []

        # Performance tracking
        self.processing_times: List[float] = []
        self.prediction_accuracies: List[Dict[str, float]] = []

        # Drift detection
        self.drift_detected = False
        self.consecutive_high_noise_batches = 0
        self.max_consecutive_high_noise = 3

        logger.info("BatchProcessor initialized with batch_size=%d, noise_threshold=%.3f",
                   config.batch_size, config.noise_threshold)

    def process_batch(self, X_batch: np.ndarray, y_batch: Optional[np.ndarray] = None) -> BatchResult:
        """
        Process a single batch of data with cluster assignment and prediction.

        Args:
            X_batch: Input feature matrix of shape (n_samples, n_features)
            y_batch: Optional target values for accuracy tracking

        Returns:
            BatchResult containing cluster assignments, predictions, and metrics

        Raises:
            DataValidationError: If input data is invalid
            PredictionError: If batch processing fails
        """
        start_time = time.time()
        self.batch_count += 1

        self.logger.info("Processing batch %d with %d samples", self.batch_count, len(X_batch))

        try:
            # Validate input data
            self._validate_batch_data(X_batch, y_batch)

            # Step 1: Scale the data using fitted scaler
            X_scaled = self._scale_batch_data(X_batch)

            # Step 2: Assign cluster labels with fallback handling
            cluster_assignments, noise_ratio = self._assign_clusters(X_scaled)

            # Step 3: Make time series predictions
            timeseries_predictions, ts_metadata = self._make_timeseries_predictions(
                cluster_assignments, X_batch
            )

            # Step 4: Make resource usage predictions
            resource_predictions = self._make_resource_predictions(cluster_assignments, X_batch)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create batch result
            batch_result = BatchResult(
                cluster_assignments=cluster_assignments,
                timeseries_predictions=timeseries_predictions,
                resource_predictions=resource_predictions,
                noise_ratio=noise_ratio,
                processing_time=processing_time
            )

            # Update processing state
            self._update_processing_state(batch_result, ts_metadata)

            # Track accuracy if ground truth provided
            if y_batch is not None:
                self._track_batch_accuracy(y_batch, timeseries_predictions, cluster_assignments)

            # Check for drift detection
            self._check_drift_detection(noise_ratio)

            # Log batch metrics to MLflow
            self._log_batch_metrics(batch_result, ts_metadata)

            self.logger.info("Batch %d processed successfully in %.3f seconds. Noise ratio: %.3f",
                           self.batch_count, processing_time, noise_ratio)

            return batch_result

        except Exception as e:
            self.logger.error("Failed to process batch %d: %s", self.batch_count, str(e))
            if isinstance(e, (DataValidationError, PredictionError)):
                raise
            raise PredictionError(
                f"Batch processing failed: {str(e)}",
                prediction_type="batch_processing",
                batch_size=len(X_batch) if X_batch is not None else 0
            ) from e

    def _validate_batch_data(self, X_batch: np.ndarray, y_batch: Optional[np.ndarray] = None):
        """
        Validate batch input data.

        Args:
            X_batch: Input feature matrix
            y_batch: Optional target values

        Raises:
            DataValidationError: If validation fails
        """
        if X_batch is None:
            raise DataValidationError("Batch data cannot be None")

        if not isinstance(X_batch, np.ndarray):
            raise DataValidationError("Batch data must be a numpy array", data_type=type(X_batch).__name__)

        if X_batch.ndim != 2:
            raise DataValidationError(
                "Batch data must be 2-dimensional",
                expected_shape="(n_samples, n_features)",
                actual_shape=X_batch.shape
            )

        if X_batch.shape[0] == 0:
            raise DataValidationError("Batch cannot be empty", actual_shape=X_batch.shape)

        if X_batch.shape[1] == 0:
            raise DataValidationError("Batch must have at least one feature", actual_shape=X_batch.shape)

        # Check for invalid values
        if np.any(np.isnan(X_batch)):
            raise DataValidationError("Batch data contains NaN values")

        if np.any(np.isinf(X_batch)):
            raise DataValidationError("Batch data contains infinite values")

        # Validate y_batch if provided
        if y_batch is not None:
            if not isinstance(y_batch, np.ndarray):
                raise DataValidationError("Target data must be a numpy array", data_type=type(y_batch).__name__)

            if len(y_batch) != len(X_batch):
                raise DataValidationError(
                    f"Target data length ({len(y_batch)}) must match batch size ({len(X_batch)})"
                )

            if np.any(np.isnan(y_batch)) or np.any(np.isinf(y_batch)):
                raise DataValidationError("Target data contains NaN or infinite values")

    def _scale_batch_data(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Scale batch data using the fitted scaler from clustering engine.

        Args:
            X_batch: Input feature matrix

        Returns:
            Scaled feature matrix

        Raises:
            PredictionError: If scaling fails
        """
        try:
            if not self.clustering_engine.is_fitted:
                raise PredictionError(
                    "Clustering engine must be fitted before batch processing",
                    prediction_type="data_scaling"
                )

            X_scaled = self.clustering_engine.scaler.transform(X_batch)
            self.logger.debug("Batch data scaled successfully")
            return X_scaled

        except Exception as e:
            raise PredictionError(
                f"Failed to scale batch data: {str(e)}",
                prediction_type="data_scaling",
                batch_size=len(X_batch)
            ) from e

    def _assign_clusters(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Assign cluster labels using HDBSCAN with kNN fallback for noise points.

        Args:
            X_scaled: Scaled feature matrix

        Returns:
            Tuple of (cluster_assignments, noise_ratio)

        Raises:
            PredictionError: If cluster assignment fails
        """
        try:
            # Use clustering engine prediction with fallback
            cluster_assignments = self.clustering_engine.predict(X_scaled)

            # Calculate noise ratio
            noise_count = np.sum(cluster_assignments == -1)
            noise_ratio = noise_count / len(cluster_assignments)

            self.logger.debug("Cluster assignment completed. Noise ratio: %.3f", noise_ratio)

            return cluster_assignments, noise_ratio

        except Exception as e:
            raise PredictionError(
                f"Failed to assign clusters: {str(e)}",
                prediction_type="cluster_assignment",
                batch_size=len(X_scaled)
            ) from e

    def _make_timeseries_predictions(self, cluster_assignments: np.ndarray,
                                   X_batch: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make time series predictions using cluster-specific models.

        Args:
            cluster_assignments: Cluster assignments for each sample
            X_batch: Original feature matrix

        Returns:
            Tuple of (predictions, metadata)

        Raises:
            PredictionError: If prediction fails
        """
        try:
            predictions, metadata = self.model_manager.predict_timeseries(
                cluster_assignments, X_batch
            )

            self.logger.debug("Time series predictions completed for %d samples", len(predictions))

            return predictions, metadata

        except Exception as e:
            raise PredictionError(
                f"Failed to make time series predictions: {str(e)}",
                prediction_type="timeseries_prediction",
                batch_size=len(X_batch)
            ) from e

    def _make_resource_predictions(self, cluster_assignments: np.ndarray,
                                 X_batch: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Make resource usage predictions for each cluster.

        Args:
            cluster_assignments: Cluster assignments for each sample
            X_batch: Original feature matrix

        Returns:
            Dictionary mapping cluster_id to resource predictions

        Raises:
            PredictionError: If resource prediction fails
        """
        try:
            # Get unique clusters in the batch
            unique_clusters = np.unique(cluster_assignments)
            resource_predictions = {}

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue

                cluster_mask = cluster_assignments == cluster_id
                cluster_X = X_batch[cluster_mask]

                if len(cluster_X) == 0:
                    continue

                try:
                    # Make resource predictions for this cluster
                    cluster_resources = self.model_manager.predict_resources_single_cluster(cluster_id, cluster_X)
                    resource_predictions[cluster_id] = cluster_resources

                except Exception as e:
                    self.logger.warning("Failed to predict resources for cluster %d: %s",
                                      cluster_id, str(e))
                    # Continue with other clusters
                    continue

            self.logger.debug("Resource predictions completed for %d clusters", len(resource_predictions))

            return resource_predictions

        except Exception as e:
            raise PredictionError(
                f"Failed to make resource predictions: {str(e)}",
                prediction_type="resource_prediction",
                batch_size=len(X_batch)
            ) from e

    def _update_processing_state(self, batch_result: BatchResult, ts_metadata: Dict[str, Any]):
        """
        Update internal processing state with batch results.

        Args:
            batch_result: Results from batch processing
            ts_metadata: Time series prediction metadata
        """
        # Update counters
        self.total_processed_samples += len(batch_result.cluster_assignments)
        self.processing_times.append(batch_result.processing_time)
        self.noise_ratio_history.append(batch_result.noise_ratio)

        # Store processing history
        processing_record = {
            'batch_id': self.batch_count,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(batch_result.cluster_assignments),
            'noise_ratio': batch_result.noise_ratio,
            'processing_time': batch_result.processing_time,
            'clusters_predicted': len(ts_metadata.get('cluster_predictions', {})),
            'successful_predictions': ts_metadata.get('successful_predictions', 0),
            'failed_predictions': len(ts_metadata.get('failed_predictions', [])),
            'noise_point_predictions': ts_metadata.get('noise_point_predictions', 0)
        }

        self.processing_history.append(processing_record)

        # Keep only recent history to prevent memory issues
        max_history_size = 1000
        if len(self.processing_history) > max_history_size:
            self.processing_history = self.processing_history[-max_history_size:]
            self.processing_times = self.processing_times[-max_history_size:]
            self.noise_ratio_history = self.noise_ratio_history[-max_history_size:]

    def _track_batch_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                            cluster_assignments: np.ndarray):
        """
        Track prediction accuracy for the current batch.

        Args:
            y_true: True target values
            y_pred: Predicted values
            cluster_assignments: Cluster assignments
        """
        try:
            # Use model manager's accuracy tracking
            self.model_manager.track_prediction_accuracy(
                y_true, y_pred, cluster_assignments, step=self.batch_count
            )

            # Store accuracy metrics locally
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            # Calculate overall accuracy metrics
            valid_mask = ~np.isnan(y_pred)
            if np.sum(valid_mask) > 0:
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]

                accuracy_metrics = {
                    'batch_id': self.batch_count,
                    'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
                    'mae': mean_absolute_error(y_true_valid, y_pred_valid),
                    'r2_score': r2_score(y_true_valid, y_pred_valid),
                    'valid_predictions': len(y_true_valid),
                    'total_predictions': len(y_true)
                }

                self.prediction_accuracies.append(accuracy_metrics)

                self.logger.debug("Batch accuracy tracked: RMSE=%.4f, MAE=%.4f, R²=%.4f",
                                accuracy_metrics['rmse'], accuracy_metrics['mae'],
                                accuracy_metrics['r2_score'])

        except Exception as e:
            self.logger.error("Failed to track batch accuracy: %s", str(e))

    def _check_drift_detection(self, noise_ratio: float):
        """
        Check for data drift based on noise ratio threshold.

        Args:
            noise_ratio: Current batch noise ratio
        """
        if noise_ratio > self.config.noise_threshold:
            self.consecutive_high_noise_batches += 1
            self.logger.warning("High noise ratio detected: %.3f > %.3f (batch %d consecutive)",
                              noise_ratio, self.config.noise_threshold,
                              self.consecutive_high_noise_batches)

            if self.consecutive_high_noise_batches >= self.max_consecutive_high_noise:
                self.drift_detected = True
                self.logger.warning("Data drift detected after %d consecutive high-noise batches",
                                  self.consecutive_high_noise_batches)
        else:
            self.consecutive_high_noise_batches = 0
            if self.drift_detected:
                self.logger.info("Noise ratio returned to normal: %.3f", noise_ratio)

    def _log_batch_metrics(self, batch_result: BatchResult, ts_metadata: Dict[str, Any]):
        """
        Log batch processing metrics to MLflow.

        Args:
            batch_result: Batch processing results
            ts_metadata: Time series prediction metadata
        """
        try:
            # Prepare batch metrics
            batch_metrics = {
                'batch_id': self.batch_count,
                'sample_count': len(batch_result.cluster_assignments),
                'noise_ratio': batch_result.noise_ratio,
                'processing_time': batch_result.processing_time,
                'samples_per_second': len(batch_result.cluster_assignments) / batch_result.processing_time,
                'clusters_predicted': len(ts_metadata.get('cluster_predictions', {})),
                'successful_predictions': ts_metadata.get('successful_predictions', 0),
                'failed_predictions': len(ts_metadata.get('failed_predictions', [])),
                'noise_point_predictions': ts_metadata.get('noise_point_predictions', 0),
                'drift_detected': self.drift_detected,
                'consecutive_high_noise_batches': self.consecutive_high_noise_batches
            }

            # Add resource prediction metrics
            resource_count = len(batch_result.resource_predictions)
            batch_metrics['resource_predictions_count'] = resource_count

            # Add cumulative metrics
            batch_metrics.update({
                'total_processed_samples': self.total_processed_samples,
                'avg_processing_time': np.mean(self.processing_times),
                'avg_noise_ratio': np.mean(self.noise_ratio_history),
                'total_batches_processed': self.batch_count
            })

            # Log metrics with step
            self.logger.log_metrics(batch_metrics, step=self.batch_count, prefix="batch_processing")

            # Log cluster-specific metrics
            for cluster_id, cluster_stats in ts_metadata.get('cluster_predictions', {}).items():
                cluster_metrics = {
                    f'cluster_{cluster_id}_sample_count': cluster_stats['sample_count'],
                    f'cluster_{cluster_id}_mean_prediction': cluster_stats['mean_prediction'],
                    f'cluster_{cluster_id}_std_prediction': cluster_stats['std_prediction']
                }
                self.logger.log_metrics(cluster_metrics, step=self.batch_count, prefix="cluster_batch")

        except Exception as e:
            self.logger.error("Failed to log batch metrics: %s", str(e))

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive processing summary statistics.

        Returns:
            Dictionary containing processing summary
        """
        if not self.processing_history:
            return {'status': 'no_batches_processed'}

        try:
            # Calculate summary statistics
            processing_times = [record['processing_time'] for record in self.processing_history]
            noise_ratios = [record['noise_ratio'] for record in self.processing_history]
            sample_counts = [record['sample_count'] for record in self.processing_history]

            summary = {
                'total_batches_processed': self.batch_count,
                'total_samples_processed': self.total_processed_samples,
                'processing_time_stats': {
                    'mean': np.mean(processing_times),
                    'std': np.std(processing_times),
                    'min': np.min(processing_times),
                    'max': np.max(processing_times),
                    'median': np.median(processing_times)
                },
                'noise_ratio_stats': {
                    'mean': np.mean(noise_ratios),
                    'std': np.std(noise_ratios),
                    'min': np.min(noise_ratios),
                    'max': np.max(noise_ratios),
                    'median': np.median(noise_ratios)
                },
                'batch_size_stats': {
                    'mean': np.mean(sample_counts),
                    'std': np.std(sample_counts),
                    'min': np.min(sample_counts),
                    'max': np.max(sample_counts),
                    'median': np.median(sample_counts)
                },
                'throughput_stats': {
                    'avg_samples_per_second': np.mean([
                        record['sample_count'] / record['processing_time']
                        for record in self.processing_history
                    ]),
                    'total_processing_time': np.sum(processing_times)
                },
                'drift_detection': {
                    'drift_currently_detected': self.drift_detected,
                    'consecutive_high_noise_batches': self.consecutive_high_noise_batches,
                    'high_noise_batch_count': sum(1 for nr in noise_ratios if nr > self.config.noise_threshold),
                    'high_noise_percentage': sum(1 for nr in noise_ratios if nr > self.config.noise_threshold) / len(noise_ratios) * 100
                }
            }

            # Add accuracy summary if available
            if self.prediction_accuracies:
                rmse_values = [acc['rmse'] for acc in self.prediction_accuracies]
                mae_values = [acc['mae'] for acc in self.prediction_accuracies]
                r2_values = [acc['r2_score'] for acc in self.prediction_accuracies]

                summary['accuracy_stats'] = {
                    'rmse_stats': {
                        'mean': np.mean(rmse_values),
                        'std': np.std(rmse_values),
                        'min': np.min(rmse_values),
                        'max': np.max(rmse_values)
                    },
                    'mae_stats': {
                        'mean': np.mean(mae_values),
                        'std': np.std(mae_values),
                        'min': np.min(mae_values),
                        'max': np.max(mae_values)
                    },
                    'r2_stats': {
                        'mean': np.mean(r2_values),
                        'std': np.std(r2_values),
                        'min': np.min(r2_values),
                        'max': np.max(r2_values)
                    }
                }

            return summary

        except Exception as e:
            self.logger.error("Failed to generate processing summary: %s", str(e))
            return {'error': str(e), 'total_batches_processed': self.batch_count}

    def reset_drift_detection(self):
        """Reset drift detection state."""
        self.drift_detected = False
        self.consecutive_high_noise_batches = 0
        self.logger.info("Drift detection state reset")

    @property
    def is_drift_detected(self) -> bool:
        """Check if data drift is currently detected."""
        return self.drift_detected

    @property
    def current_noise_ratio(self) -> Optional[float]:
        """Get the most recent noise ratio."""
        return self.noise_ratio_history[-1] if self.noise_ratio_history else None

    @property
    def average_processing_time(self) -> Optional[float]:
        """Get average processing time per batch."""
        return np.mean(self.processing_times) if self.processing_times else None