"""
Adaptive Clustering Engine for HDBSCAN with kNN fallback.

This module implements the AdaptiveClusteringEngine class that handles
HDBSCAN clustering with StandardScaler preprocessing and kNN fallback
for noise point handling.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
import logging

from ..core.data_models import ClusteringResult
from ..core.exceptions import ModelTrainingError, PredictionError, DataValidationError
from .knn_fallback import KNNFallbackModel
from .knn_fallback_reporter import KNNFallbackReporter


logger = logging.getLogger(__name__)


class AdaptiveClusteringEngine:
    """
    Adaptive clustering engine using HDBSCAN with kNN fallback.

    This class provides HDBSCAN clustering with StandardScaler preprocessing
    and kNN fallback mechanism for handling noise points during prediction.
    """

    def __init__(self, hdbscan_params: Dict[str, Any], knn_params: Dict[str, Any]):
        """
        Initialize the adaptive clustering engine.

        Args:
            hdbscan_params: Parameters for HDBSCAN clustering
            knn_params: Parameters for kNN fallback model
        """
        self.hdbscan_params = hdbscan_params.copy()
        self.knn_params = knn_params.copy()

        # Ensure prediction is enabled for HDBSCAN
        self.hdbscan_params['prediction_data'] = True

        # Initialize components
        self.scaler = StandardScaler()
        self.hdbscan_model = None
        self.knn_fallback = KNNFallbackModel(knn_params)
        self.knn_reporter = KNNFallbackReporter()

        # Training data and results
        self._X_scaled = None
        self._labels = None
        self._cluster_centers = None
        self._is_fitted = False
        self._knn_training_metrics = None

        logger.info("AdaptiveClusteringEngine initialized with HDBSCAN params: %s",
                   self.hdbscan_params)
        logger.info("kNN fallback params: %s", self.knn_params)

    def fit(self, X: np.ndarray) -> ClusteringResult:
        """
        Fit HDBSCAN clustering model with kNN fallback.

        Args:
            X: Input data array of shape (n_samples, n_features)

        Returns:
            ClusteringResult containing labels, centers, noise ratio, and metrics

        Raises:
            DataValidationError: If input data is invalid
            ModelTrainingError: If model training fails
        """
        logger.info("Starting HDBSCAN clustering fit on data shape: %s", X.shape)

        # Validate input data
        self._validate_input_data(X)

        try:
            # Scale the data
            self._X_scaled = self.scaler.fit_transform(X)
            logger.info("Data scaling completed")

            # Fit HDBSCAN model
            self.hdbscan_model = hdbscan.HDBSCAN(**self.hdbscan_params)
            self._labels = self.hdbscan_model.fit_predict(self._X_scaled)

            # Calculate cluster centers
            self._cluster_centers = self._calculate_cluster_centers()

            # Calculate noise ratio
            noise_ratio = np.sum(self._labels == -1) / len(self._labels)

            logger.info("HDBSCAN clustering completed. Found %d clusters with %.2f%% noise points",
                       len(np.unique(self._labels[self._labels != -1])), noise_ratio * 100)

            # Fit kNN fallback model
            self._fit_knn_fallback()

            # Set fitted flag before calculating metrics
            self._is_fitted = True

            # Calculate clustering metrics
            metrics = self.get_clustering_metrics()

            return ClusteringResult(
                labels=self._labels.copy(),
                cluster_centers=self._cluster_centers.copy(),
                noise_ratio=noise_ratio,
                metrics=metrics
            )

        except Exception as e:
            raise ModelTrainingError(
                f"HDBSCAN clustering failed: {str(e)}",
                model_type="HDBSCAN",
                training_data_size=X.shape[0]
            ) from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data with fallback handling.

        Args:
            X: Input data array of shape (n_samples, n_features)

        Returns:
            Array of cluster assignments

        Raises:
            PredictionError: If prediction fails
            DataValidationError: If input data is invalid
        """
        if not self._is_fitted:
            raise PredictionError("Model must be fitted before prediction",
                                prediction_type="cluster_assignment")

        logger.info("Predicting cluster assignments for data shape: %s", X.shape)

        # Validate input data
        self._validate_input_data(X)

        try:
            # Scale the data using fitted scaler
            X_scaled = self.scaler.transform(X)

            # Use HDBSCAN approximate prediction
            labels, strengths = hdbscan.approximate_predict(self.hdbscan_model, X_scaled)

            # Find noise points (where HDBSCAN prediction failed)
            noise_mask = labels == -1
            noise_count = np.sum(noise_mask)

            if noise_count > 0:
                logger.info("Using kNN fallback for %d noise points", noise_count)

                # Use kNN fallback for noise points
                if self.knn_fallback is not None and self.knn_fallback.is_fitted:
                    fallback_predictions, fallback_metrics = self.knn_fallback.predict(X_scaled[noise_mask])
                    labels[noise_mask] = fallback_predictions
                    logger.info("kNN fallback completed with avg confidence: %.4f",
                               fallback_metrics.get('avg_confidence', 0.0))
                else:
                    logger.warning("kNN fallback model not available or not fitted, keeping noise labels")

            logger.info("Prediction completed. Assigned %d points to clusters", len(labels))
            return labels

        except Exception as e:
            raise PredictionError(
                f"Cluster prediction failed: {str(e)}",
                prediction_type="cluster_assignment",
                batch_size=X.shape[0]
            ) from e

    def get_clustering_metrics(self) -> Dict[str, Any]:
        """
        Calculate clustering quality metrics.

        Returns:
            Dictionary containing clustering metrics

        Raises:
            PredictionError: If model is not fitted
        """
        if not self._is_fitted:
            raise PredictionError("Model must be fitted before calculating metrics",
                                prediction_type="clustering_metrics")

        try:
            # Get non-noise points for metric calculation
            non_noise_mask = self._labels != -1
            X_non_noise = self._X_scaled[non_noise_mask]
            labels_non_noise = self._labels[non_noise_mask]

            metrics = {
                'n_clusters': len(np.unique(labels_non_noise)),
                'n_noise_points': np.sum(self._labels == -1),
                'noise_ratio': np.sum(self._labels == -1) / len(self._labels),
                'total_points': len(self._labels)
            }

            # Calculate quality metrics only if we have multiple clusters
            if len(np.unique(labels_non_noise)) > 1 and len(X_non_noise) > 1:
                try:
                    metrics['silhouette_score'] = silhouette_score(X_non_noise, labels_non_noise)
                except Exception as e:
                    logger.warning("Failed to calculate silhouette score: %s", str(e))
                    metrics['silhouette_score'] = None

                try:
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_non_noise, labels_non_noise)
                except Exception as e:
                    logger.warning("Failed to calculate Calinski-Harabasz score: %s", str(e))
                    metrics['calinski_harabasz_score'] = None

                try:
                    metrics['davies_bouldin_score'] = davies_bouldin_score(X_non_noise, labels_non_noise)
                except Exception as e:
                    logger.warning("Failed to calculate Davies-Bouldin score: %s", str(e))
                    metrics['davies_bouldin_score'] = None
            else:
                logger.info("Insufficient clusters or data for quality metrics calculation")
                metrics['silhouette_score'] = None
                metrics['calinski_harabasz_score'] = None
                metrics['davies_bouldin_score'] = None

            return metrics

        except Exception as e:
            raise PredictionError(
                f"Failed to calculate clustering metrics: {str(e)}",
                prediction_type="clustering_metrics"
            ) from e

    def update_fallback_model(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Update the kNN fallback model with new data.

        Args:
            X: Input data array
            labels: Cluster labels for the data

        Returns:
            Dictionary containing kNN training metrics

        Raises:
            ModelTrainingError: If fallback model update fails
        """
        logger.info("Updating kNN fallback model with %d samples", X.shape[0])

        try:
            # Scale the data
            X_scaled = self.scaler.transform(X)

            # Fit the kNN fallback model
            self._knn_training_metrics = self.knn_fallback.fit(X_scaled, labels)

            logger.info("kNN fallback model updated successfully with accuracy: %.4f",
                       self._knn_training_metrics.get('training_accuracy', 0.0))

            return self._knn_training_metrics

        except Exception as e:
            raise ModelTrainingError(
                f"Failed to update kNN fallback model: {str(e)}",
                model_type="kNN_fallback",
                training_data_size=X.shape[0]
            ) from e

    def _validate_input_data(self, X: np.ndarray) -> None:
        """
        Validate input data format and content.

        Args:
            X: Input data to validate

        Raises:
            DataValidationError: If data validation fails
        """
        if not isinstance(X, np.ndarray):
            raise DataValidationError("Input data must be a numpy array", data_type=type(X).__name__)

        if X.ndim != 2:
            raise DataValidationError(
                "Input data must be 2-dimensional",
                expected_shape="(n_samples, n_features)",
                actual_shape=X.shape
            )

        if X.shape[0] == 0:
            raise DataValidationError("Input data cannot be empty", actual_shape=X.shape)

        if X.shape[1] == 0:
            raise DataValidationError("Input data must have at least one feature", actual_shape=X.shape)

        # Check for invalid values
        if np.any(np.isnan(X)):
            raise DataValidationError("Input data contains NaN values")

        if np.any(np.isinf(X)):
            raise DataValidationError("Input data contains infinite values")

    def _calculate_cluster_centers(self) -> np.ndarray:
        """
        Calculate cluster centers from fitted data.

        Returns:
            Array of cluster centers
        """
        unique_labels = np.unique(self._labels)
        # Remove noise label (-1) if present
        unique_labels = unique_labels[unique_labels != -1]

        if len(unique_labels) == 0:
            logger.warning("No valid clusters found, returning empty centers")
            return np.array([]).reshape(0, self._X_scaled.shape[1])

        centers = []
        for label in unique_labels:
            cluster_mask = self._labels == label
            cluster_center = np.mean(self._X_scaled[cluster_mask], axis=0)
            centers.append(cluster_center)

        return np.array(centers)

    def _fit_knn_fallback(self) -> None:
        """
        Fit kNN fallback model using non-noise points.
        """
        try:
            # Fit the kNN fallback model
            self._knn_training_metrics = self.knn_fallback.fit(self._X_scaled, self._labels)
            logger.info("kNN fallback model fitted successfully with accuracy: %.4f",
                       self._knn_training_metrics.get('training_accuracy', 0.0))
        except Exception as e:
            logger.error("Failed to fit kNN fallback model: %s", str(e))
            self._knn_training_metrics = None

    @property
    def is_fitted(self) -> bool:
        """Check if the model is fitted."""
        return self._is_fitted

    @property
    def cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers if model is fitted."""
        return self._cluster_centers.copy() if self._is_fitted else None

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Get training labels if model is fitted."""
        return self._labels.copy() if self._is_fitted else None

    def get_knn_fallback_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get kNN fallback model performance metrics.

        Returns:
            Dictionary containing kNN fallback metrics or None if not fitted
        """
        if not self._is_fitted or self.knn_fallback is None:
            return None

        try:
            fallback_metrics = self.knn_fallback.get_performance_metrics()
            if self._knn_training_metrics:
                fallback_metrics.update(self._knn_training_metrics)
            return fallback_metrics
        except Exception as e:
            logger.error("Failed to get kNN fallback metrics: %s", str(e))
            return None

    def generate_knn_fallback_reports(self,
                                    save_path: Optional[str] = None,
                                    usage_history: Optional[List[Dict[str, Any]]] = None,
                                    evaluation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive kNN fallback reports and visualizations.

        Args:
            save_path: Optional path to save visualizations
            usage_history: Optional historical usage data
            evaluation_data: Optional tuple of (X_test, y_test) for evaluation

        Returns:
            Dictionary containing all report data and visualization paths

        Raises:
            PredictionError: If report generation fails
        """
        if not self._is_fitted or self.knn_fallback is None:
            raise PredictionError(
                "kNN fallback model must be fitted before generating reports",
                prediction_type="knn_fallback_reports"
            )

        logger.info("Generating comprehensive kNN fallback reports")

        try:
            reports = {}

            # Get current fallback metrics
            fallback_metrics = self.get_knn_fallback_metrics()
            if not fallback_metrics:
                raise PredictionError(
                    "Failed to retrieve kNN fallback metrics",
                    prediction_type="knn_fallback_reports"
                )

            # Generate usage report
            usage_report = self.knn_reporter.generate_fallback_usage_report(
                fallback_metrics, usage_history
            )
            reports['usage_report'] = usage_report

            # Generate usage visualizations
            if save_path:
                usage_viz = self.knn_reporter.create_fallback_usage_visualizations(
                    usage_report, save_path
                )
                reports['usage_visualizations'] = usage_viz

            # Generate accuracy report if evaluation data is provided
            if evaluation_data is not None:
                X_test, y_test = evaluation_data
                evaluation_metrics = self.knn_fallback.evaluate_on_test_data(X_test, y_test)

                accuracy_report = self.knn_reporter.generate_fallback_accuracy_report(
                    evaluation_metrics
                )
                reports['accuracy_report'] = accuracy_report

                # Generate accuracy visualizations
                if save_path:
                    accuracy_viz = self.knn_reporter.create_fallback_accuracy_visualizations(
                        accuracy_report, save_path
                    )
                    reports['accuracy_visualizations'] = accuracy_viz

            logger.info("kNN fallback reports generated successfully")
            return reports

        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(
                f"Failed to generate kNN fallback reports: {str(e)}",
                prediction_type="knn_fallback_reports"
            ) from e