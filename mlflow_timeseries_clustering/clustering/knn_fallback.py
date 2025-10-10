"""
kNN Fallback Model for handling noise points in HDBSCAN clustering.

This module implements the KNNFallbackModel class that provides fallback
cluster assignment for noise points identified by HDBSCAN clustering.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

from ..core.exceptions import ModelTrainingError, PredictionError, DataValidationError


logger = logging.getLogger(__name__)


class KNNFallbackModel:
    """
    kNN fallback model for noise point handling in HDBSCAN clustering.

    This class provides a fallback mechanism for assigning cluster labels
    to noise points that HDBSCAN cannot classify during prediction.
    """

    def __init__(self, knn_params: Dict[str, Any]):
        """
        Initialize the kNN fallback model.

        Args:
            knn_params: Parameters for kNN classifier
        """
        self.knn_params = knn_params.copy()
        self.knn_model = None
        self._is_fitted = False

        # Performance tracking
        self._training_accuracy = None
        self._fallback_usage_count = 0
        self._fallback_predictions_count = 0
        self._training_data_size = 0
        self._unique_classes = None

        logger.info("KNNFallbackModel initialized with params: %s", self.knn_params)

    def fit(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Fit kNN model using cluster assignments from HDBSCAN.

        Args:
            X: Scaled input data array of shape (n_samples, n_features)
            labels: Cluster labels from HDBSCAN (excluding noise points)

        Returns:
            Dictionary containing training metrics

        Raises:
            DataValidationError: If input data is invalid
            ModelTrainingError: If model training fails
        """
        logger.info("Fitting kNN fallback model on data shape: %s", X.shape)

        # Validate input data
        self._validate_training_data(X, labels)

        try:
            # Filter out noise points for training
            non_noise_mask = labels != -1

            if np.sum(non_noise_mask) == 0:
                raise ModelTrainingError(
                    "No non-noise points available for kNN fallback training",
                    model_type="kNN_fallback",
                    training_data_size=X.shape[0]
                )

            X_train = X[non_noise_mask]
            y_train = labels[non_noise_mask]

            self._training_data_size = len(X_train)
            self._unique_classes = np.unique(y_train)

            # Check if we have enough data and multiple classes
            min_neighbors = self.knn_params.get('n_neighbors', 5)
            if len(X_train) < min_neighbors:
                raise ModelTrainingError(
                    f"Insufficient data for kNN training: {len(X_train)} < {min_neighbors}",
                    model_type="kNN_fallback",
                    training_data_size=len(X_train)
                )

            if len(self._unique_classes) < 2:
                raise ModelTrainingError(
                    f"Insufficient cluster diversity: only {len(self._unique_classes)} unique classes",
                    model_type="kNN_fallback",
                    training_data_size=len(X_train)
                )

            # Fit kNN model
            self.knn_model = KNeighborsClassifier(**self.knn_params)
            self.knn_model.fit(X_train, y_train)

            # Calculate training accuracy
            y_pred_train = self.knn_model.predict(X_train)
            self._training_accuracy = accuracy_score(y_train, y_pred_train)

            self._is_fitted = True

            # Prepare training metrics
            training_metrics = {
                'training_accuracy': self._training_accuracy,
                'training_data_size': self._training_data_size,
                'n_unique_classes': len(self._unique_classes),
                'unique_classes': self._unique_classes.tolist(),
                'knn_params': self.knn_params.copy()
            }

            logger.info("kNN fallback model fitted successfully. Training accuracy: %.4f, Classes: %d",
                       self._training_accuracy, len(self._unique_classes))

            return training_metrics

        except Exception as e:
            if isinstance(e, (ModelTrainingError, DataValidationError)):
                raise
            raise ModelTrainingError(
                f"kNN fallback model training failed: {str(e)}",
                model_type="kNN_fallback",
                training_data_size=X.shape[0]
            ) from e

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict cluster assignments for noise points using kNN fallback.

        Args:
            X: Scaled input data array of shape (n_samples, n_features)

        Returns:
            Tuple of (predictions, prediction_metrics)

        Raises:
            PredictionError: If prediction fails
            DataValidationError: If input data is invalid
        """
        if not self._is_fitted:
            raise PredictionError(
                "kNN fallback model must be fitted before prediction",
                prediction_type="knn_fallback"
            )

        logger.info("Predicting with kNN fallback for %d samples", X.shape[0])

        # Validate input data
        self._validate_prediction_data(X)

        try:
            # Make predictions
            predictions = self.knn_model.predict(X)
            prediction_probabilities = self.knn_model.predict_proba(X)

            # Update usage tracking
            self._fallback_usage_count += 1
            self._fallback_predictions_count += len(predictions)

            # Calculate prediction confidence metrics
            max_probabilities = np.max(prediction_probabilities, axis=1)
            avg_confidence = np.mean(max_probabilities)
            min_confidence = np.min(max_probabilities)

            # Prepare prediction metrics
            prediction_metrics = {
                'n_predictions': len(predictions),
                'avg_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'predicted_classes': np.unique(predictions).tolist(),
                'class_distribution': {
                    int(cls): int(np.sum(predictions == cls))
                    for cls in np.unique(predictions)
                }
            }

            logger.info("kNN fallback prediction completed. Avg confidence: %.4f, Classes: %s",
                       avg_confidence, prediction_metrics['predicted_classes'])

            return predictions, prediction_metrics

        except Exception as e:
            raise PredictionError(
                f"kNN fallback prediction failed: {str(e)}",
                prediction_type="knn_fallback",
                batch_size=X.shape[0]
            ) from e

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for the kNN fallback model.

        Returns:
            Dictionary containing performance metrics

        Raises:
            PredictionError: If model is not fitted
        """
        if not self._is_fitted:
            raise PredictionError(
                "kNN fallback model must be fitted before getting metrics",
                prediction_type="knn_fallback_metrics"
            )

        metrics = {
            'is_fitted': self._is_fitted,
            'training_accuracy': self._training_accuracy,
            'training_data_size': self._training_data_size,
            'n_unique_classes': len(self._unique_classes) if self._unique_classes is not None else 0,
            'unique_classes': self._unique_classes.tolist() if self._unique_classes is not None else [],
            'fallback_usage_count': self._fallback_usage_count,
            'total_fallback_predictions': self._fallback_predictions_count,
            'avg_predictions_per_usage': (
                self._fallback_predictions_count / self._fallback_usage_count
                if self._fallback_usage_count > 0 else 0
            ),
            'knn_params': self.knn_params.copy()
        }

        return metrics

    def evaluate_on_test_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate kNN fallback model on test data.

        Args:
            X_test: Test input data
            y_test: True test labels

        Returns:
            Dictionary containing evaluation metrics

        Raises:
            PredictionError: If model is not fitted or evaluation fails
        """
        if not self._is_fitted:
            raise PredictionError(
                "kNN fallback model must be fitted before evaluation",
                prediction_type="knn_fallback_evaluation"
            )

        logger.info("Evaluating kNN fallback model on test data shape: %s", X_test.shape)

        try:
            # Make predictions
            y_pred = self.knn_model.predict(X_test)
            y_proba = self.knn_model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Get classification report as dict
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Calculate confidence metrics
            max_probabilities = np.max(y_proba, axis=1)
            avg_confidence = np.mean(max_probabilities)

            evaluation_metrics = {
                'test_accuracy': accuracy,
                'avg_test_confidence': avg_confidence,
                'classification_report': class_report,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'test_data_size': len(X_test),
                'n_test_classes': len(np.unique(y_test))
            }

            logger.info("kNN fallback evaluation completed. Test accuracy: %.4f", accuracy)

            return evaluation_metrics

        except Exception as e:
            raise PredictionError(
                f"kNN fallback evaluation failed: {str(e)}",
                prediction_type="knn_fallback_evaluation",
                batch_size=X_test.shape[0]
            ) from e

    def reset_usage_tracking(self) -> None:
        """Reset fallback usage tracking counters."""
        self._fallback_usage_count = 0
        self._fallback_predictions_count = 0
        logger.info("kNN fallback usage tracking reset")

    def _validate_training_data(self, X: np.ndarray, labels: np.ndarray) -> None:
        """
        Validate training data format and content.

        Args:
            X: Input data to validate
            labels: Labels to validate

        Raises:
            DataValidationError: If data validation fails
        """
        # Validate X
        if not isinstance(X, np.ndarray):
            raise DataValidationError("Input data X must be a numpy array", data_type=type(X).__name__)

        if X.ndim != 2:
            raise DataValidationError(
                "Input data X must be 2-dimensional",
                expected_shape="(n_samples, n_features)",
                actual_shape=X.shape
            )

        if X.shape[0] == 0:
            raise DataValidationError("Input data X cannot be empty", actual_shape=X.shape)

        # Validate labels
        if not isinstance(labels, np.ndarray):
            raise DataValidationError("Labels must be a numpy array", data_type=type(labels).__name__)

        if labels.ndim != 1:
            raise DataValidationError(
                "Labels must be 1-dimensional",
                expected_shape="(n_samples,)",
                actual_shape=labels.shape
            )

        if len(X) != len(labels):
            raise DataValidationError(
                f"X and labels must have same length: {len(X)} != {len(labels)}"
            )

        # Check for invalid values in X
        if np.any(np.isnan(X)):
            raise DataValidationError("Input data X contains NaN values")

        if np.any(np.isinf(X)):
            raise DataValidationError("Input data X contains infinite values")

    def _validate_prediction_data(self, X: np.ndarray) -> None:
        """
        Validate prediction data format and content.

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

        # Check for invalid values
        if np.any(np.isnan(X)):
            raise DataValidationError("Input data contains NaN values")

        if np.any(np.isinf(X)):
            raise DataValidationError("Input data contains infinite values")

    @property
    def is_fitted(self) -> bool:
        """Check if the model is fitted."""
        return self._is_fitted

    @property
    def training_accuracy(self) -> Optional[float]:
        """Get training accuracy if model is fitted."""
        return self._training_accuracy

    @property
    def fallback_usage_count(self) -> int:
        """Get number of times fallback has been used."""
        return self._fallback_usage_count

    @property
    def total_fallback_predictions(self) -> int:
        """Get total number of predictions made by fallback."""
        return self._fallback_predictions_count