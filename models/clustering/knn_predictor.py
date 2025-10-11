"""
kNN predictor for cluster assignments using HDBSCAN reference data.

This module provides the kNNPredictor class, which learns from the
HDBSCAN-fitted dataset and assigns clusters to incoming points using
the same feature representation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from core.exceptions import ModelTrainingError, PredictionError, ValidationError

logger = logging.getLogger(__name__)


class KNNPredictor:
    """
    kNN-based predictor trained on HDBSCAN cluster assignments.

    The predictor learns from the (reduced) feature space generated during
    HDBSCAN fitting and uses the resulting kNN model to assign clusters to
    new data points.
    """

    def __init__(self, knn_params: Dict[str, Any]):
        self.knn_params = knn_params.copy()
        self.knn_model: Optional[KNeighborsClassifier] = None
        self._is_fitted = False

        # Performance tracking
        self._training_accuracy: Optional[float] = None
        self._training_data_size: int = 0
        self._unique_classes: Optional[np.ndarray] = None

        logger.info("KNNPredictor initialized with params: %s", self.knn_params)

    def fit(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Fit the kNN predictor on HDBSCAN-labelled data.

        Args:
            X: Feature array (typically UMAP-reduced) of shape (n_samples, n_features).
            labels: Cluster labels from HDBSCAN.

        Returns:
            Dictionary with training metrics.
        """
        logger.info("Fitting kNN predictor on data shape: %s", X.shape)
        self._validate_training_data(X, labels)

        try:
            self.knn_model = KNeighborsClassifier(**self.knn_params)
            self.knn_model.fit(X, labels)

            y_pred_train = self.knn_model.predict(X)
            self._training_accuracy = accuracy_score(labels, y_pred_train)
            self._training_data_size = int(X.shape[0])
            self._unique_classes = np.unique(labels)
            self._is_fitted = True

            metrics = {
                "training_accuracy": self._training_accuracy,
                "training_data_size": self._training_data_size,
                "n_unique_classes": len(self._unique_classes),
                "unique_classes": self._unique_classes.tolist(),
                "knn_params": self.knn_params.copy(),
            }
            logger.info(
                "kNN predictor fitted successfully. Training accuracy: %.4f, classes: %d",
                self._training_accuracy,
                len(self._unique_classes),
            )
            return metrics
        except Exception as exc:
            raise ModelTrainingError(
                f"kNN predictor training failed: {exc}",
                {"model_type": "kNN_predictor", "training_data_size": int(X.shape[0])},
            ) from exc

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Assign clusters to new data using the fitted kNN model."""
        if not self._is_fitted or self.knn_model is None:
            raise PredictionError(
                "kNN predictor must be fitted before prediction",
                {"prediction_type": "knn_predictor"},
            )

        logger.info("Predicting with kNN predictor for %d samples", X.shape[0])
        self._validate_prediction_data(X)

        try:
            predictions = self.knn_model.predict(X)
            probabilities = self.knn_model.predict_proba(X)

            max_probs = np.max(probabilities, axis=1)
            metrics = {
                "n_predictions": len(predictions),
                "avg_confidence": float(np.mean(max_probs)),
                "min_confidence": float(np.min(max_probs)),
                "predicted_classes": np.unique(predictions).tolist(),
            }
            return predictions, metrics
        except Exception as exc:
            raise PredictionError(
                f"kNN predictor failed to assign clusters: {exc}",
                {"prediction_type": "knn_predictor", "batch_size": int(X.shape[0])},
            ) from exc

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _validate_training_data(self, X: np.ndarray, labels: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise ValidationError("Input data X must be a numpy array", {"data_type": type(X).__name__})
        if X.ndim != 2:
            raise ValidationError(
                "Input data X must be 2-dimensional",
                {"expected_shape": "(n_samples, n_features)", "actual_shape": X.shape},
            )
        if X.shape[0] == 0:
            raise ValidationError("Input data X cannot be empty", {"actual_shape": X.shape})
        if not isinstance(labels, np.ndarray):
            raise ValidationError("Labels must be a numpy array", {"data_type": type(labels).__name__})
        if labels.ndim != 1:
            raise ValidationError(
                "Labels must be 1-dimensional",
                {"expected_shape": "(n_samples,)", "actual_shape": labels.shape},
            )
        if len(labels) != len(X):
            raise ValidationError(
                "X and labels must have the same number of samples",
                {"expected_size": int(X.shape[0]), "actual_size": int(len(labels))},
            )
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValidationError("Input data X contains NaN or infinite values")

    def _validate_prediction_data(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise ValidationError("Input data must be a numpy array", {"data_type": type(X).__name__})
        if X.ndim != 2:
            raise ValidationError(
                "Input data must be 2-dimensional",
                {"expected_shape": "(n_samples, n_features)", "actual_shape": X.shape},
            )
        if X.shape[0] == 0:
            raise ValidationError("Input data cannot be empty", {"actual_shape": X.shape})
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValidationError("Input data contains NaN or infinite values")
