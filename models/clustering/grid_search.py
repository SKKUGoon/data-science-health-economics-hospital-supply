"""
Utilities for running HDBSCAN grid searches with MLflow logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import hdbscan
import mlflow
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid

logger = logging.getLogger(__name__)


@dataclass
class GridSearchAttempt:
    """Container for a single grid-search attempt."""

    index: int
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float


@dataclass
class GridSearchResult:
    """Result bundle for HDBSCAN grid search."""

    model: hdbscan.HDBSCAN
    labels: np.ndarray
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float
    history: List[GridSearchAttempt] = field(default_factory=list)


class HDBSCANGridSearch:
    """Run HDBSCAN grid searches and emit diagnostics."""

    def __init__(
        self,
        base_params: Dict[str, Any],
        grid_spec: Optional[Iterable[Dict[str, Any]] | Dict[str, Iterable[Any]]] = None,
    ) -> None:
        self.base_params = base_params.copy()
        self.grid_spec = grid_spec

    def evaluate(self, X: np.ndarray) -> GridSearchResult:
        """Execute the search and return the best configuration."""
        history: List[GridSearchAttempt] = []
        best: Optional[GridSearchAttempt] = None
        best_model: Optional[hdbscan.HDBSCAN] = None
        best_labels: Optional[np.ndarray] = None

        for idx, params in enumerate(self._iter_param_sets()):
            try:
                model = hdbscan.HDBSCAN(**params)
                labels = model.fit_predict(X)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("HDBSCAN fit failed for params %s: %s", params, exc)
                self._log_attempt_failure(idx, params, exc)
                continue

            metrics = self.compute_metrics(X, labels)
            score = self._score_metrics(metrics)
            attempt = GridSearchAttempt(index=idx, params=params, metrics=metrics, score=score)
            history.append(attempt)

            self._log_attempt(attempt)

            if best is None or attempt.score > best.score:
                best = attempt
                best_model = model
                best_labels = labels

        if best is None or best_model is None or best_labels is None:
            raise RuntimeError("Grid search failed to produce a valid HDBSCAN model")

        result = GridSearchResult(
            model=best_model,
            labels=best_labels,
            params=best.params.copy(),
            metrics=best.metrics.copy(),
            score=best.score,
            history=history,
        )

        self._log_summary(result)
        result.metrics['selected_hdbscan_params'] = result.params.copy()
        result.metrics['grid_search_score'] = result.score
        result.metrics['grid_search_evaluated'] = len(history)

        return result

    def _iter_param_sets(self) -> Iterable[Dict[str, Any]]:
        if not self.grid_spec:
            yield self._with_prediction_flag(self.base_params)
            return

        if isinstance(self.grid_spec, dict):
            iterable = ParameterGrid(self.grid_spec)
        else:
            iterable = self.grid_spec

        for overrides in iterable:
            params = self.base_params.copy()
            params.update(overrides)
            yield self._with_prediction_flag(params)

    @staticmethod
    def _with_prediction_flag(params: Dict[str, Any]) -> Dict[str, Any]:
        params = params.copy()
        params['prediction_data'] = True
        return params

    @staticmethod
    def compute_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        non_noise_mask = labels != -1
        metrics: Dict[str, Any] = {
            'n_clusters': int(len(np.unique(labels[non_noise_mask]))) if np.any(non_noise_mask) else 0,
            'n_noise_points': int(np.sum(labels == -1)),
            'noise_ratio': float(np.sum(labels == -1) / len(labels)) if len(labels) else 0.0,
            'total_points': int(len(labels)),
        }

        if metrics['n_clusters'] > 1 and np.sum(non_noise_mask) > 1:
            metrics['silhouette_score'] = HDBSCANGridSearch._safe_metric(
                "silhouette", lambda: silhouette_score(X[non_noise_mask], labels[non_noise_mask])
            )
            metrics['calinski_harabasz_score'] = HDBSCANGridSearch._safe_metric(
                "calinski_harabasz", lambda: calinski_harabasz_score(X[non_noise_mask], labels[non_noise_mask])
            )
            metrics['davies_bouldin_score'] = HDBSCANGridSearch._safe_metric(
                "davies_bouldin", lambda: davies_bouldin_score(X[non_noise_mask], labels[non_noise_mask])
            )
        else:
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_score'] = None
            metrics['davies_bouldin_score'] = None

        return metrics

    def _score_metrics(self, metrics: Dict[str, Any]) -> float:
        if metrics.get('n_clusters', 0) == 0 or metrics.get('total_points', 0) == 0:
            return float('-inf')

        score = 0.0
        has_quality_metric = False

        silhouette = metrics.get('silhouette_score')
        if silhouette is not None:
            score += silhouette
            has_quality_metric = True

        calinski = metrics.get('calinski_harabasz_score')
        if calinski is not None and calinski > 0:
            score += np.log1p(calinski)
            has_quality_metric = True

        davies = metrics.get('davies_bouldin_score')
        if davies is not None and davies > 0:
            score -= davies
            has_quality_metric = True

        noise_ratio = metrics.get('noise_ratio', 0.0)
        score -= noise_ratio

        if not has_quality_metric:
            total_points = max(metrics.get('total_points', 1), 1)
            score -= metrics.get('n_noise_points', 0) / total_points

        return float(score)

    @staticmethod
    def _safe_metric(name: str, func) -> Optional[float]:
        try:
            value = func()
            return float(value)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to calculate %s score: %s", name, exc)
            return None

    def _log_attempt_failure(self, index: int, params: Dict[str, Any], exc: Exception) -> None:
        if mlflow.active_run() is None:
            return
        try:
            run_name = f"hdbscan_grid_{index:03d}_failed"
            with mlflow.start_run(run_name=run_name, nested=True):
                payload = {f"hdbscan__{key}": str(value) for key, value in params.items()}
                if payload:
                    mlflow.log_params(payload)
                mlflow.log_param("status", "failed")
                mlflow.log_param("exception", str(exc))
        except Exception as log_exc:  # pragma: no cover - defensive
            logger.debug("Skipping MLflow logging for failed attempt %s: %s", index, log_exc)

    def _log_attempt(self, attempt: GridSearchAttempt) -> None:
        if mlflow.active_run() is None:
            return

        try:
            run_name = f"hdbscan_grid_{attempt.index:03d}"
            with mlflow.start_run(run_name=run_name, nested=True):
                param_payload = {f"hdbscan__{key}": str(value) for key, value in attempt.params.items()}
                if param_payload:
                    mlflow.log_params(param_payload)

                metric_payload = {}
                for key, value in attempt.metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        metric_payload[f"hdbscan_{key}"] = float(value)
                metric_payload['hdbscan_grid_score'] = float(attempt.score)

                if metric_payload:
                    mlflow.log_metrics(metric_payload)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skipping MLflow logging for grid search attempt %s: %s", attempt.index, exc)

    def _log_summary(self, result: GridSearchResult) -> None:
        if mlflow.active_run() is None:
            return

        try:
            summary_metrics = {}
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    summary_metrics[f"hdbscan_grid_best_{key}"] = float(value)
            summary_metrics['hdbscan_grid_best_score'] = float(result.score)

            if summary_metrics:
                mlflow.log_metrics(summary_metrics)

            tag_payload = {
                f"hdbscan_grid_best__{key}": str(value)
                for key, value in result.params.items()
            }
            if tag_payload:
                mlflow.set_tags(tag_payload)

            artifact_payload = {
                'best_params': result.params,
                'best_score': float(result.score),
                'best_metrics': result.metrics,
                'grid_search_history': [
                    {
                        'attempt': attempt.index,
                        'params': attempt.params,
                        'metrics': attempt.metrics,
                        'score': attempt.score,
                    }
                    for attempt in result.history
                ],
            }
            mlflow.log_dict(artifact_payload, artifact_file="clustering/hdbscan_grid_search.json")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to log grid search summary to MLflow: %s", exc)
