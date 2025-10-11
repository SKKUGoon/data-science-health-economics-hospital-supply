import logging
from datetime import datetime
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import mlflow

from core.config import PipelineConfig
from core.data_models import ClusteringResult
from models.data_container.expanding_window_feeder import ExpandingWindowFeeder
from models.clustering.engine import AdaptiveClusteringEngine

logger = logging.getLogger(__name__)


class BatchBacktestSummary(BaseModel):
    """Container for per-batch backtesting diagnostics."""

    train_start: datetime = Field(..., description="Start of the training window")
    train_end: datetime = Field(..., description="End of the training window")
    test_start: datetime = Field(..., description="Start of the test window")
    test_end: datetime = Field(..., description="End of the test window")
    n_train: int = Field(..., ge=0, description="Number of training samples")
    n_test: int = Field(..., ge=0, description="Number of test samples")

    model_num: int = Field(..., ge=0, description="Model number (starting from 0)")
    clustering_result: Optional[ClusteringResult] = Field(
        default=None,
        description="Clustering result captured after fitting/retraining (if available)"
    )
    train_asssignments: Optional[pd.DataFrame] = Field(
        default=None,
        description="Cumulative training records since the last refit with id/date/cluster identifiers"
    )
    test_assignments: Optional[pd.DataFrame] = Field(
        default=None,
        description="Cumulative testing records since the last refit with id/date/cluster identifiers"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "arbitrary_types_allowed": True,
    }


class BacktestingExperiment:
    """
    Run expanding-window clustering with automatic HDBSCAN retraining.

    Each batch uses ``ExpandingWindowFeeder`` to draw the expanding training span
    and a one-week test window. The model is refit on schedule and incoming data
    is assigned using the kNN reference built from the latest HDBSCAN fit.
    """

    def __init__(
        self,
        config: PipelineConfig,
        feeder: Optional[ExpandingWindowFeeder] = None,
        refit_interval_weeks: int = 13,
    ) -> None:
        if refit_interval_weeks <= 0:
            raise ValueError("refit_interval_weeks must be positive")

        self.config = config
        self.feeder = feeder or ExpandingWindowFeeder()
        self.refit_interval_weeks = int(refit_interval_weeks)
        self._engine: Optional[AdaptiveClusteringEngine] = None
        self._history: List[BatchBacktestSummary] = []

    @property
    def history(self) -> List[BatchBacktestSummary]:
        """Return collected batch summaries."""
        return self._history

    def run(
        self,
        data: pd.DataFrame,
        datetime_col: str,
        feature_cols: Sequence[str],
        id_cols: Sequence[str] = ("id", "date"),
        run_name: Optional[str] = None,
        log_to_mlflow: bool = True,
    ) -> Iterable[BatchBacktestSummary]:
        """
        Execute the experiment over expanding-window batches.

        Args:
            data: DataFrame containing datetime and feature columns.
            datetime_col: Column identifying chronological order.
            feature_cols: Columns used as clustering features (must be numeric).
            id_cols: Columns preserved alongside cluster assignments.
            run_name: Optional name for the parent MLflow run.
            log_to_mlflow: Disable MLflow logging when False (useful for tests).

        Yields:
            BatchBacktestSummary for each processed batch.
        """
        feature_cols = tuple(feature_cols)
        if len(feature_cols) == 0:
            raise ValueError("feature_cols must contain at least one column name")

        id_cols = tuple(id_cols)
        if len(id_cols) == 0:
            raise ValueError("id_cols must contain at least one column name")

        missing_ids = [col for col in id_cols if col not in data.columns]
        if missing_ids:
            raise KeyError(f"Missing id columns: {missing_ids}")

        self._history = []
        batch_generator = self._process_batches(data, datetime_col, feature_cols, id_cols, log_to_mlflow)

        if not log_to_mlflow:
            yield from batch_generator
            return

        mlflow.set_experiment(self.config.experiment_name)
        parent_run_name = run_name or f"backtesting_{datetime.now():%Y%m%d_%H%M%S}"

        active_run = mlflow.active_run()
        run_kwargs = {"run_name": parent_run_name}
        if active_run is not None:
            logger.debug("Found active MLflow run %s; using nested run", active_run.info.run_id)
            run_kwargs["nested"] = True

        with mlflow.start_run(**run_kwargs):
            self._log_parent_params()
            for summary in batch_generator:
                yield summary

    def _fit_new_engine(self, train_X: np.ndarray) -> ClusteringResult:
        """Fit a new clustering engine on the training slice."""
        logger.info("Fitting AdaptiveClusteringEngine on %d samples", train_X.shape[0])
        self._engine = AdaptiveClusteringEngine(
            hdbscan_params=self.config.hdbscan_params,
            knn_params=self.config.knn_params,
            umap_params=self.config.umap_params,
            clustering_param_grid={
                "min_cluster_size": [2, 3, 5, 10],
                "min_samples": [2, 3, 5],
                "cluster_selection_epsilon": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "cluster_selection_method": ["eom", "leaf"]
            }
        )
        return self._engine.fit(train_X)

    def _refit(self, train_X: np.ndarray) -> ClusteringResult:
        """Refit the current engine on the provided training data."""
        self._engine = AdaptiveClusteringEngine(
            hdbscan_params=self.config.hdbscan_params,
            knn_params=self.config.knn_params,
            umap_params=self.config.umap_params,
        )
        return self._engine.fit(train_X)

    @staticmethod
    def _to_numpy(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
        """Convert selected columns to a numpy array with validation."""
        cols = list(feature_cols)
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        values = df[cols]
        if not all(pd.api.types.is_numeric_dtype(values[col]) for col in cols):
            raise TypeError("feature_cols must be numeric")
        return values.to_numpy(dtype=float)

    @staticmethod
    def _as_datetime(value: object) -> datetime:
        """Convert pandas/scalar timestamps to native datetime."""
        if isinstance(value, datetime):
            return value
        to_py = getattr(value, "to_pydatetime", None)
        if callable(to_py):
            return to_py()
        return pd.to_datetime(value).to_pydatetime()

    def _process_batches(
        self,
        data: pd.DataFrame,
        datetime_col: str,
        feature_cols: Sequence[str],
        id_cols: Sequence[str],
        log_to_mlflow: bool,
    ) -> Iterator[BatchBacktestSummary]:
        """Iterate over feeder windows and optionally log each batch."""
        refit_interval = self.refit_interval_weeks
        batches_since_refit = refit_interval
        cumulative_assignments: Optional[pd.DataFrame] = None
        model_num = -1

        for train_df, test_df, train_date_period, test_date_period in self.feeder.feed(data, datetime_col):
            if train_df.empty or test_df.empty:
                logger.debug("Skipping batch with train=%d test=%d", len(train_df), len(test_df))
                continue

            need_refit = self._engine is None or batches_since_refit >= refit_interval
            if need_refit:
                train_X = self._to_numpy(train_df, feature_cols)
                clustering_result = self._fit_new_engine(train_X) if self._engine is None else self._refit(train_X)

                model_num += 1  # new model is fitted with expanded dataset
                batches_since_refit = 0
                cumulative_assignments = None  # reset for newer model

                train_assign_df = train_df.loc[:, list(dict.fromkeys(id_cols))].copy()
                train_assign_df['cluster_id'] = clustering_result.labels
            else:
                clustering_result = self._engine.get_fitted_result()
                batches_since_refit += 1

            test_X = self._to_numpy(test_df, feature_cols)
            predictions = self._engine.predict(test_X)
            test_assign_df = test_df.loc[:, list(dict.fromkeys(id_cols))].copy()
            test_assign_df['cluster_id'] = predictions
            test_assign_df = test_assign_df.drop(columns=['cluster_info'], errors='ignore')

            if cumulative_assignments is None:
                cumulative_assignments = test_assign_df.copy()
            else:
                cumulative_assignments = pd.concat([cumulative_assignments, test_assign_df], axis=0)

            logger.info(
                "Batch train %s-%s | test %s-%s",
                train_date_period[0].strftime("%Y-%m-%d"),
                train_date_period[1].strftime("%Y-%m-%d"),
                test_date_period[0].strftime("%Y-%m-%d"),
                test_date_period[1].strftime("%Y-%m-%d"),
            )

            summary = BatchBacktestSummary(
                # Data specific
                train_start=self._as_datetime(train_df[datetime_col].min()),
                train_end=self._as_datetime(train_df[datetime_col].max()),
                test_start=self._as_datetime(test_df[datetime_col].min()),
                test_end=self._as_datetime(test_df[datetime_col].max()),
                n_train=len(train_df),
                n_test=len(test_df),
                # Clustering result
                model_num=model_num,
                clustering_result=clustering_result,
                train_asssignments=train_assign_df,
                test_assignments=cumulative_assignments.copy(),
            )

            if log_to_mlflow:
                self._log_batch_mlflow(summary)

            self._history.append(summary)
            yield summary

    def _log_parent_params(self) -> None:
        """Log experiment-level parameters to the active MLflow run."""
        params = {
            "refit_interval_weeks": self.refit_interval_weeks,
            "window_size_days": self.feeder.window_size_days,
            "forward_window_days": self.feeder.forward_window_days,
        }
        mlflow.log_params({k: str(v) for k, v in params.items()})

        for key, value in self.config.hdbscan_params.items():
            mlflow.log_param(f"hdbscan__{key}", str(value))
        for key, value in self.config.umap_params.items():
            mlflow.log_param(f"umap__{key}", str(value))
        for key, value in self.config.knn_params.items():
            mlflow.log_param(f"knn__{key}", str(value))

    def _log_batch_mlflow(self, summary: BatchBacktestSummary) -> None:
        """Create a nested MLflow run for a single batch."""
        run_name = f"batch_{summary.test_start:%Y%m%d}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params({
                "train_start": summary.train_start.isoformat(),
                "train_end": summary.train_end.isoformat(),
                "test_start": summary.test_start.isoformat(),
                "test_end": summary.test_end.isoformat(),
            })
            mlflow.log_metrics({
                "n_train": float(summary.n_train),
                "n_test": float(summary.n_test),
                "noise_ratio": (
                    summary.clustering_result.noise_ratio
                    if summary.clustering_result
                    else np.nan
                ),
            })

            if self._engine and self._engine.is_fitted:
                try:
                    metrics = self._engine.get_clustering_metrics()
                    self._log_numeric_metrics(metrics, prefix="engine_")
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Failed to log clustering metrics: %s", exc)

            if summary.clustering_result is not None:
                self._log_clustering_result_metrics(summary.clustering_result)

    @staticmethod
    def _log_clustering_result_metrics(result: ClusteringResult) -> None:
        """Log clustering metrics from the latest fit."""
        metrics = {"train_noise_ratio": result.noise_ratio}
        metrics.update(result.metrics or {})
        BacktestingExperiment._log_numeric_metrics(metrics, prefix="fit_")

    @staticmethod
    def _log_numeric_metrics(metrics: dict, prefix: str = "") -> None:
        """Log a dictionary of numeric metrics with optional prefix."""
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                numeric_metrics[f"{prefix}{key}"] = float(value)
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
