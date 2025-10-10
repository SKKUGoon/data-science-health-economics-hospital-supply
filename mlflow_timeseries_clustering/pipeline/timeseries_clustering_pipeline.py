"""
Main pipeline controller for the MLflow Time Series Clustering Pipeline.

This module implements the TimeSeriesClusteringPipeline class that orchestrates
the entire pipeline lifecycle including initial training, batch processing,
and adaptive retraining with comprehensive MLflow integration.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import warnings

from ..core.config import PipelineConfig
from ..core.data_models import ClusteringResult, BatchResult, PerformanceReport
from ..core.exceptions import (
    PipelineError, DataValidationError, ModelTrainingError,
    PredictionError, MLflowIntegrationError, RetrainingError
)
from ..clustering.adaptive_clustering_engine import AdaptiveClusteringEngine
from ..models.cluster_specific_model_manager import ClusterSpecificModelManager
from ..mlflow_integration.experiment_manager import ExperimentManager
from ..mlflow_integration.artifact_manager import ArtifactManager
from ..monitoring.adaptive_retraining_manager import AdaptiveRetrainingManager
from .batch_processor import BatchProcessor
from .error_handler import ErrorHandler, error_handler_decorator, PipelineErrorContext, create_graceful_batch_result
from .performance_monitor import PerformanceMonitor, PerformanceAlert


logger = logging.getLogger(__name__)


class TimeSeriesClusteringPipeline:
    """
    Main pipeline controller for MLflow Time Series Clustering.

    This class orchestrates the complete pipeline lifecycle:
    1. Initial training phase with HDBSCAN clustering and LightGBM models
    2. Streaming processing phase with batch processing and drift monitoring
    3. Adaptive retraining phase with automatic model updates

    All operations are fully integrated with MLflow for experiment tracking,
    model management, and performance monitoring.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the Time Series Clustering Pipeline.

        Args:
            config: Pipeline configuration containing all parameters

        Raises:
            PipelineError: If initialization fails
        """
        try:
            self.config = config
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

            # Initialize error handler
            self.error_handler = ErrorHandler(f"TimeSeriesClusteringPipeline_{config.hospital_profile_id or 'default'}")

            # Initialize MLflow components
            self.experiment_manager = ExperimentManager(config)
            self.artifact_manager = ArtifactManager(config)

            # Initialize core components (will be set during fit_initial)
            self.clustering_engine: Optional[AdaptiveClusteringEngine] = None
            self.model_manager: Optional[ClusterSpecificModelManager] = None
            self.batch_processor: Optional[BatchProcessor] = None
            self.retraining_manager: Optional[AdaptiveRetrainingManager] = None

            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(
                config=config,
                alert_callback=self._handle_performance_alert
            )

            # Pipeline state
            self._is_fitted = False
            self._initial_training_completed = False
            self._batch_processing_enabled = False

            # Training data storage for retraining
            self._training_data: Optional[Dict[str, np.ndarray]] = None
            self._model_version = "1.0.0"

            # Performance tracking
            self._performance_history: List[Dict[str, Any]] = []
            self._last_performance_report: Optional[PerformanceReport] = None

            # Callbacks for external integration
            self._pre_training_callback: Optional[Callable[[], None]] = None
            self._post_training_callback: Optional[Callable[[ClusteringResult], None]] = None
            self._pre_batch_callback: Optional[Callable[[np.ndarray], None]] = None
            self._post_batch_callback: Optional[Callable[[BatchResult], None]] = None

            self.logger.info("TimeSeriesClusteringPipeline initialized with experiment: %s",
                           config.experiment_name)

        except Exception as e:
            raise PipelineError(
                f"Failed to initialize pipeline: {str(e)}",
                component="PipelineController",
                details={"config": config.to_dict() if config else None}
            ) from e

    def fit_initial(self, X: np.ndarray, y: np.ndarray) -> ClusteringResult:
        """
        Perform initial training of the complete pipeline.

        This method:
        1. Fits HDBSCAN clustering with kNN fallback
        2. Trains cluster-specific LightGBM time series models
        3. Trains cluster-specific resource usage models
        4. Registers all models in MLflow model registry
        5. Generates comprehensive performance reports

        Args:
            X: Input feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            ClusteringResult containing clustering information

        Raises:
            DataValidationError: If input data is invalid
            ModelTrainingError: If model training fails
            MLflowIntegrationError: If MLflow operations fail
        """
        self.logger.info("Starting initial pipeline training with %d samples, %d features",
                        X.shape[0], X.shape[1])

        try:
            # Validate input data
            self._validate_training_data(X, y)

            # Store training data for potential retraining
            self._training_data = {'X': X.copy(), 'y': y.copy()}

            # Call pre-training callback
            if self._pre_training_callback:
                self._pre_training_callback()

            # Start parent MLflow run for initial training
            with self.experiment_manager.start_parent_run(
                run_name=f"initial_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={
                    "pipeline_phase": "initial_training",
                    "model_version": self._model_version,
                    "hospital_profile_id": self.config.hospital_profile_id or "default"
                }
            ) as parent_run:

                # Step 1: Train HDBSCAN clustering with kNN fallback
                clustering_result = self._train_clustering_models(X, y)

                # Step 2: Train cluster-specific time series models
                timeseries_metrics = self._train_timeseries_models(X, y, clustering_result.labels)

                # Step 3: Train resource usage models
                resource_metrics = self._train_resource_models(X, y, clustering_result.labels)

                # Step 4: Initialize batch processor and retraining manager
                self._initialize_processing_components()

                # Step 5: Register models in MLflow model registry
                self._register_pipeline_models()

                # Step 6: Generate and save performance reports
                performance_report = self._generate_initial_performance_report(
                    clustering_result, timeseries_metrics, resource_metrics
                )

                # Update pipeline state
                self._is_fitted = True
                self._initial_training_completed = True
                self._batch_processing_enabled = True
                self._last_performance_report = performance_report

                # Call post-training callback
                if self._post_training_callback:
                    self._post_training_callback(clustering_result)

                self.logger.info("Initial pipeline training completed successfully. "
                               "Found %d clusters with %.2f%% noise points",
                               clustering_result.metrics['n_clusters'],
                               clustering_result.noise_ratio * 100)

                return clustering_result

        except Exception as e:
            self.logger.error("Initial training failed: %s", str(e))
            if isinstance(e, (DataValidationError, ModelTrainingError, MLflowIntegrationError)):
                raise
            raise ModelTrainingError(
                f"Initial pipeline training failed: {str(e)}",
                model_type="pipeline",
                training_data_size=X.shape[0] if X is not None else 0
            ) from e

    def process_batch(self, X_batch: np.ndarray, y_batch: Optional[np.ndarray] = None) -> BatchResult:
        """
        Process a new data batch using the trained pipeline.

        This method:
        1. Validates and processes the batch data
        2. Assigns cluster labels with fallback handling
        3. Makes time series and resource predictions
        4. Monitors for data drift and triggers retraining if needed
        5. Logs all metrics to MLflow

        Args:
            X_batch: Input feature matrix for the batch
            y_batch: Optional target values for accuracy tracking

        Returns:
            BatchResult containing predictions and processing metrics

        Raises:
            PipelineError: If pipeline is not fitted
            DataValidationError: If input data is invalid
            PredictionError: If batch processing fails
        """
        if not self._is_fitted or not self._batch_processing_enabled:
            raise PipelineError(
                "Pipeline must be fitted and batch processing enabled before processing batches",
                component="PipelineController"
            )

        self.logger.info("Processing batch with %d samples", len(X_batch))

        with PipelineErrorContext(self.error_handler, "batch_processing") as ctx:
            ctx.add_context("batch_size", len(X_batch))
            ctx.add_context("has_targets", y_batch is not None)

            try:
                # Call pre-batch callback
                if self._pre_batch_callback:
                    self._pre_batch_callback(X_batch)

                # Process the batch
                batch_result = self.batch_processor.process_batch(X_batch, y_batch)

                # Monitor for retraining triggers
                retraining_triggered = self.retraining_manager.monitor_batch_result(
                    batch_result,
                    {'X': X_batch, 'y': y_batch} if y_batch is not None else {'X': X_batch}
                )

                if retraining_triggered:
                    self.logger.info("Retraining was triggered by batch processing")

                # Track performance metrics
                self.performance_monitor.track_batch_performance(batch_result)

                # Perform health check if needed
                if self.performance_monitor.should_perform_health_check():
                    health_report = self.performance_monitor.perform_health_check()
                    self.logger.debug("Health check performed: status=%s", health_report['overall_health'])

                # Call post-batch callback
                if self._post_batch_callback:
                    self._post_batch_callback(batch_result)

                self.logger.info("Batch processed successfully. Noise ratio: %.3f, Processing time: %.3f seconds",
                               batch_result.noise_ratio, batch_result.processing_time)

                return batch_result

            except Exception as e:
                self.logger.error("Batch processing failed: %s", str(e))

                # Handle error with graceful degradation
                error_result = self.error_handler.handle_error(
                    e,
                    context={
                        'batch_size': len(X_batch),
                        'has_targets': y_batch is not None,
                        'data': {'X': X_batch, 'y': y_batch}
                    },
                    enable_recovery=True
                )

                # If recovery was successful, return fallback result
                if error_result.get('recovery_result', {}).get('recovery_successful', False):
                    self.logger.warning("Using graceful degradation for batch processing")
                    return create_graceful_batch_result(e, len(X_batch))

                # Otherwise, raise the original error
                if isinstance(e, (DataValidationError, PredictionError)):
                    raise
                raise PredictionError(
                    f"Batch processing failed: {str(e)}",
                    prediction_type="batch_processing",
                    batch_size=len(X_batch) if X_batch is not None else 0
                ) from e

    def retrain(self, trigger_reason: str = "Manual request") -> None:
        """
        Manually trigger pipeline retraining.

        This method initiates the adaptive retraining process which:
        1. Refits HDBSCAN using expanded historical data
        2. Updates kNN fallback model
        3. Retrains cluster-specific models
        4. Compares performance with previous models
        5. Updates model registry with new versions

        Args:
            trigger_reason: Reason for triggering retraining

        Raises:
            PipelineError: If pipeline is not fitted
            RetrainingError: If retraining fails
        """
        if not self._is_fitted:
            raise PipelineError(
                "Pipeline must be fitted before retraining",
                component="PipelineController"
            )

        self.logger.info("Manual retraining requested: %s", trigger_reason)

        try:
            # Request manual retraining through the retraining manager
            retraining_initiated = self.retraining_manager.request_manual_retraining(trigger_reason)

            if not retraining_initiated:
                self.logger.warning("Retraining request was not initiated (possibly already in progress)")
                return

            # Wait for retraining to complete and get result
            # Note: The actual retraining is handled asynchronously by the retraining manager
            self.logger.info("Manual retraining initiated successfully")

        except Exception as e:
            self.logger.error("Manual retraining failed: %s", str(e))
            raise RetrainingError(
                f"Manual retraining failed: {str(e)}",
                trigger_reason=trigger_reason
            ) from e

    def get_performance_report(self) -> PerformanceReport:
        """
        Generate comprehensive performance report for the pipeline.

        Returns:
            PerformanceReport containing current pipeline performance metrics

        Raises:
            PipelineError: If pipeline is not fitted
        """
        if not self._is_fitted:
            raise PipelineError(
                "Pipeline must be fitted before generating performance report",
                component="PipelineController"
            )

        try:
            self.logger.info("Generating comprehensive performance report")

            # Collect clustering metrics
            clustering_metrics = self.clustering_engine.get_clustering_metrics()

            # Collect model metrics
            model_metrics = {}
            if self.model_manager:
                model_metrics['timeseries'] = self.model_manager.get_model_metrics("timeseries")
                model_metrics['resource'] = self.model_manager.get_model_metrics("resource")

            # Collect batch processing metrics
            batch_metrics = {}
            if self.batch_processor:
                batch_metrics = self.batch_processor.get_processing_summary()

            # Collect retraining metrics
            retraining_metrics = {}
            if self.retraining_manager:
                retraining_metrics = self.retraining_manager.get_retraining_status()

            # Generate visualizations
            visualizations = self._generate_performance_visualizations()

            # Create performance report
            performance_report = PerformanceReport(
                clustering_metrics=clustering_metrics,
                model_metrics={
                    'timeseries_models': model_metrics.get('timeseries', {}),
                    'resource_models': model_metrics.get('resource', {}),
                    'batch_processing': batch_metrics,
                    'retraining': retraining_metrics
                },
                visualizations=visualizations,
                timestamp=datetime.now(),
                model_version=self._model_version
            )

            # Store report
            self._last_performance_report = performance_report
            self._performance_history.append({
                'timestamp': performance_report.timestamp,
                'model_version': performance_report.model_version,
                'clustering_metrics': clustering_metrics,
                'model_metrics': model_metrics
            })

            # Save report as MLflow artifact
            self._save_performance_report(performance_report)

            self.logger.info("Performance report generated successfully")
            return performance_report

        except Exception as e:
            self.logger.error("Failed to generate performance report: %s", str(e))
            raise PipelineError(
                f"Failed to generate performance report: {str(e)}",
                component="PipelineController"
            ) from e

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate training data format and content.

        Args:
            X: Input feature matrix
            y: Target values

        Raises:
            DataValidationError: If validation fails
        """
        # Validate X
        if X is None:
            raise DataValidationError("Training features cannot be None")

        if not isinstance(X, np.ndarray):
            raise DataValidationError("Training features must be a numpy array", data_type=type(X).__name__)

        if X.ndim != 2:
            raise DataValidationError(
                "Training features must be 2-dimensional",
                expected_shape="(n_samples, n_features)",
                actual_shape=X.shape
            )

        if X.shape[0] == 0:
            raise DataValidationError("Training data cannot be empty", actual_shape=X.shape)

        if X.shape[1] == 0:
            raise DataValidationError("Training data must have at least one feature", actual_shape=X.shape)

        # Check for invalid values in X
        if np.any(np.isnan(X)):
            raise DataValidationError("Training features contain NaN values")

        if np.any(np.isinf(X)):
            raise DataValidationError("Training features contain infinite values")

        # Validate y
        if y is None:
            raise DataValidationError("Training targets cannot be None")

        if not isinstance(y, np.ndarray):
            raise DataValidationError("Training targets must be a numpy array", data_type=type(y).__name__)

        if y.ndim != 1:
            raise DataValidationError(
                "Training targets must be 1-dimensional",
                expected_shape="(n_samples,)",
                actual_shape=y.shape
            )

        if len(y) != X.shape[0]:
            raise DataValidationError(
                f"Training targets length ({len(y)}) must match number of samples ({X.shape[0]})"
            )

        # Check for invalid values in y
        if np.any(np.isnan(y)):
            raise DataValidationError("Training targets contain NaN values")

        if np.any(np.isinf(y)):
            raise DataValidationError("Training targets contain infinite values")

        # Check minimum data requirements
        min_samples = max(self.config.hdbscan_params.get('min_cluster_size', 5) * 3, 20)
        if X.shape[0] < min_samples:
            raise DataValidationError(
                f"Insufficient training data: need at least {min_samples} samples, got {X.shape[0]}"
            )

    def _train_clustering_models(self, X: np.ndarray, y: np.ndarray) -> ClusteringResult:
        """
        Train HDBSCAN clustering with kNN fallback.

        Args:
            X: Input feature matrix
            y: Target values

        Returns:
            ClusteringResult from clustering training

        Raises:
            ModelTrainingError: If clustering training fails
        """
        try:
            with self.experiment_manager.start_nested_run(
                run_name="hdbscan_clustering",
                component="clustering",
                tags={"model_type": "hdbscan_with_knn_fallback"}
            ):
                # Initialize clustering engine
                self.clustering_engine = AdaptiveClusteringEngine(
                    hdbscan_params=self.config.hdbscan_params,
                    knn_params=self.config.knn_params
                )

                # Fit clustering model
                clustering_result = self.clustering_engine.fit(X)

                # Log clustering parameters and metrics
                self.experiment_manager.log_pipeline_parameters(self.config.hdbscan_params)
                self.experiment_manager.log_pipeline_parameters(self.config.knn_params)
                self.experiment_manager.log_pipeline_metrics(clustering_result.metrics)

                # Save clustering models as artifacts
                self.artifact_manager.save_model_artifact(
                    self.clustering_engine,
                    "clustering/adaptive_clustering_engine.joblib"
                )

                self.logger.info("Clustering training completed: %d clusters, %.2f%% noise",
                               clustering_result.metrics['n_clusters'],
                               clustering_result.noise_ratio * 100)

                return clustering_result

        except Exception as e:
            raise ModelTrainingError(
                f"Clustering model training failed: {str(e)}",
                model_type="HDBSCAN",
                training_data_size=X.shape[0]
            ) from e

    def _train_timeseries_models(self, X: np.ndarray, y: np.ndarray,
                               cluster_labels: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Train cluster-specific LightGBM time series models.

        Args:
            X: Input feature matrix
            y: Target values
            cluster_labels: Cluster assignments

        Returns:
            Dictionary mapping cluster_id to performance metrics

        Raises:
            ModelTrainingError: If time series model training fails
        """
        try:
            with self.experiment_manager.start_nested_run(
                run_name="lightgbm_timeseries",
                component="timeseries_models",
                tags={"model_type": "lightgbm_timeseries"}
            ):
                # Initialize model manager
                self.model_manager = ClusterSpecificModelManager(
                    config=self.config,
                    artifact_manager=self.artifact_manager
                )

                # Prepare cluster data
                cluster_data = self._prepare_cluster_data(X, y, cluster_labels)

                # Train time series models
                timeseries_metrics = self.model_manager.fit_timeseries_models(cluster_data)

                # Log time series parameters and metrics
                self.experiment_manager.log_pipeline_parameters(self.config.lgb_timeseries_params)

                # Log metrics for each cluster
                for cluster_id, metrics in timeseries_metrics.items():
                    cluster_metrics = {f"cluster_{cluster_id}_{k}": v for k, v in metrics.items()}
                    self.experiment_manager.log_pipeline_metrics(cluster_metrics)

                # Save time series models as artifacts
                for cluster_id in timeseries_metrics.keys():
                    model_path = f"timeseries_models/cluster_{cluster_id}_model.joblib"
                    # Model saving is handled by the model manager

                self.logger.info("Time series models trained for %d clusters", len(timeseries_metrics))

                return timeseries_metrics

        except Exception as e:
            raise ModelTrainingError(
                f"Time series model training failed: {str(e)}",
                model_type="LightGBM_TimeSeries",
                training_data_size=X.shape[0]
            ) from e

    def _train_resource_models(self, X: np.ndarray, y: np.ndarray,
                             cluster_labels: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Train cluster-specific resource usage models.

        Args:
            X: Input feature matrix
            y: Target values
            cluster_labels: Cluster assignments

        Returns:
            Dictionary mapping cluster_id to resource metrics

        Raises:
            ModelTrainingError: If resource model training fails
        """
        try:
            with self.experiment_manager.start_nested_run(
                run_name="lightgbm_resource",
                component="resource_models",
                tags={"model_type": "lightgbm_resource"}
            ):
                # Prepare cluster data for resource modeling
                cluster_data = self._prepare_cluster_data(X, y, cluster_labels)
                resource_data = self._prepare_resource_data(cluster_data)

                # Train resource usage models
                resource_metrics = self.model_manager.fit_resource_models(cluster_data, resource_data)

                # Log resource parameters and metrics
                self.experiment_manager.log_pipeline_parameters(self.config.lgb_resource_params)

                # Log metrics for each cluster
                for cluster_id, metrics in resource_metrics.items():
                    cluster_metrics = {f"resource_cluster_{cluster_id}_{k}": v for k, v in metrics.items()}
                    self.experiment_manager.log_pipeline_metrics(cluster_metrics)

                self.logger.info("Resource usage models trained for %d clusters", len(resource_metrics))

                return resource_metrics

        except Exception as e:
            raise ModelTrainingError(
                f"Resource model training failed: {str(e)}",
                model_type="LightGBM_Resource",
                training_data_size=X.shape[0]
            ) from e

    def _prepare_cluster_data(self, X: np.ndarray, y: np.ndarray,
                            cluster_labels: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Prepare data organized by cluster for model training.

        Args:
            X: Input feature matrix
            y: Target values
            cluster_labels: Cluster assignments

        Returns:
            Dictionary mapping cluster_id to {'X': features, 'y': targets}
        """
        cluster_data = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_X = X[cluster_mask]
            cluster_y = y[cluster_mask]

            if len(cluster_X) > 0:
                cluster_data[cluster_id] = {
                    'X': cluster_X,
                    'y': cluster_y
                }

        return cluster_data

    def _prepare_resource_data(self, cluster_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Prepare synthetic resource usage data for each cluster.

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

    def _initialize_processing_components(self) -> None:
        """Initialize batch processor and retraining manager."""
        try:
            # Initialize batch processor
            self.batch_processor = BatchProcessor(
                config=self.config,
                clustering_engine=self.clustering_engine,
                model_manager=self.model_manager,
                experiment_manager=self.experiment_manager
            )

            # Initialize retraining manager
            self.retraining_manager = AdaptiveRetrainingManager(
                config=self.config,
                experiment_manager=self.experiment_manager,
                artifact_manager=self.artifact_manager
            )

            # Set current models in retraining manager
            self.retraining_manager.set_current_models(
                self.clustering_engine,
                self.model_manager
            )

            # Set retraining callbacks
            self.retraining_manager.set_retraining_callbacks(
                pre_callback=self._handle_pre_retraining,
                post_callback=self._handle_post_retraining
            )

            self.logger.info("Processing components initialized successfully")

        except Exception as e:
            raise PipelineError(
                f"Failed to initialize processing components: {str(e)}",
                component="PipelineController"
            ) from e

    def _register_pipeline_models(self) -> None:
        """Register all pipeline models in MLflow model registry."""
        try:
            model_components = {
                "clustering_engine": "clustering/adaptive_clustering_engine.joblib",
                "timeseries_models": "timeseries_models/",
                "resource_models": "resource_models/"
            }

            model_name = self.experiment_manager.register_pipeline_model(
                model_components=model_components,
                model_version=self._model_version,
                stage="Staging"
            )

            self.logger.info("Pipeline models registered in MLflow registry: %s", model_name)

        except Exception as e:
            # Log warning but don't fail the pipeline
            self.logger.warning("Failed to register pipeline models: %s", str(e))

    def _generate_initial_performance_report(self, clustering_result: ClusteringResult,
                                           timeseries_metrics: Dict[int, Dict[str, float]],
                                           resource_metrics: Dict[int, Dict[str, float]]) -> PerformanceReport:
        """Generate initial performance report after training."""
        try:
            # Collect all metrics
            all_model_metrics = {
                'timeseries_models': timeseries_metrics,
                'resource_models': resource_metrics,
                'training_summary': {
                    'total_clusters': len(timeseries_metrics),
                    'training_samples': self._training_data['X'].shape[0] if self._training_data else 0,
                    'training_features': self._training_data['X'].shape[1] if self._training_data else 0
                }
            }

            # Generate visualizations
            visualizations = self._generate_performance_visualizations()

            # Create performance report
            performance_report = PerformanceReport(
                clustering_metrics=clustering_result.metrics,
                model_metrics=all_model_metrics,
                visualizations=visualizations,
                timestamp=datetime.now(),
                model_version=self._model_version
            )

            return performance_report

        except Exception as e:
            self.logger.error("Failed to generate initial performance report: %s", str(e))
            # Return minimal report
            return PerformanceReport(
                clustering_metrics=clustering_result.metrics,
                model_metrics={'error': str(e)},
                visualizations={},
                timestamp=datetime.now(),
                model_version=self._model_version
            )

    def _generate_performance_visualizations(self) -> Dict[str, str]:
        """Generate performance visualizations."""
        visualizations = {}

        try:
            # Generate clustering visualizations
            if self.clustering_engine and self.clustering_engine.is_fitted:
                clustering_viz = self.clustering_engine.generate_knn_fallback_reports(
                    save_path="visualizations/clustering"
                )
                if 'usage_visualizations' in clustering_viz:
                    visualizations.update(clustering_viz['usage_visualizations'])

            # Generate model performance visualizations
            if self.model_manager:
                model_viz = self.model_manager.generate_performance_report()
                if 'visualizations' in model_viz:
                    visualizations.update(model_viz['visualizations'])

        except Exception as e:
            self.logger.warning("Failed to generate some visualizations: %s", str(e))

        return visualizations

    def _save_performance_report(self, performance_report: PerformanceReport) -> None:
        """Save performance report as MLflow artifact."""
        try:
            report_data = {
                'timestamp': performance_report.timestamp.isoformat(),
                'model_version': performance_report.model_version,
                'clustering_metrics': performance_report.clustering_metrics,
                'model_metrics': performance_report.model_metrics,
                'visualizations': performance_report.visualizations
            }

            report_path = f"performance_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.artifact_manager.save_json_artifact(report_data, report_path)

            self.logger.info("Performance report saved as artifact: %s", report_path)

        except Exception as e:
            self.logger.warning("Failed to save performance report: %s", str(e))

    def _handle_pre_retraining(self) -> None:
        """Handle pre-retraining callback."""
        self.logger.info("Pre-retraining callback: Preparing for model retraining")
        # Disable batch processing during retraining
        self._batch_processing_enabled = False

    def _handle_post_retraining(self, retraining_result) -> None:
        """Handle post-retraining callback."""
        self.logger.info("Post-retraining callback: Retraining completed")

        # Update model version
        self._model_version = f"{self._model_version.split('.')[0]}.{int(self._model_version.split('.')[1]) + 1}.0"

        # Re-enable batch processing
        self._batch_processing_enabled = True

        # Update batch processor with new models
        if self.batch_processor:
            self.batch_processor.clustering_engine = self.clustering_engine
            self.batch_processor.model_manager = self.model_manager

    # Properties and utility methods
    @property
    def is_fitted(self) -> bool:
        """Check if the pipeline is fitted and ready for use."""
        return self._is_fitted

    @property
    def is_batch_processing_enabled(self) -> bool:
        """Check if batch processing is currently enabled."""
        return self._batch_processing_enabled

    @property
    def current_model_version(self) -> str:
        """Get the current model version."""
        return self._model_version

    @property
    def last_performance_report(self) -> Optional[PerformanceReport]:
        """Get the last generated performance report."""
        return self._last_performance_report

    def set_callbacks(self,
                     pre_training: Optional[Callable[[], None]] = None,
                     post_training: Optional[Callable[[ClusteringResult], None]] = None,
                     pre_batch: Optional[Callable[[np.ndarray], None]] = None,
                     post_batch: Optional[Callable[[BatchResult], None]] = None) -> None:
        """
        Set callbacks for pipeline events.

        Args:
            pre_training: Called before initial training starts
            post_training: Called after initial training completes
            pre_batch: Called before each batch is processed
            post_batch: Called after each batch is processed
        """
        self._pre_training_callback = pre_training
        self._post_training_callback = post_training
        self._pre_batch_callback = pre_batch
        self._post_batch_callback = post_batch

        self.logger.info("Pipeline callbacks configured")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status information.

        Returns:
            Dictionary containing pipeline status and statistics
        """
        status = {
            'pipeline_fitted': self._is_fitted,
            'initial_training_completed': self._initial_training_completed,
            'batch_processing_enabled': self._batch_processing_enabled,
            'current_model_version': self._model_version,
            'experiment_name': self.config.experiment_name,
            'hospital_profile_id': self.config.hospital_profile_id
        }

        # Add component status
        if self.clustering_engine:
            status['clustering_engine_fitted'] = self.clustering_engine.is_fitted

        if self.batch_processor:
            status['batch_processing_summary'] = self.batch_processor.get_processing_summary()

        if self.retraining_manager:
            status['retraining_status'] = self.retraining_manager.get_retraining_status()

        # Add performance history
        status['performance_history_count'] = len(self._performance_history)
        if self._last_performance_report:
            status['last_performance_report_timestamp'] = self._last_performance_report.timestamp.isoformat()

        return status

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current models.

        Returns:
            Dictionary containing model information
        """
        model_info = {
            'model_version': self._model_version,
            'pipeline_fitted': self._is_fitted
        }

        if self.clustering_engine and self.clustering_engine.is_fitted:
            model_info['clustering'] = {
                'n_clusters': len(np.unique(self.clustering_engine.labels)) - (1 if -1 in self.clustering_engine.labels else 0),
                'noise_ratio': np.sum(self.clustering_engine.labels == -1) / len(self.clustering_engine.labels),
                'hdbscan_params': self.config.hdbscan_params,
                'knn_params': self.config.knn_params
            }

        if self.model_manager:
            model_info['timeseries_models'] = {
                'n_models': len(self.model_manager.timeseries_models),
                'cluster_ids': list(self.model_manager.timeseries_models.keys()),
                'lgb_params': self.config.lgb_timeseries_params
            }

            model_info['resource_models'] = {
                'n_models': len(self.model_manager.resource_models),
                'cluster_ids': list(self.model_manager.resource_models.keys()),
                'lgb_params': self.config.lgb_resource_params
            }

        return model_info

    def update_configuration(self, new_config: PipelineConfig) -> None:
        """
        Update pipeline configuration.

        Args:
            new_config: New pipeline configuration

        Note:
            This will update configuration for future operations but won't
            retrain existing models. Use retrain() to apply new parameters.
        """
        old_config = self.config
        self.config = new_config

        # Update retraining manager configuration if available
        if self.retraining_manager:
            self.retraining_manager.update_configuration(new_config)

        self.logger.info("Pipeline configuration updated. "
                        "Noise threshold: %.3f -> %.3f, "
                        "Batch size: %d -> %d",
                        old_config.noise_threshold, new_config.noise_threshold,
                        old_config.batch_size, new_config.batch_size)

    def reset_pipeline_state(self) -> None:
        """
        Reset pipeline state (useful for testing or reinitialization).

        Warning: This will clear all trained models and processing history.
        """
        self.logger.warning("Resetting pipeline state - all models and history will be cleared")

        self._is_fitted = False
        self._initial_training_completed = False
        self._batch_processing_enabled = False
        self._training_data = None
        self._model_version = "1.0.0"
        self._performance_history.clear()
        self._last_performance_report = None

        # Reset components
        self.clustering_engine = None
        self.model_manager = None
        self.batch_processor = None

        if self.retraining_manager:
            self.retraining_manager.reset_retraining_state()

        self.logger.info("Pipeline state reset completed")

    # Error handling and debugging methods
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error statistics from the error handler.

        Returns:
            Dictionary containing error statistics and recent errors
        """
        return self.error_handler.get_error_statistics()

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent error records for debugging.

        Args:
            limit: Maximum number of recent errors to return

        Returns:
            List of recent error records
        """
        return self.error_handler.get_recent_errors(limit)

    def enable_graceful_degradation(self, enable: bool = True) -> None:
        """
        Enable or disable graceful degradation for error handling.

        Args:
            enable: Whether to enable graceful degradation
        """
        self.error_handler.update_fallback_config({
            'enable_graceful_degradation': enable
        })
        self.logger.info("Graceful degradation %s", "enabled" if enable else "disabled")

    def set_error_recovery_config(self, config: Dict[str, Any]) -> None:
        """
        Update error recovery configuration.

        Args:
            config: Dictionary containing error recovery settings
        """
        self.error_handler.update_fallback_config(config)
        self.logger.info("Error recovery configuration updated: %s", config)

    def clear_error_history(self) -> None:
        """Clear error history for maintenance or testing."""
        self.error_handler.clear_error_history()
        self.logger.info("Error history cleared")

    def diagnose_pipeline_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive pipeline health diagnosis.

        Returns:
            Dictionary containing health diagnosis results
        """
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_status': self.get_pipeline_status(),
            'error_statistics': self.get_error_statistics(),
            'recent_errors': self.get_recent_errors(5),
            'component_health': {},
            'recommendations': []
        }

        # Check component health
        try:
            if self.clustering_engine:
                health_report['component_health']['clustering_engine'] = {
                    'fitted': self.clustering_engine.is_fitted,
                    'has_fallback': self.clustering_engine.knn_fallback is not None
                }

            if self.model_manager:
                health_report['component_health']['model_manager'] = {
                    'timeseries_models_count': len(self.model_manager.timeseries_models),
                    'resource_models_count': len(self.model_manager.resource_models)
                }

            if self.batch_processor:
                health_report['component_health']['batch_processor'] = {
                    'drift_detected': self.batch_processor.is_drift_detected,
                    'avg_processing_time': self.batch_processor.average_processing_time
                }

            if self.retraining_manager:
                health_report['component_health']['retraining_manager'] = {
                    'retraining_in_progress': self.retraining_manager.is_retraining_in_progress,
                    'last_retraining': self.retraining_manager.last_retraining_result.timestamp.isoformat() if self.retraining_manager.last_retraining_result else None
                }

        except Exception as e:
            health_report['component_health_error'] = str(e)

        # Generate recommendations
        error_stats = health_report['error_statistics']
        if error_stats['total_errors'] > 10:
            health_report['recommendations'].append("High error count detected - consider reviewing error patterns")

        if error_stats.get('recent_errors', 0) > 5:
            health_report['recommendations'].append("Recent error spike detected - investigate immediate causes")

        if not health_report['component_health'].get('clustering_engine', {}).get('fitted', False):
            health_report['recommendations'].append("Clustering engine not fitted - run initial training")

        return health_report

    def _handle_performance_alert(self, alert: PerformanceAlert) -> None:
        """
        Handle performance alerts from the performance monitor.

        Args:
            alert: Performance alert to handle
        """
        try:
            # Log the alert
            self.logger.warning("Performance alert received: [%s] %s - %s",
                              alert.level.value.upper(), alert.component, alert.message)

            # Take action based on alert type and severity
            if alert.level.value == "critical":
                if alert.metric == "noise_ratio" and alert.current_value >= self.config.noise_threshold:
                    # Trigger manual retraining for critical noise ratio
                    self.logger.warning("Critical noise ratio detected, triggering retraining")
                    try:
                        self.retrain(f"Critical noise ratio: {alert.current_value:.3f}")
                    except Exception as e:
                        self.logger.error("Failed to trigger retraining from alert: %s", str(e))

                elif alert.metric == "processing_time":
                    # Log performance degradation for investigation
                    self.logger.error("Critical processing time degradation detected: %.3f seconds",
                                    alert.current_value)

            # Store alert in pipeline history for reporting
            if not hasattr(self, '_performance_alerts'):
                self._performance_alerts = []
            self._performance_alerts.append(alert)

            # Keep only recent alerts
            max_alerts = 100
            if len(self._performance_alerts) > max_alerts:
                self._performance_alerts = self._performance_alerts[-max_alerts:]

        except Exception as e:
            self.logger.error("Failed to handle performance alert: %s", str(e))

    # Performance monitoring methods
    def get_performance_metrics(self, metric_names: Optional[List[str]] = None,
                              time_window: Optional[timedelta] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get performance metrics from the performance monitor.

        Args:
            metric_names: Optional list of specific metrics to retrieve
            time_window: Optional time window to filter metrics

        Returns:
            Dictionary mapping metric names to their history
        """
        return self.performance_monitor.get_performance_metrics(metric_names, time_window)

    def get_alert_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get alert summary from the performance monitor.

        Args:
            time_window: Time window to filter alerts

        Returns:
            Dictionary containing alert summary
        """
        return self.performance_monitor.get_alert_summary(time_window)

    def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive pipeline health check.

        Returns:
            Dictionary containing health check results
        """
        return self.performance_monitor.perform_health_check()

    def update_performance_thresholds(self, alert_thresholds: Optional[Dict[str, float]] = None,
                                    drift_thresholds: Optional[Dict[str, float]] = None) -> None:
        """
        Update performance monitoring thresholds.

        Args:
            alert_thresholds: Optional new alert thresholds
            drift_thresholds: Optional new drift detection thresholds
        """
        if alert_thresholds:
            self.performance_monitor.update_alert_thresholds(alert_thresholds)

        if drift_thresholds:
            self.performance_monitor.update_drift_thresholds(drift_thresholds)

        self.logger.info("Performance thresholds updated")

    def export_performance_data(self, file_path: str) -> None:
        """
        Export performance monitoring data to file.

        Args:
            file_path: Path to save the performance data
        """
        self.performance_monitor.export_monitoring_data(file_path)

    @property
    def current_health_status(self) -> str:
        """Get current pipeline health status."""
        return self.performance_monitor.current_health_status

    @property
    def monitoring_statistics(self) -> Dict[str, Any]:
        """Get performance monitoring statistics."""
        return self.performance_monitor.monitoring_statistics