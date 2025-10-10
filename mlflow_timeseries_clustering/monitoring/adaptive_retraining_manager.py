"""
Adaptive Retraining Manager for the MLflow Time Series Clustering Pipeline.

This module implements the AdaptiveRetrainingManager class that integrates
all retraining components including trigger monitoring, expanded window retraining,
and comprehensive reporting with MLflow integration.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import numpy as np

from ..core.config import PipelineConfig
from ..core.data_models import BatchResult
from ..core.exceptions import PipelineError, ModelTrainingError
from ..clustering.adaptive_clustering_engine import AdaptiveClusteringEngine
from ..models.cluster_specific_model_manager import ClusterSpecificModelManager
from ..mlflow_integration.experiment_manager import ExperimentManager
from ..mlflow_integration.artifact_manager import ArtifactManager
from .retraining_trigger import RetrainingTrigger, TriggerReason, RetrainingEvent
from .expanded_window_retrainer import ExpandedWindowRetrainer, RetrainingResult
from .retraining_reporter import RetrainingReporter


logger = logging.getLogger(__name__)


class AdaptiveRetrainingManager:
    """
    Manages the complete adaptive retraining lifecycle for the clustering pipeline.

    This class integrates trigger monitoring, expanded window retraining, and
    comprehensive reporting with MLflow tracking. It provides both automatic
    and manual retraining capabilities with full performance tracking.
    """

    def __init__(self,
                 config: PipelineConfig,
                 experiment_manager: ExperimentManager,
                 artifact_manager: ArtifactManager):
        """
        Initialize the adaptive retraining manager.

        Args:
            config: Pipeline configuration
            experiment_manager: MLflow experiment manager
            artifact_manager: MLflow artifact manager
        """
        self.config = config
        self.experiment_manager = experiment_manager
        self.artifact_manager = artifact_manager

        # Initialize retraining components
        self.trigger = RetrainingTrigger(config)
        self.retrainer = ExpandedWindowRetrainer(config)
        self.reporter = RetrainingReporter()

        # Set up retraining callback
        self.trigger.set_retraining_callback(self._handle_retraining_trigger)

        # Current models (to be set by pipeline)
        self._clustering_engine: Optional[AdaptiveClusteringEngine] = None
        self._model_manager: Optional[ClusterSpecificModelManager] = None

        # Retraining state
        self._retraining_in_progress = False
        self._last_retraining_result: Optional[RetrainingResult] = None

        # Callbacks for pipeline integration
        self._pre_retraining_callback: Optional[Callable[[], None]] = None
        self._post_retraining_callback: Optional[Callable[[RetrainingResult], None]] = None

        logger.info("AdaptiveRetrainingManager initialized")

    def set_current_models(self,
                          clustering_engine: AdaptiveClusteringEngine,
                          model_manager: Optional[ClusterSpecificModelManager] = None) -> None:
        """
        Set the current models that will be managed for retraining.

        Args:
            clustering_engine: Current clustering engine
            model_manager: Optional model manager for cluster-specific models
        """
        self._clustering_engine = clustering_engine
        self._model_manager = model_manager

        # Set models in retrainer
        self.retrainer.set_current_models(clustering_engine, model_manager)

        logger.info("Current models set in adaptive retraining manager")

    def set_retraining_callbacks(self,
                                pre_callback: Optional[Callable[[], None]] = None,
                                post_callback: Optional[Callable[[RetrainingResult], None]] = None) -> None:
        """
        Set callbacks for pre and post retraining events.

        Args:
            pre_callback: Called before retraining starts
            post_callback: Called after retraining completes
        """
        self._pre_retraining_callback = pre_callback
        self._post_retraining_callback = post_callback
        logger.info("Retraining callbacks configured")

    def monitor_batch_result(self, batch_result: BatchResult, batch_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Monitor batch processing result and trigger retraining if necessary.

        Args:
            batch_result: Result from batch processing
            batch_data: Optional batch data for historical storage

        Returns:
            True if retraining was triggered, False otherwise
        """
        # Add batch data to historical storage if provided
        if batch_data and 'X' in batch_data and 'y' in batch_data:
            self.retrainer.add_historical_data(batch_data['X'], batch_data['y'])

        # Monitor noise ratio
        return self.trigger.monitor_noise_ratio(
            batch_result.noise_ratio,
            {
                'processing_time': batch_result.processing_time,
                'cluster_assignments': len(np.unique(batch_result.cluster_assignments)),
                'timestamp': datetime.now()
            }
        )

    def request_manual_retraining(self, reason: str = "Manual request") -> bool:
        """
        Request manual retraining.

        Args:
            reason: Reason for manual retraining

        Returns:
            True if retraining was successfully initiated
        """
        logger.info("Manual retraining requested: %s", reason)

        return self.trigger.request_manual_retraining({
            'reason': reason,
            'timestamp': datetime.now(),
            'user_initiated': True
        })

    def check_scheduled_retraining(self, schedule_interval: Optional[timedelta] = None) -> bool:
        """
        Check if scheduled retraining should be triggered.

        Args:
            schedule_interval: Time interval for scheduled retraining

        Returns:
            True if retraining was triggered
        """
        if schedule_interval is None:
            # Default to 24 hours
            schedule_interval = timedelta(hours=24)

        return self.trigger.check_scheduled_retraining(schedule_interval)

    def _handle_retraining_trigger(self, trigger_event: RetrainingEvent) -> None:
        """
        Handle retraining trigger event.

        Args:
            trigger_event: The event that triggered retraining
        """
        if self._retraining_in_progress:
            logger.warning("Retraining already in progress, ignoring trigger")
            return

        logger.info("Handling retraining trigger: %s", trigger_event.reason.value)

        try:
            self._retraining_in_progress = True

            # Call pre-retraining callback
            if self._pre_retraining_callback:
                self._pre_retraining_callback()

            # Start MLflow run for retraining
            with self.experiment_manager.start_run(
                run_name=f"retraining_{trigger_event.reason.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                nested=True
            ) as run:
                # Log trigger information
                self._log_trigger_information(trigger_event)

                # Perform retraining
                retraining_result = self.retrainer.perform_retraining(trigger_event)

                # Log retraining results
                self._log_retraining_results(retraining_result)

                # Generate and save reports
                self._generate_and_save_reports(retraining_result)

                # Save updated models as artifacts
                self._save_retrained_models(retraining_result)

                # Store result
                self._last_retraining_result = retraining_result

                logger.info("Retraining completed successfully")

                # Call post-retraining callback
                if self._post_retraining_callback:
                    self._post_retraining_callback(retraining_result)

        except Exception as e:
            logger.error("Retraining failed: %s", str(e))
            # Log error to MLflow
            if hasattr(self.experiment_manager, 'log_param'):
                self.experiment_manager.log_param("retraining_error", str(e))
            raise

        finally:
            self._retraining_in_progress = False

    def _log_trigger_information(self, trigger_event: RetrainingEvent) -> None:
        """Log trigger information to MLflow."""
        try:
            # Log trigger parameters
            self.experiment_manager.log_param("trigger_reason", trigger_event.reason.value)
            self.experiment_manager.log_param("trigger_timestamp", trigger_event.timestamp.isoformat())

            if trigger_event.trigger_value is not None:
                self.experiment_manager.log_param("trigger_value", trigger_event.trigger_value)

            if trigger_event.threshold_value is not None:
                self.experiment_manager.log_param("threshold_value", trigger_event.threshold_value)

            # Log trigger metadata
            if trigger_event.metadata:
                for key, value in trigger_event.metadata.items():
                    self.experiment_manager.log_param(f"trigger_metadata_{key}", value)

            logger.debug("Trigger information logged to MLflow")

        except Exception as e:
            logger.warning("Failed to log trigger information: %s", str(e))

    def _log_retraining_results(self, retraining_result: RetrainingResult) -> None:
        """Log retraining results to MLflow."""
        try:
            # Log retraining metrics
            for key, value in retraining_result.retraining_metrics.items():
                if isinstance(value, (int, float)):
                    self.experiment_manager.log_metric(f"retraining_{key}", value)
                else:
                    self.experiment_manager.log_param(f"retraining_{key}", str(value))

            # Log clustering result metrics
            for key, value in retraining_result.clustering_result.metrics.items():
                if isinstance(value, (int, float)):
                    self.experiment_manager.log_metric(f"new_clustering_{key}", value)
                else:
                    self.experiment_manager.log_param(f"new_clustering_{key}", str(value))

            # Log model comparison metrics
            comparison = retraining_result.model_comparison
            if 'overall_improvement_ratio' in comparison:
                self.experiment_manager.log_metric("overall_improvement_ratio",
                                                 comparison['overall_improvement_ratio'])

            # Log improvement and degradation counts
            improvements = comparison.get('improvements', {})
            degradations = comparison.get('degradations', {})

            self.experiment_manager.log_metric("improvement_count", len(improvements))
            self.experiment_manager.log_metric("degradation_count", len(degradations))

            if improvements:
                self.experiment_manager.log_metric("improvement_magnitude", sum(improvements.values()))

            if degradations:
                self.experiment_manager.log_metric("degradation_magnitude", sum(degradations.values()))

            # Log reassignment summary
            reassignment = retraining_result.reassignment_summary
            for key, value in reassignment.items():
                if isinstance(value, (int, float)):
                    self.experiment_manager.log_metric(f"reassignment_{key}", value)

            logger.debug("Retraining results logged to MLflow")

        except Exception as e:
            logger.warning("Failed to log retraining results: %s", str(e))

    def _generate_and_save_reports(self, retraining_result: RetrainingResult) -> None:
        """Generate and save retraining reports."""
        try:
            # Generate comprehensive report
            report = self.reporter.generate_retraining_comparison_report(retraining_result)

            # Save report as artifact
            report_path = f"retraining_reports/retraining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.artifact_manager.save_json_artifact(report, report_path)

            # Generate visualizations
            viz_dir = f"retraining_visualizations/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            visualization_paths = self.reporter.create_retraining_visualizations(
                retraining_result, viz_dir
            )

            # Save visualizations as artifacts
            for viz_name, viz_path in visualization_paths.items():
                artifact_path = f"{viz_dir}/{viz_name}.png"
                self.artifact_manager.save_file_artifact(viz_path, artifact_path)

            logger.info("Retraining reports and visualizations saved")

        except Exception as e:
            logger.warning("Failed to generate and save reports: %s", str(e))

    def _save_retrained_models(self, retraining_result: RetrainingResult) -> None:
        """Save retrained models as artifacts."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save clustering engine
            if self._clustering_engine:
                clustering_path = f"retrained_models/{timestamp}/clustering_engine.joblib"
                self.artifact_manager.save_model_artifact(self._clustering_engine, clustering_path)

            # Save model manager if available
            if self._model_manager:
                model_manager_path = f"retrained_models/{timestamp}/model_manager.joblib"
                self.artifact_manager.save_model_artifact(self._model_manager, model_manager_path)

            # Save clustering result
            clustering_result_path = f"retrained_models/{timestamp}/clustering_result.joblib"
            self.artifact_manager.save_model_artifact(retraining_result.clustering_result, clustering_result_path)

            logger.info("Retrained models saved as artifacts")

        except Exception as e:
            logger.warning("Failed to save retrained models: %s", str(e))

    def get_retraining_status(self) -> Dict[str, Any]:
        """
        Get current retraining status and statistics.

        Returns:
            Dictionary containing retraining status information
        """
        trigger_stats = self.trigger.get_trigger_statistics()
        retraining_history = self.retrainer.get_retraining_history(limit=10)
        historical_data_summary = self.retrainer.get_historical_data_summary()

        status = {
            'retraining_in_progress': self._retraining_in_progress,
            'last_retraining': self._last_retraining_result.timestamp if self._last_retraining_result else None,
            'trigger_statistics': trigger_stats,
            'recent_retraining_history': [
                {
                    'timestamp': result.timestamp,
                    'trigger_reason': result.trigger_event.reason.value,
                    'improvement_ratio': result.model_comparison.get('overall_improvement_ratio', 0),
                    'cluster_count_change': result.retraining_metrics.get('new_cluster_count', 0) -
                                          result.retraining_metrics.get('old_cluster_count', 0)
                }
                for result in retraining_history
            ],
            'historical_data_summary': historical_data_summary,
            'configuration': {
                'noise_threshold': self.config.noise_threshold,
                'expanding_window_size': self.config.expanding_window_size,
                'experiment_name': self.config.experiment_name
            }
        }

        return status

    def get_retraining_recommendations(self) -> List[str]:
        """
        Get recommendations for retraining configuration optimization.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Get recent retraining history
        recent_results = self.retrainer.get_retraining_history(limit=5)

        if not recent_results:
            recommendations.append("No retraining history available for recommendations")
            return recommendations

        # Analyze recent performance
        recent_improvements = [
            result.model_comparison.get('overall_improvement_ratio', 0)
            for result in recent_results
        ]

        avg_improvement = np.mean(recent_improvements)

        if avg_improvement < 0.3:
            recommendations.append(
                "Recent retrainings show low improvement; consider adjusting trigger thresholds"
            )
        elif avg_improvement > 0.8:
            recommendations.append(
                "Recent retrainings are highly effective; current configuration is optimal"
            )

        # Analyze trigger frequency
        trigger_stats = self.trigger.get_trigger_statistics()
        noise_stats = trigger_stats.get('noise_ratio_statistics', {})

        if noise_stats.get('mean', 0) > self.config.noise_threshold * 0.8:
            recommendations.append(
                "Noise ratios are consistently high; consider optimizing HDBSCAN parameters"
            )

        # Analyze retraining frequency
        if len(recent_results) > 3:
            time_diffs = []
            for i in range(1, len(recent_results)):
                diff = (recent_results[i].timestamp - recent_results[i-1].timestamp).total_seconds() / 3600
                time_diffs.append(diff)

            avg_interval = np.mean(time_diffs)
            if avg_interval < 2:  # Less than 2 hours between retrainings
                recommendations.append(
                    "Frequent retraining detected; consider increasing noise threshold or minimum interval"
                )

        return recommendations

    def update_configuration(self, new_config: PipelineConfig) -> None:
        """
        Update retraining configuration.

        Args:
            new_config: New pipeline configuration
        """
        old_threshold = self.config.noise_threshold
        self.config = new_config

        # Update trigger configuration
        self.trigger.update_configuration(new_config)

        logger.info("Retraining configuration updated (noise threshold: %.3f -> %.3f)",
                   old_threshold, new_config.noise_threshold)

    def reset_retraining_state(self) -> None:
        """Reset retraining state (useful for testing or manual reset)."""
        logger.info("Resetting adaptive retraining manager state")

        self.trigger.reset_trigger_state()
        self._retraining_in_progress = False
        self._last_retraining_result = None

    @property
    def is_retraining_in_progress(self) -> bool:
        """Check if retraining is currently in progress."""
        return self._retraining_in_progress

    @property
    def last_retraining_result(self) -> Optional[RetrainingResult]:
        """Get the last retraining result."""
        return self._last_retraining_result