"""
Performance monitoring and drift detection for the MLflow Time Series Clustering Pipeline.

This module implements the PerformanceMonitor class that provides comprehensive
performance tracking, drift detection, health monitoring, and alerting capabilities.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from collections import deque
import warnings
from dataclasses import dataclass
from enum import Enum

from ..core.data_models import BatchResult, PerformanceReport
from ..core.config import PipelineConfig
from ..core.exceptions import PipelineError
from ..mlflow_integration.logging_utils import PipelineLogger


logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    timestamp: datetime
    level: AlertLevel
    component: str
    metric: str
    current_value: float
    threshold_value: float
    message: str
    context: Dict[str, Any]


@dataclass
class DriftDetectionResult:
    """Drift detection result data structure."""
    drift_detected: bool
    drift_score: float
    drift_type: str
    affected_metrics: List[str]
    timestamp: datetime
    context: Dict[str, Any]


class PerformanceMonitor:
    """
    Comprehensive performance monitor for the Time Series Clustering Pipeline.

    This class provides:
    - Real-time performance metric tracking
    - Data drift detection using multiple methods
    - Performance degradation alerts
    - Health monitoring and reporting
    - Automated alerting and notifications
    """

    def __init__(self, config: PipelineConfig, alert_callback: Optional[Callable[[PerformanceAlert], None]] = None):
        """
        Initialize the performance monitor.

        Args:
            config: Pipeline configuration
            alert_callback: Optional callback function for alerts
        """
        self.config = config
        self.alert_callback = alert_callback
        self.logger = PipelineLogger("PerformanceMonitor")

        # Performance tracking
        self.metrics_history: Dict[str, deque] = {}
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.alert_history: List[PerformanceAlert] = []

        # Drift detection
        self.drift_detection_window = 100  # Number of recent samples for drift detection
        self.drift_thresholds = {
            'noise_ratio': 0.1,  # 10% change in noise ratio
            'processing_time': 0.3,  # 30% change in processing time
            'prediction_accuracy': 0.15,  # 15% change in accuracy
            'cluster_stability': 0.2  # 20% change in cluster assignments
        }

        # Alert thresholds
        self.alert_thresholds = {
            'noise_ratio_warning': config.noise_threshold * 0.8,
            'noise_ratio_critical': config.noise_threshold,
            'processing_time_warning': 5.0,  # seconds
            'processing_time_critical': 10.0,  # seconds
            'prediction_accuracy_warning': 0.7,  # R² score
            'prediction_accuracy_critical': 0.5,  # R² score
            'error_rate_warning': 0.05,  # 5% error rate
            'error_rate_critical': 0.1  # 10% error rate
        }

        # Health monitoring
        self.health_check_interval = timedelta(minutes=5)
        self.last_health_check = datetime.now()
        self.health_status = "healthy"
        self.consecutive_unhealthy_checks = 0

        # Performance statistics
        self.performance_stats = {
            'total_batches_monitored': 0,
            'total_alerts_generated': 0,
            'drift_detections': 0,
            'health_checks_performed': 0,
            'last_performance_report': None
        }

        self.logger.info(f"PerformanceMonitor initialized with drift detection window: {self.drift_detection_window}")

    def track_batch_performance(self, batch_result: BatchResult,
                              additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Track performance metrics from batch processing.

        Args:
            batch_result: Result from batch processing
            additional_metrics: Optional additional metrics to track
        """
        try:
            # Extract core metrics from batch result
            metrics = {
                'noise_ratio': batch_result.noise_ratio,
                'processing_time': batch_result.processing_time,
                'batch_size': len(batch_result.cluster_assignments),
                'unique_clusters': len(np.unique(batch_result.cluster_assignments[batch_result.cluster_assignments != -1])),
                'resource_predictions_count': len(batch_result.resource_predictions)
            }

            # Add additional metrics if provided
            if additional_metrics:
                metrics.update(additional_metrics)

            # Store metrics in history
            for metric_name, metric_value in metrics.items():
                if metric_name not in self.metrics_history:
                    self.metrics_history[metric_name] = deque(maxlen=1000)  # Keep last 1000 values

                self.metrics_history[metric_name].append({
                    'timestamp': datetime.now(),
                    'value': metric_value,
                    'batch_id': self.performance_stats['total_batches_monitored']
                })

            # Update performance statistics
            self.performance_stats['total_batches_monitored'] += 1

            # Check for alerts
            self._check_performance_alerts(metrics)

            # Check for drift
            drift_result = self._detect_drift(metrics)
            if drift_result.drift_detected:
                self._handle_drift_detection(drift_result)

            self.logger.debug("Batch performance tracked: noise_ratio=%.3f, processing_time=%.3f",
                            metrics['noise_ratio'], metrics['processing_time'])

        except Exception as e:
            self.logger.error(f"Failed to track batch performance: {str(e)}")

    def track_model_performance(self, model_metrics: Dict[str, Dict[str, float]],
                              model_type: str = "timeseries") -> None:
        """
        Track model performance metrics.

        Args:
            model_metrics: Dictionary mapping cluster_id to performance metrics
            model_type: Type of model (timeseries, resource, etc.)
        """
        try:
            # Aggregate metrics across clusters
            aggregated_metrics = {}

            if model_metrics:
                # Calculate mean metrics across clusters
                all_cluster_metrics = list(model_metrics.values())
                if all_cluster_metrics:
                    metric_names = all_cluster_metrics[0].keys()

                    for metric_name in metric_names:
                        values = [cluster_metrics.get(metric_name, 0) for cluster_metrics in all_cluster_metrics]
                        aggregated_metrics[f"{model_type}_{metric_name}_mean"] = np.mean(values)
                        aggregated_metrics[f"{model_type}_{metric_name}_std"] = np.std(values)
                        aggregated_metrics[f"{model_type}_{metric_name}_min"] = np.min(values)
                        aggregated_metrics[f"{model_type}_{metric_name}_max"] = np.max(values)

                # Add cluster count
                aggregated_metrics[f"{model_type}_cluster_count"] = len(model_metrics)

            # Store aggregated metrics
            for metric_name, metric_value in aggregated_metrics.items():
                if metric_name not in self.metrics_history:
                    self.metrics_history[metric_name] = deque(maxlen=1000)

                self.metrics_history[metric_name].append({
                    'timestamp': datetime.now(),
                    'value': metric_value,
                    'batch_id': self.performance_stats['total_batches_monitored']
                })

            # Check for model performance alerts
            self._check_model_performance_alerts(aggregated_metrics, model_type)

            self.logger.debug(f"Model performance tracked for {model_type}: {len(model_metrics)} clusters")

        except Exception as e:
            self.logger.error(f"Failed to track model performance: {str(e)}")

    def _check_performance_alerts(self, metrics: Dict[str, float]) -> None:
        """Check for performance alerts based on current metrics."""
        try:
            current_time = datetime.now()

            # Check noise ratio alerts
            if 'noise_ratio' in metrics:
                noise_ratio = metrics['noise_ratio']

                if noise_ratio >= self.alert_thresholds['noise_ratio_critical']:
                    alert = PerformanceAlert(
                        timestamp=current_time,
                        level=AlertLevel.CRITICAL,
                        component="clustering",
                        metric="noise_ratio",
                        current_value=noise_ratio,
                        threshold_value=self.alert_thresholds['noise_ratio_critical'],
                        message=f"Critical noise ratio detected: {noise_ratio:.3f}",
                        context=metrics
                    )
                    self._trigger_alert(alert)

                elif noise_ratio >= self.alert_thresholds['noise_ratio_warning']:
                    alert = PerformanceAlert(
                        timestamp=current_time,
                        level=AlertLevel.WARNING,
                        component="clustering",
                        metric="noise_ratio",
                        current_value=noise_ratio,
                        threshold_value=self.alert_thresholds['noise_ratio_warning'],
                        message=f"High noise ratio detected: {noise_ratio:.3f}",
                        context=metrics
                    )
                    self._trigger_alert(alert)

            # Check processing time alerts
            if 'processing_time' in metrics:
                processing_time = metrics['processing_time']

                if processing_time >= self.alert_thresholds['processing_time_critical']:
                    alert = PerformanceAlert(
                        timestamp=current_time,
                        level=AlertLevel.CRITICAL,
                        component="batch_processing",
                        metric="processing_time",
                        current_value=processing_time,
                        threshold_value=self.alert_thresholds['processing_time_critical'],
                        message=f"Critical processing time detected: {processing_time:.3f}s",
                        context=metrics
                    )
                    self._trigger_alert(alert)

                elif processing_time >= self.alert_thresholds['processing_time_warning']:
                    alert = PerformanceAlert(
                        timestamp=current_time,
                        level=AlertLevel.WARNING,
                        component="batch_processing",
                        metric="processing_time",
                        current_value=processing_time,
                        threshold_value=self.alert_thresholds['processing_time_warning'],
                        message=f"Slow processing time detected: {processing_time:.3f}s",
                        context=metrics
                    )
                    self._trigger_alert(alert)

        except Exception as e:
            self.logger.error(f"Failed to check performance alerts: {str(e)}")

    def _check_model_performance_alerts(self, metrics: Dict[str, float], model_type: str) -> None:
        """Check for model performance alerts."""
        try:
            current_time = datetime.now()

            # Check R² score for timeseries models
            if model_type == "timeseries" and 'timeseries_r2_score_mean' in metrics:
                r2_score = metrics['timeseries_r2_score_mean']

                if r2_score <= self.alert_thresholds['prediction_accuracy_critical']:
                    alert = PerformanceAlert(
                        timestamp=current_time,
                        level=AlertLevel.CRITICAL,
                        component="timeseries_models",
                        metric="r2_score",
                        current_value=r2_score,
                        threshold_value=self.alert_thresholds['prediction_accuracy_critical'],
                        message=f"Critical prediction accuracy: R²={r2_score:.3f}",
                        context=metrics
                    )
                    self._trigger_alert(alert)

                elif r2_score <= self.alert_thresholds['prediction_accuracy_warning']:
                    alert = PerformanceAlert(
                        timestamp=current_time,
                        level=AlertLevel.WARNING,
                        component="timeseries_models",
                        metric="r2_score",
                        current_value=r2_score,
                        threshold_value=self.alert_thresholds['prediction_accuracy_warning'],
                        message=f"Low prediction accuracy: R²={r2_score:.3f}",
                        context=metrics
                    )
                    self._trigger_alert(alert)

        except Exception as e:
            self.logger.error(f"Failed to check model performance alerts: {str(e)}")

    def _detect_drift(self, current_metrics: Dict[str, float]) -> DriftDetectionResult:
        """
        Detect data drift using statistical methods.

        Args:
            current_metrics: Current batch metrics

        Returns:
            DriftDetectionResult containing drift detection information
        """
        try:
            drift_detected = False
            drift_score = 0.0
            drift_type = "none"
            affected_metrics = []

            # Check each metric for drift
            for metric_name, threshold in self.drift_thresholds.items():
                if metric_name in current_metrics and metric_name in self.metrics_history:

                    # Get recent history for this metric
                    history = list(self.metrics_history[metric_name])
                    if len(history) < self.drift_detection_window:
                        continue  # Not enough history for drift detection

                    # Get recent values
                    recent_values = [entry['value'] for entry in history[-self.drift_detection_window:]]

                    # Split into two windows for comparison
                    window_size = len(recent_values) // 2
                    if window_size < 10:  # Need at least 10 samples per window
                        continue

                    old_window = recent_values[:window_size]
                    new_window = recent_values[window_size:]

                    # Calculate statistical difference
                    old_mean = np.mean(old_window)
                    new_mean = np.mean(new_window)

                    if old_mean != 0:
                        relative_change = abs(new_mean - old_mean) / abs(old_mean)

                        if relative_change > threshold:
                            drift_detected = True
                            drift_score = max(drift_score, relative_change)
                            affected_metrics.append(metric_name)

                            if metric_name == 'noise_ratio':
                                drift_type = "data_quality_drift"
                            elif metric_name == 'processing_time':
                                drift_type = "performance_drift"
                            elif metric_name == 'prediction_accuracy':
                                drift_type = "model_drift"
                            else:
                                drift_type = "general_drift"

            result = DriftDetectionResult(
                drift_detected=drift_detected,
                drift_score=drift_score,
                drift_type=drift_type,
                affected_metrics=affected_metrics,
                timestamp=datetime.now(),
                context=current_metrics
            )

            return result

        except Exception as e:
            self.logger.error(f"Failed to detect drift: {str(e)}")
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type="error",
                affected_metrics=[],
                timestamp=datetime.now(),
                context={"error": str(e)}
            )

    def _handle_drift_detection(self, drift_result: DriftDetectionResult) -> None:
        """Handle detected drift with appropriate actions."""
        try:
            self.performance_stats['drift_detections'] += 1

            # Create drift alert
            alert = PerformanceAlert(
                timestamp=drift_result.timestamp,
                level=AlertLevel.WARNING if drift_result.drift_score < 0.5 else AlertLevel.CRITICAL,
                component="drift_detection",
                metric="drift_score",
                current_value=drift_result.drift_score,
                threshold_value=max(self.drift_thresholds.values()),
                message=f"Data drift detected: {drift_result.drift_type} (score: {drift_result.drift_score:.3f})",
                context={
                    'drift_type': drift_result.drift_type,
                    'affected_metrics': drift_result.affected_metrics,
                    'drift_score': drift_result.drift_score
                }
            )

            self._trigger_alert(alert)

            self.logger.warning(f"Data drift detected: type={drift_result.drift_type}, score={drift_result.drift_score:.3f}, affected_metrics={drift_result.affected_metrics}")

        except Exception as e:
            self.logger.error(f"Failed to handle drift detection: {str(e)}")

    def _trigger_alert(self, alert: PerformanceAlert) -> None:
        """Trigger a performance alert."""
        try:
            # Store alert in history
            self.alert_history.append(alert)
            self.performance_stats['total_alerts_generated'] += 1

            # Keep only recent alerts to prevent memory issues
            max_alert_history = 1000
            if len(self.alert_history) > max_alert_history:
                self.alert_history = self.alert_history[-max_alert_history:]

            # Log alert
            log_level = logging.WARNING if alert.level == AlertLevel.WARNING else logging.ERROR
            alert_message = f"ALERT [{alert.level.value.upper()}] {alert.component}: {alert.message} (current: {alert.current_value:.3f}, threshold: {alert.threshold_value:.3f})"
            self.logger.logger.log(log_level, alert_message)

            # Call alert callback if provided
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as callback_error:
                    self.logger.error(f"Alert callback failed: {str(callback_error)}")

        except Exception as e:
            self.logger.error(f"Failed to trigger alert: {str(e)}")

    def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive pipeline health check.

        Returns:
            Dictionary containing health check results
        """
        try:
            current_time = datetime.now()
            self.performance_stats['health_checks_performed'] += 1

            health_report = {
                'timestamp': current_time.isoformat(),
                'overall_health': 'healthy',
                'component_health': {},
                'performance_summary': {},
                'recent_alerts': [],
                'drift_status': {},
                'recommendations': []
            }

            # Check recent performance metrics
            if self.metrics_history:
                health_report['performance_summary'] = self._generate_performance_summary()

            # Check recent alerts
            recent_alerts = [
                alert for alert in self.alert_history
                if (current_time - alert.timestamp).total_seconds() < 3600  # Last hour
            ]

            health_report['recent_alerts'] = [
                {
                    'level': alert.level.value,
                    'component': alert.component,
                    'metric': alert.metric,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in recent_alerts
            ]

            # Determine overall health
            critical_alerts = [alert for alert in recent_alerts if alert.level == AlertLevel.CRITICAL]
            warning_alerts = [alert for alert in recent_alerts if alert.level == AlertLevel.WARNING]

            if critical_alerts:
                health_report['overall_health'] = 'critical'
                self.health_status = 'critical'
                self.consecutive_unhealthy_checks += 1
            elif warning_alerts:
                health_report['overall_health'] = 'warning'
                self.health_status = 'warning'
                self.consecutive_unhealthy_checks += 1
            else:
                health_report['overall_health'] = 'healthy'
                self.health_status = 'healthy'
                self.consecutive_unhealthy_checks = 0

            # Generate recommendations
            health_report['recommendations'] = self._generate_health_recommendations(health_report)

            # Update last health check time
            self.last_health_check = current_time

            self.logger.info(f"Health check completed: status={health_report['overall_health']}, alerts={len(recent_alerts)}")

            return health_report

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'error',
                'error': str(e)
            }

    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from recent metrics."""
        summary = {}

        try:
            # Get recent metrics (last 100 entries)
            recent_window = 100

            for metric_name, history in self.metrics_history.items():
                if len(history) > 0:
                    recent_values = [entry['value'] for entry in list(history)[-recent_window:]]

                    summary[metric_name] = {
                        'current': recent_values[-1] if recent_values else 0,
                        'mean': np.mean(recent_values),
                        'std': np.std(recent_values),
                        'min': np.min(recent_values),
                        'max': np.max(recent_values),
                        'trend': self._calculate_trend(recent_values)
                    }

        except Exception as e:
            self.logger.error(f"Failed to generate performance summary: {str(e)}")
            summary['error'] = str(e)

        return summary

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "insufficient_data"

        try:
            # Simple linear trend calculation
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]

            if abs(slope) < 0.001:  # Threshold for stable
                return "stable"
            elif slope > 0:
                return "increasing"
            else:
                return "decreasing"

        except Exception:
            return "unknown"

    def _generate_health_recommendations(self, health_report: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on current status."""
        recommendations = []

        try:
            # Check for critical alerts
            critical_alerts = [alert for alert in health_report['recent_alerts']
                             if alert['level'] == 'critical']

            if critical_alerts:
                recommendations.append("Critical alerts detected - immediate attention required")

                # Specific recommendations based on alert types
                noise_alerts = [alert for alert in critical_alerts if 'noise' in alert['metric']]
                if noise_alerts:
                    recommendations.append("High noise ratio detected - consider retraining models")

                performance_alerts = [alert for alert in critical_alerts if 'processing_time' in alert['metric']]
                if performance_alerts:
                    recommendations.append("Performance degradation detected - check system resources")

            # Check for consecutive unhealthy checks
            if self.consecutive_unhealthy_checks > 5:
                recommendations.append("Persistent health issues detected - investigate root causes")

            # Check performance trends
            performance_summary = health_report.get('performance_summary', {})
            for metric_name, metric_data in performance_summary.items():
                if isinstance(metric_data, dict) and metric_data.get('trend') == 'increasing':
                    if 'noise_ratio' in metric_name:
                        recommendations.append("Increasing noise ratio trend - monitor data quality")
                    elif 'processing_time' in metric_name:
                        recommendations.append("Increasing processing time trend - optimize performance")

            # Default recommendation if no issues
            if not recommendations:
                recommendations.append("Pipeline health is good - continue monitoring")

        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")

        return recommendations

    def get_performance_metrics(self, metric_names: Optional[List[str]] = None,
                              time_window: Optional[timedelta] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get performance metrics history.

        Args:
            metric_names: Optional list of specific metrics to retrieve
            time_window: Optional time window to filter metrics

        Returns:
            Dictionary mapping metric names to their history
        """
        try:
            result = {}
            current_time = datetime.now()

            # Determine which metrics to include
            metrics_to_include = metric_names if metric_names else list(self.metrics_history.keys())

            for metric_name in metrics_to_include:
                if metric_name in self.metrics_history:
                    history = list(self.metrics_history[metric_name])

                    # Filter by time window if specified
                    if time_window:
                        cutoff_time = current_time - time_window
                        history = [
                            entry for entry in history
                            if entry['timestamp'] >= cutoff_time
                        ]

                    result[metric_name] = history

            return result

        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {str(e)}")
            return {}

    def get_alert_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get alert summary for a specified time window.

        Args:
            time_window: Time window to filter alerts (default: last 24 hours)

        Returns:
            Dictionary containing alert summary
        """
        try:
            if time_window is None:
                time_window = timedelta(hours=24)

            current_time = datetime.now()
            cutoff_time = current_time - time_window

            # Filter alerts by time window
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ]

            # Categorize alerts
            alert_summary = {
                'total_alerts': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL]),
                'warning_alerts': len([a for a in recent_alerts if a.level == AlertLevel.WARNING]),
                'info_alerts': len([a for a in recent_alerts if a.level == AlertLevel.INFO]),
                'alerts_by_component': {},
                'alerts_by_metric': {},
                'time_window_hours': time_window.total_seconds() / 3600
            }

            # Group by component
            for alert in recent_alerts:
                component = alert.component
                if component not in alert_summary['alerts_by_component']:
                    alert_summary['alerts_by_component'][component] = 0
                alert_summary['alerts_by_component'][component] += 1

            # Group by metric
            for alert in recent_alerts:
                metric = alert.metric
                if metric not in alert_summary['alerts_by_metric']:
                    alert_summary['alerts_by_metric'][metric] = 0
                alert_summary['alerts_by_metric'][metric] += 1

            return alert_summary

        except Exception as e:
            self.logger.error(f"Failed to get alert summary: {str(e)}")
            return {'error': str(e)}

    def update_alert_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update alert thresholds.

        Args:
            new_thresholds: Dictionary of new threshold values
        """
        try:
            old_thresholds = self.alert_thresholds.copy()
            self.alert_thresholds.update(new_thresholds)

            self.logger.info(f"Alert thresholds updated: {new_thresholds}")

            # Log changes
            for threshold_name, new_value in new_thresholds.items():
                old_value = old_thresholds.get(threshold_name, "not_set")
                self.logger.debug(f"Threshold {threshold_name}: {old_value} -> {new_value}")

        except Exception as e:
            self.logger.error(f"Failed to update alert thresholds: {str(e)}")

    def update_drift_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update drift detection thresholds.

        Args:
            new_thresholds: Dictionary of new drift threshold values
        """
        try:
            old_thresholds = self.drift_thresholds.copy()
            self.drift_thresholds.update(new_thresholds)

            self.logger.info(f"Drift thresholds updated: {new_thresholds}")

            # Log changes
            for threshold_name, new_value in new_thresholds.items():
                old_value = old_thresholds.get(threshold_name, "not_set")
                self.logger.debug(f"Drift threshold {threshold_name}: {old_value} -> {new_value}")

        except Exception as e:
            self.logger.error(f"Failed to update drift thresholds: {str(e)}")

    def reset_monitoring_state(self) -> None:
        """Reset monitoring state (useful for testing or maintenance)."""
        try:
            self.metrics_history.clear()
            self.alert_history.clear()
            self.performance_baselines.clear()

            self.performance_stats = {
                'total_batches_monitored': 0,
                'total_alerts_generated': 0,
                'drift_detections': 0,
                'health_checks_performed': 0,
                'last_performance_report': None
            }

            self.health_status = "healthy"
            self.consecutive_unhealthy_checks = 0
            self.last_health_check = datetime.now()

            self.logger.info("Performance monitoring state reset")

        except Exception as e:
            self.logger.error(f"Failed to reset monitoring state: {str(e)}")

    def export_monitoring_data(self, file_path: str) -> None:
        """
        Export monitoring data to file for analysis.

        Args:
            file_path: Path to save the monitoring data
        """
        try:
            import json

            export_data = {
                'timestamp': datetime.now().isoformat(),
                'performance_stats': self.performance_stats,
                'alert_thresholds': self.alert_thresholds,
                'drift_thresholds': self.drift_thresholds,
                'health_status': self.health_status,
                'metrics_history': {
                    name: [
                        {
                            'timestamp': entry['timestamp'].isoformat(),
                            'value': entry['value'],
                            'batch_id': entry['batch_id']
                        }
                        for entry in list(history)
                    ]
                    for name, history in self.metrics_history.items()
                },
                'alert_history': [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'level': alert.level.value,
                        'component': alert.component,
                        'metric': alert.metric,
                        'current_value': alert.current_value,
                        'threshold_value': alert.threshold_value,
                        'message': alert.message,
                        'context': alert.context
                    }
                    for alert in self.alert_history
                ]
            }

            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"Monitoring data exported to: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to export monitoring data: {str(e)}")

    @property
    def current_health_status(self) -> str:
        """Get current health status."""
        return self.health_status

    @property
    def monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return self.performance_stats.copy()

    def should_perform_health_check(self) -> bool:
        """Check if it's time to perform a health check."""
        return (datetime.now() - self.last_health_check) >= self.health_check_interval