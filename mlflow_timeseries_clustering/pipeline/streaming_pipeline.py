"""
Streaming prediction pipeline for the MLflow Time Series Clustering Pipeline.

This module implements the StreamingPipeline class that handles continuous
data processing with time series prediction, resource usage prediction,
and performance tracking.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, List, Callable, Generator
from datetime import datetime, timedelta
from collections import deque
import threading
import logging

from ..core.data_models import BatchResult
from ..core.config import PipelineConfig
from ..core.exceptions import PredictionError, DataValidationError
from ..mlflow_integration.logging_utils import PipelineLogger
from ..mlflow_integration.experiment_manager import ExperimentManager
from .batch_processor import BatchProcessor


logger = logging.getLogger(__name__)


class StreamingPipeline:
    """
    Streaming prediction pipeline for continuous data processing.

    This class provides streaming capabilities for time series prediction,
    resource usage prediction, and real-time performance monitoring.
    """

    def __init__(self,
                 config: PipelineConfig,
                 batch_processor: BatchProcessor,
                 experiment_manager: ExperimentManager):
        """
        Initialize the streaming pipeline.

        Args:
            config: Pipeline configuration
            batch_processor: Configured batch processor
            experiment_manager: MLflow experiment manager
        """
        self.config = config
        self.batch_processor = batch_processor
        self.experiment_manager = experiment_manager
        self.logger = PipelineLogger("StreamingPipeline")

        # Streaming state
        self.is_streaming = False
        self.stream_start_time = None
        self.stream_end_time = None
        self.streaming_thread = None

        # Performance tracking
        self.streaming_metrics = {
            'total_batches_processed': 0,
            'total_samples_processed': 0,
            'total_processing_time': 0.0,
            'avg_latency': 0.0,
            'throughput_samples_per_second': 0.0,
            'error_count': 0,
            'drift_detection_count': 0
        }

        # Real-time monitoring
        self.recent_results = deque(maxlen=100)  # Keep last 100 batch results
        self.performance_window = deque(maxlen=50)  # Performance metrics window
        self.prediction_buffer = deque(maxlen=1000)  # Prediction buffer for analysis

        # Resource usage tracking
        self.resource_usage_history: List[Dict[str, Any]] = []
        self.resource_predictions_accuracy: List[Dict[str, float]] = []

        # Callbacks for real-time monitoring
        self.batch_callbacks: List[Callable[[BatchResult], None]] = []
        self.drift_callbacks: List[Callable[[float], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []

        logger.info("StreamingPipeline initialized with batch_size=%d", config.batch_size)

    def start_streaming(self, data_generator: Generator[np.ndarray, None, None],
                       target_generator: Optional[Generator[np.ndarray, None, None]] = None,
                       max_batches: Optional[int] = None,
                       max_duration: Optional[timedelta] = None) -> None:
        """
        Start streaming data processing.

        Args:
            data_generator: Generator yielding feature batches
            target_generator: Optional generator yielding target batches for accuracy tracking
            max_batches: Maximum number of batches to process (None for unlimited)
            max_duration: Maximum streaming duration (None for unlimited)

        Raises:
            PredictionError: If streaming fails to start
        """
        if self.is_streaming:
            raise PredictionError("Streaming is already active", prediction_type="streaming_start")

        self.logger.info("Starting streaming pipeline with max_batches=%s, max_duration=%s",
                        max_batches, max_duration)

        try:
            # Initialize streaming state
            self.is_streaming = True
            self.stream_start_time = datetime.now()
            self.stream_end_time = None

            # Reset metrics
            self._reset_streaming_metrics()

            # Start streaming in separate thread
            self.streaming_thread = threading.Thread(
                target=self._streaming_worker,
                args=(data_generator, target_generator, max_batches, max_duration),
                daemon=True
            )
            self.streaming_thread.start()

            self.logger.info("Streaming pipeline started successfully")

        except Exception as e:
            self.is_streaming = False
            raise PredictionError(
                f"Failed to start streaming pipeline: {str(e)}",
                prediction_type="streaming_start"
            ) from e

    def stop_streaming(self, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Stop streaming data processing.

        Args:
            timeout: Maximum time to wait for streaming to stop

        Returns:
            Dictionary containing final streaming statistics

        Raises:
            PredictionError: If streaming fails to stop properly
        """
        if not self.is_streaming:
            self.logger.warning("Streaming is not currently active")
            return self.get_streaming_summary()

        self.logger.info("Stopping streaming pipeline")

        try:
            # Signal stop
            self.is_streaming = False
            self.stream_end_time = datetime.now()

            # Wait for streaming thread to finish
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=timeout)

                if self.streaming_thread.is_alive():
                    self.logger.warning("Streaming thread did not stop within timeout")

            # Generate final summary
            final_summary = self.get_streaming_summary()

            # Log final metrics
            self._log_streaming_summary(final_summary)

            self.logger.info("Streaming pipeline stopped successfully")
            return final_summary

        except Exception as e:
            raise PredictionError(
                f"Failed to stop streaming pipeline: {str(e)}",
                prediction_type="streaming_stop"
            ) from e

    def _streaming_worker(self, data_generator: Generator[np.ndarray, None, None],
                         target_generator: Optional[Generator[np.ndarray, None, None]],
                         max_batches: Optional[int],
                         max_duration: Optional[timedelta]) -> None:
        """
        Worker function for streaming data processing.

        Args:
            data_generator: Generator yielding feature batches
            target_generator: Optional generator yielding target batches
            max_batches: Maximum number of batches to process
            max_duration: Maximum streaming duration
        """
        batch_count = 0
        start_time = time.time()

        try:
            # Create target iterator if provided
            target_iter = iter(target_generator) if target_generator else None

            for X_batch in data_generator:
                if not self.is_streaming:
                    break

                # Check batch limit
                if max_batches and batch_count >= max_batches:
                    self.logger.info("Reached maximum batch limit: %d", max_batches)
                    break

                # Check duration limit
                if max_duration:
                    elapsed = timedelta(seconds=time.time() - start_time)
                    if elapsed >= max_duration:
                        self.logger.info("Reached maximum duration: %s", max_duration)
                        break

                try:
                    # Get target batch if available
                    y_batch = None
                    if target_iter:
                        try:
                            y_batch = next(target_iter)
                        except StopIteration:
                            self.logger.warning("Target generator exhausted before data generator")
                            target_iter = None

                    # Process batch
                    batch_result = self._process_streaming_batch(X_batch, y_batch)

                    # Update streaming metrics
                    self._update_streaming_metrics(batch_result)

                    # Track resource usage predictions
                    self._track_resource_usage(batch_result)

                    # Execute callbacks
                    self._execute_batch_callbacks(batch_result)

                    # Check for drift
                    if batch_result.noise_ratio > self.config.noise_threshold:
                        self._execute_drift_callbacks(batch_result.noise_ratio)

                    batch_count += 1

                except Exception as e:
                    self.logger.error("Error processing streaming batch %d: %s", batch_count, str(e))
                    self.streaming_metrics['error_count'] += 1
                    self._execute_error_callbacks(e)

                    # Continue processing unless it's a critical error
                    if isinstance(e, DataValidationError):
                        continue
                    else:
                        # For other errors, we might want to stop streaming
                        break

        except Exception as e:
            self.logger.error("Critical error in streaming worker: %s", str(e))
            self._execute_error_callbacks(e)

        finally:
            self.is_streaming = False
            self.stream_end_time = datetime.now()
            self.logger.info("Streaming worker finished after processing %d batches", batch_count)

    def _process_streaming_batch(self, X_batch: np.ndarray,
                               y_batch: Optional[np.ndarray] = None) -> BatchResult:
        """
        Process a single streaming batch with performance tracking.

        Args:
            X_batch: Input feature batch
            y_batch: Optional target batch

        Returns:
            BatchResult from batch processing

        Raises:
            PredictionError: If batch processing fails
        """
        batch_start_time = time.time()

        try:
            # Process batch using batch processor
            batch_result = self.batch_processor.process_batch(X_batch, y_batch)

            # Add to recent results
            self.recent_results.append(batch_result)

            # Track performance metrics
            batch_latency = time.time() - batch_start_time
            performance_metrics = {
                'timestamp': datetime.now().isoformat(),
                'batch_size': len(X_batch),
                'processing_time': batch_result.processing_time,
                'total_latency': batch_latency,
                'noise_ratio': batch_result.noise_ratio,
                'successful_predictions': len([p for p in batch_result.timeseries_predictions if not np.isnan(p)]),
                'resource_predictions_count': len(batch_result.resource_predictions)
            }

            self.performance_window.append(performance_metrics)

            # Store predictions for analysis
            self.prediction_buffer.extend(batch_result.timeseries_predictions)

            return batch_result

        except Exception as e:
            raise PredictionError(
                f"Failed to process streaming batch: {str(e)}",
                prediction_type="streaming_batch",
                batch_size=len(X_batch) if X_batch is not None else 0
            ) from e

    def _update_streaming_metrics(self, batch_result: BatchResult):
        """
        Update streaming performance metrics.

        Args:
            batch_result: Results from batch processing
        """
        self.streaming_metrics['total_batches_processed'] += 1
        self.streaming_metrics['total_samples_processed'] += len(batch_result.cluster_assignments)
        self.streaming_metrics['total_processing_time'] += batch_result.processing_time

        # Update averages
        if self.streaming_metrics['total_batches_processed'] > 0:
            self.streaming_metrics['avg_latency'] = (
                self.streaming_metrics['total_processing_time'] /
                self.streaming_metrics['total_batches_processed']
            )

        # Calculate throughput
        if self.stream_start_time:
            elapsed_time = (datetime.now() - self.stream_start_time).total_seconds()
            if elapsed_time > 0:
                self.streaming_metrics['throughput_samples_per_second'] = (
                    self.streaming_metrics['total_samples_processed'] / elapsed_time
                )

        # Track drift detection
        if batch_result.noise_ratio > self.config.noise_threshold:
            self.streaming_metrics['drift_detection_count'] += 1

    def _track_resource_usage(self, batch_result: BatchResult):
        """
        Track resource usage predictions and actual usage.

        Args:
            batch_result: Results from batch processing
        """
        try:
            # Calculate actual resource usage
            actual_usage = {
                'processing_time': batch_result.processing_time,
                'memory_usage': self._estimate_memory_usage(batch_result),
                'computational_complexity': self._estimate_computational_complexity(batch_result)
            }

            # Compare with predictions if available
            resource_accuracy = {}
            for cluster_id, predicted_resources in batch_result.resource_predictions.items():
                if 'processing_time' in predicted_resources:
                    predicted_time = predicted_resources['processing_time']
                    actual_time = batch_result.processing_time

                    # Calculate prediction accuracy (as percentage error)
                    if actual_time > 0:
                        time_error = abs(predicted_time - actual_time) / actual_time * 100
                        resource_accuracy[f'cluster_{cluster_id}_time_error'] = time_error

            # Store resource usage history
            resource_record = {
                'timestamp': datetime.now().isoformat(),
                'batch_id': self.streaming_metrics['total_batches_processed'],
                'actual_usage': actual_usage,
                'predicted_usage': batch_result.resource_predictions,
                'accuracy_metrics': resource_accuracy
            }

            self.resource_usage_history.append(resource_record)

            # Keep only recent history
            max_history_size = 500
            if len(self.resource_usage_history) > max_history_size:
                self.resource_usage_history = self.resource_usage_history[-max_history_size:]

            # Track accuracy metrics
            if resource_accuracy:
                self.resource_predictions_accuracy.append(resource_accuracy)

        except Exception as e:
            self.logger.error("Failed to track resource usage: %s", str(e))

    def _estimate_memory_usage(self, batch_result: BatchResult) -> float:
        """
        Estimate memory usage for the batch processing.

        Args:
            batch_result: Results from batch processing

        Returns:
            Estimated memory usage in MB
        """
        try:
            # Rough estimation based on data sizes
            cluster_assignments_size = batch_result.cluster_assignments.nbytes
            predictions_size = batch_result.timeseries_predictions.nbytes

            # Add overhead for resource predictions (estimated)
            resource_overhead = len(batch_result.resource_predictions) * 1024  # 1KB per cluster

            total_bytes = cluster_assignments_size + predictions_size + resource_overhead
            return total_bytes / (1024 * 1024)  # Convert to MB

        except Exception:
            return 0.0

    def _estimate_computational_complexity(self, batch_result: BatchResult) -> float:
        """
        Estimate computational complexity for the batch processing.

        Args:
            batch_result: Results from batch processing

        Returns:
            Estimated computational complexity score
        """
        try:
            # Simple complexity estimation based on:
            # - Number of samples processed
            # - Number of clusters involved
            # - Processing time

            sample_count = len(batch_result.cluster_assignments)
            cluster_count = len(batch_result.resource_predictions)
            processing_time = batch_result.processing_time

            # Complexity score (arbitrary units)
            complexity = (sample_count * cluster_count * processing_time) / 1000.0
            return complexity

        except Exception:
            return 0.0

    def _execute_batch_callbacks(self, batch_result: BatchResult):
        """
        Execute registered batch processing callbacks.

        Args:
            batch_result: Results from batch processing
        """
        for callback in self.batch_callbacks:
            try:
                callback(batch_result)
            except Exception as e:
                self.logger.error("Error in batch callback: %s", str(e))

    def _execute_drift_callbacks(self, noise_ratio: float):
        """
        Execute registered drift detection callbacks.

        Args:
            noise_ratio: Current noise ratio
        """
        for callback in self.drift_callbacks:
            try:
                callback(noise_ratio)
            except Exception as e:
                self.logger.error("Error in drift callback: %s", str(e))

    def _execute_error_callbacks(self, error: Exception):
        """
        Execute registered error callbacks.

        Args:
            error: Exception that occurred
        """
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error("Error in error callback: %s", str(e))

    def _reset_streaming_metrics(self):
        """Reset streaming metrics to initial state."""
        self.streaming_metrics = {
            'total_batches_processed': 0,
            'total_samples_processed': 0,
            'total_processing_time': 0.0,
            'avg_latency': 0.0,
            'throughput_samples_per_second': 0.0,
            'error_count': 0,
            'drift_detection_count': 0
        }
        self.recent_results.clear()
        self.performance_window.clear()
        self.prediction_buffer.clear()
        self.resource_usage_history.clear()
        self.resource_predictions_accuracy.clear()

    def _log_streaming_summary(self, summary: Dict[str, Any]):
        """
        Log streaming summary to MLflow.

        Args:
            summary: Streaming summary statistics
        """
        try:
            # Log main streaming metrics
            streaming_metrics = {
                'streaming_duration_seconds': summary.get('streaming_duration_seconds', 0),
                'total_batches_processed': summary.get('total_batches_processed', 0),
                'total_samples_processed': summary.get('total_samples_processed', 0),
                'avg_processing_time': summary.get('avg_processing_time', 0),
                'throughput_samples_per_second': summary.get('throughput_samples_per_second', 0),
                'error_rate': summary.get('error_rate', 0),
                'drift_detection_rate': summary.get('drift_detection_rate', 0)
            }

            self.logger.log_metrics(streaming_metrics, prefix="streaming_summary")

            # Log performance statistics
            if 'performance_stats' in summary:
                perf_stats = summary['performance_stats']
                performance_metrics = {
                    'avg_batch_latency': perf_stats.get('avg_batch_latency', 0),
                    'max_batch_latency': perf_stats.get('max_batch_latency', 0),
                    'min_batch_latency': perf_stats.get('min_batch_latency', 0),
                    'avg_noise_ratio': perf_stats.get('avg_noise_ratio', 0),
                    'max_noise_ratio': perf_stats.get('max_noise_ratio', 0)
                }
                self.logger.log_metrics(performance_metrics, prefix="streaming_performance")

            # Log resource usage statistics
            if 'resource_stats' in summary:
                resource_stats = summary['resource_stats']
                resource_metrics = {
                    'avg_memory_usage_mb': resource_stats.get('avg_memory_usage', 0),
                    'avg_computational_complexity': resource_stats.get('avg_computational_complexity', 0),
                    'resource_prediction_accuracy': resource_stats.get('avg_prediction_accuracy', 0)
                }
                self.logger.log_metrics(resource_metrics, prefix="streaming_resources")

        except Exception as e:
            self.logger.error("Failed to log streaming summary: %s", str(e))

    def get_streaming_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive streaming performance summary.

        Returns:
            Dictionary containing streaming statistics
        """
        try:
            # Calculate streaming duration
            if self.stream_start_time:
                end_time = self.stream_end_time or datetime.now()
                duration = (end_time - self.stream_start_time).total_seconds()
            else:
                duration = 0

            summary = {
                'streaming_status': 'active' if self.is_streaming else 'stopped',
                'streaming_duration_seconds': duration,
                'stream_start_time': self.stream_start_time.isoformat() if self.stream_start_time else None,
                'stream_end_time': self.stream_end_time.isoformat() if self.stream_end_time else None,
                **self.streaming_metrics
            }

            # Add error rate
            if self.streaming_metrics['total_batches_processed'] > 0:
                summary['error_rate'] = (
                    self.streaming_metrics['error_count'] /
                    self.streaming_metrics['total_batches_processed']
                )
                summary['drift_detection_rate'] = (
                    self.streaming_metrics['drift_detection_count'] /
                    self.streaming_metrics['total_batches_processed']
                )
            else:
                summary['error_rate'] = 0.0
                summary['drift_detection_rate'] = 0.0

            # Add performance statistics
            if self.performance_window:
                latencies = [p['total_latency'] for p in self.performance_window]
                noise_ratios = [p['noise_ratio'] for p in self.performance_window]
                batch_sizes = [p['batch_size'] for p in self.performance_window]

                summary['performance_stats'] = {
                    'avg_batch_latency': np.mean(latencies),
                    'max_batch_latency': np.max(latencies),
                    'min_batch_latency': np.min(latencies),
                    'std_batch_latency': np.std(latencies),
                    'avg_noise_ratio': np.mean(noise_ratios),
                    'max_noise_ratio': np.max(noise_ratios),
                    'min_noise_ratio': np.min(noise_ratios),
                    'avg_batch_size': np.mean(batch_sizes),
                    'total_performance_samples': len(self.performance_window)
                }

            # Add resource usage statistics
            if self.resource_usage_history:
                memory_usage = [r['actual_usage']['memory_usage'] for r in self.resource_usage_history]
                complexity = [r['actual_usage']['computational_complexity'] for r in self.resource_usage_history]

                # Calculate average prediction accuracy
                accuracy_values = []
                for acc_dict in self.resource_predictions_accuracy:
                    accuracy_values.extend(acc_dict.values())

                summary['resource_stats'] = {
                    'avg_memory_usage': np.mean(memory_usage),
                    'max_memory_usage': np.max(memory_usage),
                    'avg_computational_complexity': np.mean(complexity),
                    'max_computational_complexity': np.max(complexity),
                    'avg_prediction_accuracy': np.mean(accuracy_values) if accuracy_values else 0.0,
                    'total_resource_samples': len(self.resource_usage_history)
                }

            # Add prediction statistics
            if self.prediction_buffer:
                valid_predictions = [p for p in self.prediction_buffer if not np.isnan(p)]
                if valid_predictions:
                    summary['prediction_stats'] = {
                        'total_predictions': len(self.prediction_buffer),
                        'valid_predictions': len(valid_predictions),
                        'prediction_success_rate': len(valid_predictions) / len(self.prediction_buffer),
                        'avg_prediction_value': np.mean(valid_predictions),
                        'std_prediction_value': np.std(valid_predictions),
                        'min_prediction_value': np.min(valid_predictions),
                        'max_prediction_value': np.max(valid_predictions)
                    }

            return summary

        except Exception as e:
            self.logger.error("Failed to generate streaming summary: %s", str(e))
            return {
                'error': str(e),
                'streaming_status': 'error',
                **self.streaming_metrics
            }

    def add_batch_callback(self, callback: Callable[[BatchResult], None]):
        """
        Add callback function to be executed after each batch processing.

        Args:
            callback: Function that takes BatchResult as argument
        """
        self.batch_callbacks.append(callback)
        self.logger.info("Added batch callback: %s", callback.__name__)

    def add_drift_callback(self, callback: Callable[[float], None]):
        """
        Add callback function to be executed when drift is detected.

        Args:
            callback: Function that takes noise_ratio as argument
        """
        self.drift_callbacks.append(callback)
        self.logger.info("Added drift callback: %s", callback.__name__)

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """
        Add callback function to be executed when errors occur.

        Args:
            callback: Function that takes Exception as argument
        """
        self.error_callbacks.append(callback)
        self.logger.info("Added error callback: %s", callback.__name__)

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get real-time streaming metrics for monitoring dashboards.

        Returns:
            Dictionary containing current streaming metrics
        """
        if not self.performance_window:
            return {'status': 'no_data'}

        try:
            # Get recent performance data (last 10 batches)
            recent_performance = list(self.performance_window)[-10:]

            current_metrics = {
                'timestamp': datetime.now().isoformat(),
                'is_streaming': self.is_streaming,
                'recent_batch_count': len(recent_performance),
                'current_throughput': self.streaming_metrics['throughput_samples_per_second'],
                'current_error_rate': self.streaming_metrics['error_count'] / max(1, self.streaming_metrics['total_batches_processed']),
                'recent_avg_latency': np.mean([p['total_latency'] for p in recent_performance]),
                'recent_avg_noise_ratio': np.mean([p['noise_ratio'] for p in recent_performance]),
                'recent_max_noise_ratio': np.max([p['noise_ratio'] for p in recent_performance]),
                'drift_currently_detected': self.batch_processor.is_drift_detected,
                'total_batches_processed': self.streaming_metrics['total_batches_processed'],
                'total_samples_processed': self.streaming_metrics['total_samples_processed']
            }

            # Add resource usage if available
            if self.resource_usage_history:
                recent_resources = self.resource_usage_history[-10:]
                current_metrics.update({
                    'recent_avg_memory_usage': np.mean([r['actual_usage']['memory_usage'] for r in recent_resources]),
                    'recent_avg_complexity': np.mean([r['actual_usage']['computational_complexity'] for r in recent_resources])
                })

            return current_metrics

        except Exception as e:
            self.logger.error("Failed to get real-time metrics: %s", str(e))
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    @property
    def is_active(self) -> bool:
        """Check if streaming is currently active."""
        return self.is_streaming

    @property
    def current_throughput(self) -> float:
        """Get current throughput in samples per second."""
        return self.streaming_metrics['throughput_samples_per_second']

    @property
    def total_samples_processed(self) -> int:
        """Get total number of samples processed."""
        return self.streaming_metrics['total_samples_processed']