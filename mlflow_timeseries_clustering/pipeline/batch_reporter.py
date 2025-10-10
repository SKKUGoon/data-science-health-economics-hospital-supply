"""
Batch processing reporter for the MLflow Time Series Clustering Pipeline.

This module implements the BatchReporter class that generates comprehensive
reports and visualizations for batch processing performance, noise ratio tracking,
and processing metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..core.config import PipelineConfig
from ..core.exceptions import PredictionError
from ..mlflow_integration.logging_utils import PipelineLogger
from ..mlflow_integration.artifact_manager import ArtifactManager


logger = logging.getLogger(__name__)


class BatchReporter:
    """
    Reporter for batch processing performance and metrics.

    This class generates comprehensive reports and visualizations for batch processing
    including noise ratio tracking, processing performance, and drift detection analysis.
    """

    def __init__(self, config: PipelineConfig, artifact_manager: ArtifactManager):
        """
        Initialize the batch reporter.

        Args:
            config: Pipeline configuration
            artifact_manager: Artifact manager for saving reports and visualizations
        """
        self.config = config
        self.artifact_manager = artifact_manager
        self.logger = PipelineLogger("BatchReporter")

        # Set up visualization style
        self._setup_visualization_style()

        logger.info("BatchReporter initialized")

    def _setup_visualization_style(self):
        """Set up consistent visualization style using custom color scheme."""
        # Custom color palette: Black, White, Dark Grey, Dark Green
        self.colors = ['black', 'white', 'darkgray', '#0d4f01']
        sns.set_palette(self.colors)
        plt.style.use('default')

        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.size': 10
        })

    def generate_batch_processing_summary_report(self,
                                               processing_history: List[Dict[str, Any]],
                                               streaming_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive batch processing summary report.

        Args:
            processing_history: List of batch processing records
            streaming_summary: Optional streaming pipeline summary

        Returns:
            Dictionary containing report data and visualization paths

        Raises:
            PredictionError: If report generation fails
        """
        try:
            self.logger.info("Generating batch processing summary report")

            if not processing_history:
                return {
                    'status': 'no_data',
                    'message': 'No batch processing history available',
                    'timestamp': datetime.now().isoformat()
                }

            # Generate report sections
            report = {
                'timestamp': datetime.now().isoformat(),
                'report_type': 'batch_processing_summary',
                'processing_overview': self._generate_processing_overview(processing_history),
                'performance_analysis': self._generate_performance_analysis(processing_history),
                'noise_ratio_analysis': self._generate_noise_ratio_analysis(processing_history),
                'drift_detection_analysis': self._generate_drift_detection_analysis(processing_history),
                'visualizations': {}
            }

            # Add streaming analysis if available
            if streaming_summary:
                report['streaming_analysis'] = self._generate_streaming_analysis(streaming_summary)

            # Generate visualizations
            report['visualizations'] = self._generate_batch_processing_visualizations(
                processing_history, streaming_summary
            )

            # Create HTML report
            html_report = self._create_batch_processing_html_report(report)
            report_path = self.artifact_manager.save_report_artifact(
                report_content=html_report,
                filename="batch_processing_summary_report.html",
                subfolder="batch_processing"
            )

            report['report_path'] = report_path

            self.logger.info("Successfully generated batch processing summary report")
            return report

        except Exception as e:
            self.logger.error("Failed to generate batch processing summary report: %s", str(e))
            raise PredictionError(
                f"Failed to generate batch processing summary report: {str(e)}",
                prediction_type="batch_processing_report"
            ) from e

    def _generate_processing_overview(self, processing_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate processing overview statistics.

        Args:
            processing_history: List of batch processing records

        Returns:
            Dictionary containing processing overview
        """
        try:
            total_batches = len(processing_history)
            total_samples = sum(record['sample_count'] for record in processing_history)
            total_processing_time = sum(record['processing_time'] for record in processing_history)

            # Calculate time range
            timestamps = [datetime.fromisoformat(record['timestamp']) for record in processing_history]
            time_range = max(timestamps) - min(timestamps)

            # Calculate throughput
            avg_throughput = total_samples / total_processing_time if total_processing_time > 0 else 0

            overview = {
                'total_batches_processed': total_batches,
                'total_samples_processed': total_samples,
                'total_processing_time_seconds': total_processing_time,
                'processing_time_range': {
                    'start': min(timestamps).isoformat(),
                    'end': max(timestamps).isoformat(),
                    'duration_seconds': time_range.total_seconds()
                },
                'average_batch_size': total_samples / total_batches if total_batches > 0 else 0,
                'average_processing_time_per_batch': total_processing_time / total_batches if total_batches > 0 else 0,
                'average_throughput_samples_per_second': avg_throughput,
                'batch_size_stats': self._calculate_stats([record['sample_count'] for record in processing_history]),
                'processing_time_stats': self._calculate_stats([record['processing_time'] for record in processing_history])
            }

            return overview

        except Exception as e:
            self.logger.error("Failed to generate processing overview: %s", str(e))
            return {'error': str(e)}

    def _generate_performance_analysis(self, processing_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate performance analysis statistics.

        Args:
            processing_history: List of batch processing records

        Returns:
            Dictionary containing performance analysis
        """
        try:
            # Extract performance metrics
            successful_predictions = [record.get('successful_predictions', 0) for record in processing_history]
            failed_predictions = [record.get('failed_predictions', 0) for record in processing_history]
            clusters_predicted = [record.get('clusters_predicted', 0) for record in processing_history]
            noise_predictions = [record.get('noise_point_predictions', 0) for record in processing_history]

            # Calculate success rates
            total_successful = sum(successful_predictions)
            total_failed = sum(failed_predictions)
            total_predictions = total_successful + total_failed

            success_rate = total_successful / total_predictions if total_predictions > 0 else 0
            failure_rate = total_failed / total_predictions if total_predictions > 0 else 0

            analysis = {
                'prediction_performance': {
                    'total_successful_predictions': total_successful,
                    'total_failed_predictions': total_failed,
                    'overall_success_rate': success_rate,
                    'overall_failure_rate': failure_rate,
                    'avg_successful_per_batch': np.mean(successful_predictions),
                    'avg_failed_per_batch': np.mean(failed_predictions)
                },
                'cluster_analysis': {
                    'avg_clusters_per_batch': np.mean(clusters_predicted),
                    'max_clusters_per_batch': np.max(clusters_predicted),
                    'min_clusters_per_batch': np.min(clusters_predicted),
                    'cluster_count_stats': self._calculate_stats(clusters_predicted)
                },
                'noise_point_analysis': {
                    'total_noise_predictions': sum(noise_predictions),
                    'avg_noise_per_batch': np.mean(noise_predictions),
                    'max_noise_per_batch': np.max(noise_predictions),
                    'noise_prediction_stats': self._calculate_stats(noise_predictions)
                }
            }

            return analysis

        except Exception as e:
            self.logger.error("Failed to generate performance analysis: %s", str(e))
            return {'error': str(e)}

    def _generate_noise_ratio_analysis(self, processing_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate noise ratio analysis and drift detection insights.

        Args:
            processing_history: List of batch processing records

        Returns:
            Dictionary containing noise ratio analysis
        """
        try:
            noise_ratios = [record['noise_ratio'] for record in processing_history]
            threshold = self.config.noise_threshold

            # Basic statistics
            noise_stats = self._calculate_stats(noise_ratios)

            # Threshold analysis
            high_noise_batches = [nr for nr in noise_ratios if nr > threshold]
            high_noise_count = len(high_noise_batches)
            high_noise_percentage = high_noise_count / len(noise_ratios) * 100 if noise_ratios else 0

            # Trend analysis
            trend_analysis = self._analyze_noise_ratio_trend(noise_ratios)

            # Consecutive high noise analysis
            consecutive_analysis = self._analyze_consecutive_high_noise(noise_ratios, threshold)

            analysis = {
                'noise_ratio_statistics': noise_stats,
                'threshold_analysis': {
                    'noise_threshold': threshold,
                    'high_noise_batch_count': high_noise_count,
                    'high_noise_percentage': high_noise_percentage,
                    'batches_above_threshold': high_noise_count,
                    'batches_below_threshold': len(noise_ratios) - high_noise_count
                },
                'trend_analysis': trend_analysis,
                'consecutive_high_noise_analysis': consecutive_analysis,
                'drift_risk_assessment': self._assess_drift_risk(noise_ratios, threshold)
            }

            return analysis

        except Exception as e:
            self.logger.error("Failed to generate noise ratio analysis: %s", str(e))
            return {'error': str(e)}

    def _generate_drift_detection_analysis(self, processing_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate drift detection analysis.

        Args:
            processing_history: List of batch processing records

        Returns:
            Dictionary containing drift detection analysis
        """
        try:
            noise_ratios = [record['noise_ratio'] for record in processing_history]
            timestamps = [datetime.fromisoformat(record['timestamp']) for record in processing_history]
            threshold = self.config.noise_threshold

            # Detect drift periods
            drift_periods = self._detect_drift_periods(noise_ratios, timestamps, threshold)

            # Calculate drift statistics
            drift_stats = {
                'total_drift_periods': len(drift_periods),
                'total_drift_duration_seconds': sum(
                    (period['end_time'] - period['start_time']).total_seconds()
                    for period in drift_periods
                ),
                'avg_drift_duration_seconds': 0,
                'max_drift_duration_seconds': 0,
                'drift_frequency_per_hour': 0
            }

            if drift_periods:
                durations = [(period['end_time'] - period['start_time']).total_seconds() for period in drift_periods]
                drift_stats['avg_drift_duration_seconds'] = np.mean(durations)
                drift_stats['max_drift_duration_seconds'] = np.max(durations)

                # Calculate frequency
                total_time_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600
                if total_time_hours > 0:
                    drift_stats['drift_frequency_per_hour'] = len(drift_periods) / total_time_hours

            analysis = {
                'drift_statistics': drift_stats,
                'drift_periods': drift_periods,
                'drift_pattern_analysis': self._analyze_drift_patterns(drift_periods),
                'recovery_analysis': self._analyze_drift_recovery(drift_periods, noise_ratios, timestamps)
            }

            return analysis

        except Exception as e:
            self.logger.error("Failed to generate drift detection analysis: %s", str(e))
            return {'error': str(e)}

    def _generate_streaming_analysis(self, streaming_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate streaming pipeline analysis.

        Args:
            streaming_summary: Streaming pipeline summary

        Returns:
            Dictionary containing streaming analysis
        """
        try:
            analysis = {
                'streaming_overview': {
                    'streaming_status': streaming_summary.get('streaming_status', 'unknown'),
                    'streaming_duration_seconds': streaming_summary.get('streaming_duration_seconds', 0),
                    'total_throughput': streaming_summary.get('throughput_samples_per_second', 0),
                    'error_rate': streaming_summary.get('error_rate', 0),
                    'drift_detection_rate': streaming_summary.get('drift_detection_rate', 0)
                }
            }

            # Add performance stats if available
            if 'performance_stats' in streaming_summary:
                analysis['performance_comparison'] = {
                    'batch_vs_streaming_latency': {
                        'streaming_avg_latency': streaming_summary['performance_stats'].get('avg_batch_latency', 0),
                        'streaming_max_latency': streaming_summary['performance_stats'].get('max_batch_latency', 0)
                    }
                }

            # Add resource stats if available
            if 'resource_stats' in streaming_summary:
                analysis['resource_utilization'] = streaming_summary['resource_stats']

            return analysis

        except Exception as e:
            self.logger.error("Failed to generate streaming analysis: %s", str(e))
            return {'error': str(e)}

    def _generate_batch_processing_visualizations(self,
                                                processing_history: List[Dict[str, Any]],
                                                streaming_summary: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate batch processing visualizations.

        Args:
            processing_history: List of batch processing records
            streaming_summary: Optional streaming summary

        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualizations = {}

        try:
            # 1. Processing Performance Over Time
            fig = self._create_processing_performance_plot(processing_history)
            viz_path = self.artifact_manager.save_visualization(
                figure=fig,
                filename="processing_performance_over_time.png",
                subfolder="batch_processing"
            )
            visualizations['processing_performance'] = viz_path
            plt.close(fig)

            # 2. Noise Ratio Tracking
            fig = self._create_noise_ratio_tracking_plot(processing_history)
            viz_path = self.artifact_manager.save_visualization(
                figure=fig,
                filename="noise_ratio_tracking.png",
                subfolder="batch_processing"
            )
            visualizations['noise_ratio_tracking'] = viz_path
            plt.close(fig)

            # 3. Batch Size and Processing Time Distribution
            fig = self._create_batch_distribution_plot(processing_history)
            viz_path = self.artifact_manager.save_visualization(
                figure=fig,
                filename="batch_distribution.png",
                subfolder="batch_processing"
            )
            visualizations['batch_distribution'] = viz_path
            plt.close(fig)

            # 4. Prediction Success Rate Analysis
            fig = self._create_prediction_success_plot(processing_history)
            viz_path = self.artifact_manager.save_visualization(
                figure=fig,
                filename="prediction_success_analysis.png",
                subfolder="batch_processing"
            )
            visualizations['prediction_success'] = viz_path
            plt.close(fig)

            # 5. Drift Detection Visualization
            fig = self._create_drift_detection_plot(processing_history)
            viz_path = self.artifact_manager.save_visualization(
                figure=fig,
                filename="drift_detection_analysis.png",
                subfolder="batch_processing"
            )
            visualizations['drift_detection'] = viz_path
            plt.close(fig)

            # 6. Cluster Activity Heatmap
            fig = self._create_cluster_activity_heatmap(processing_history)
            viz_path = self.artifact_manager.save_visualization(
                figure=fig,
                filename="cluster_activity_heatmap.png",
                subfolder="batch_processing"
            )
            visualizations['cluster_activity'] = viz_path
            plt.close(fig)

        except Exception as e:
            self.logger.error("Failed to generate visualizations: %s", str(e))
            visualizations['error'] = str(e)

        return visualizations

    def _create_processing_performance_plot(self, processing_history: List[Dict[str, Any]]):
        """Create processing performance over time plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Batch Processing Performance Over Time', fontsize=16, fontweight='bold')

        # Extract data
        timestamps = [datetime.fromisoformat(record['timestamp']) for record in processing_history]
        processing_times = [record['processing_time'] for record in processing_history]
        sample_counts = [record['sample_count'] for record in processing_history]
        throughput = [count / time if time > 0 else 0 for count, time in zip(sample_counts, processing_times)]

        # Processing time over time
        axes[0, 0].plot(timestamps, processing_times, color='#0d4f01', linewidth=2)
        axes[0, 0].set_title('Processing Time per Batch')
        axes[0, 0].set_ylabel('Processing Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Sample count over time
        axes[0, 1].plot(timestamps, sample_counts, color='darkgray', linewidth=2)
        axes[0, 1].set_title('Batch Size Over Time')
        axes[0, 1].set_ylabel('Sample Count')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Throughput over time
        axes[1, 0].plot(timestamps, throughput, color='black', linewidth=2)
        axes[1, 0].set_title('Throughput Over Time')
        axes[1, 0].set_ylabel('Samples per Second')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Processing time distribution
        axes[1, 1].hist(processing_times, bins=20, color='#0d4f01', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Processing Time Distribution')
        axes[1, 1].set_xlabel('Processing Time (seconds)')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    def _create_noise_ratio_tracking_plot(self, processing_history: List[Dict[str, Any]]):
        """Create noise ratio tracking plot with threshold line."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Noise Ratio Tracking and Drift Detection', fontsize=16, fontweight='bold')

        # Extract data
        timestamps = [datetime.fromisoformat(record['timestamp']) for record in processing_history]
        noise_ratios = [record['noise_ratio'] for record in processing_history]
        threshold = self.config.noise_threshold

        # Noise ratio over time
        axes[0].plot(timestamps, noise_ratios, color='#0d4f01', linewidth=2, label='Noise Ratio')
        axes[0].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        axes[0].fill_between(timestamps, noise_ratios, threshold,
                           where=[nr > threshold for nr in noise_ratios],
                           color='red', alpha=0.3, label='Above Threshold')
        axes[0].set_title('Noise Ratio Over Time')
        axes[0].set_ylabel('Noise Ratio')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)

        # Noise ratio distribution
        axes[1].hist(noise_ratios, bins=20, color='darkgray', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        axes[1].axvline(x=np.mean(noise_ratios), color='#0d4f01', linestyle='-', linewidth=2, label='Mean')
        axes[1].set_title('Noise Ratio Distribution')
        axes[1].set_xlabel('Noise Ratio')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()

        plt.tight_layout()
        return fig

    def _create_batch_distribution_plot(self, processing_history: List[Dict[str, Any]]):
        """Create batch size and processing time distribution plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Batch Processing Distributions', fontsize=16, fontweight='bold')

        # Extract data
        sample_counts = [record['sample_count'] for record in processing_history]
        processing_times = [record['processing_time'] for record in processing_history]

        # Batch size distribution
        axes[0, 0].hist(sample_counts, bins=20, color='#0d4f01', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Batch Size Distribution')
        axes[0, 0].set_xlabel('Sample Count')
        axes[0, 0].set_ylabel('Frequency')

        # Processing time distribution
        axes[0, 1].hist(processing_times, bins=20, color='darkgray', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Processing Time Distribution')
        axes[0, 1].set_xlabel('Processing Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')

        # Batch size vs processing time scatter
        axes[1, 0].scatter(sample_counts, processing_times, color='#0d4f01', alpha=0.6)
        axes[1, 0].set_title('Batch Size vs Processing Time')
        axes[1, 0].set_xlabel('Sample Count')
        axes[1, 0].set_ylabel('Processing Time (seconds)')

        # Processing efficiency (samples per second)
        efficiency = [count / time if time > 0 else 0 for count, time in zip(sample_counts, processing_times)]
        axes[1, 1].hist(efficiency, bins=20, color='black', alpha=0.7, edgecolor='darkgray')
        axes[1, 1].set_title('Processing Efficiency Distribution')
        axes[1, 1].set_xlabel('Samples per Second')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    def _create_prediction_success_plot(self, processing_history: List[Dict[str, Any]]):
        """Create prediction success rate analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Prediction Success Analysis', fontsize=16, fontweight='bold')

        # Extract data
        timestamps = [datetime.fromisoformat(record['timestamp']) for record in processing_history]
        successful = [record.get('successful_predictions', 0) for record in processing_history]
        failed = [record.get('failed_predictions', 0) for record in processing_history]
        clusters = [record.get('clusters_predicted', 0) for record in processing_history]

        # Success rate over time
        total_predictions = [s + f for s, f in zip(successful, failed)]
        success_rates = [s / t if t > 0 else 0 for s, t in zip(successful, total_predictions)]

        axes[0, 0].plot(timestamps, success_rates, color='#0d4f01', linewidth=2)
        axes[0, 0].set_title('Prediction Success Rate Over Time')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Successful vs failed predictions
        axes[0, 1].bar(['Successful', 'Failed'], [sum(successful), sum(failed)],
                      color=['#0d4f01', 'darkgray'])
        axes[0, 1].set_title('Total Successful vs Failed Predictions')
        axes[0, 1].set_ylabel('Count')

        # Clusters predicted over time
        axes[1, 0].plot(timestamps, clusters, color='black', linewidth=2)
        axes[1, 0].set_title('Clusters Predicted per Batch')
        axes[1, 0].set_ylabel('Number of Clusters')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Success rate distribution
        axes[1, 1].hist(success_rates, bins=20, color='darkgray', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Success Rate Distribution')
        axes[1, 1].set_xlabel('Success Rate')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    def _create_drift_detection_plot(self, processing_history: List[Dict[str, Any]]):
        """Create drift detection analysis plot."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Drift Detection Analysis', fontsize=16, fontweight='bold')

        # Extract data
        timestamps = [datetime.fromisoformat(record['timestamp']) for record in processing_history]
        noise_ratios = [record['noise_ratio'] for record in processing_history]
        threshold = self.config.noise_threshold

        # Detect drift periods
        drift_periods = self._detect_drift_periods(noise_ratios, timestamps, threshold)

        # Noise ratio with drift periods highlighted
        axes[0].plot(timestamps, noise_ratios, color='#0d4f01', linewidth=2, label='Noise Ratio')
        axes[0].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

        # Highlight drift periods
        for period in drift_periods:
            axes[0].axvspan(period['start_time'], period['end_time'],
                          color='red', alpha=0.2, label='Drift Period' if period == drift_periods[0] else "")

        axes[0].set_title('Noise Ratio with Drift Periods')
        axes[0].set_ylabel('Noise Ratio')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)

        # Drift period analysis
        if drift_periods:
            durations = [(period['end_time'] - period['start_time']).total_seconds() / 60 for period in drift_periods]
            max_noise = [period['max_noise_ratio'] for period in drift_periods]

            axes[1].scatter(durations, max_noise, color='red', s=100, alpha=0.7)
            axes[1].set_title('Drift Period Analysis: Duration vs Peak Noise Ratio')
            axes[1].set_xlabel('Duration (minutes)')
            axes[1].set_ylabel('Peak Noise Ratio')
        else:
            axes[1].text(0.5, 0.5, 'No drift periods detected',
                        transform=axes[1].transAxes, ha='center', va='center', fontsize=14)
            axes[1].set_title('Drift Period Analysis')

        plt.tight_layout()
        return fig

    def _create_cluster_activity_heatmap(self, processing_history: List[Dict[str, Any]]):
        """Create cluster activity heatmap over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle('Cluster Activity Heatmap', fontsize=16, fontweight='bold')

        try:
            # Extract cluster activity data
            timestamps = [datetime.fromisoformat(record['timestamp']) for record in processing_history]
            clusters_predicted = [record.get('clusters_predicted', 0) for record in processing_history]

            # Create time bins (e.g., hourly)
            if len(timestamps) > 1:
                time_range = max(timestamps) - min(timestamps)
                if time_range.total_seconds() > 3600:  # More than 1 hour
                    bin_size = timedelta(hours=1)
                elif time_range.total_seconds() > 60:  # More than 1 minute
                    bin_size = timedelta(minutes=10)
                else:
                    bin_size = timedelta(seconds=30)

                # Create heatmap data
                time_bins = []
                current_time = min(timestamps)
                while current_time <= max(timestamps):
                    time_bins.append(current_time)
                    current_time += bin_size

                # Aggregate cluster activity by time bins
                activity_matrix = []
                time_labels = []

                for i in range(len(time_bins) - 1):
                    bin_start = time_bins[i]
                    bin_end = time_bins[i + 1]

                    # Find records in this time bin
                    bin_clusters = []
                    for j, ts in enumerate(timestamps):
                        if bin_start <= ts < bin_end:
                            bin_clusters.append(clusters_predicted[j])

                    activity_matrix.append([np.mean(bin_clusters) if bin_clusters else 0])
                    time_labels.append(bin_start.strftime('%H:%M'))

                if activity_matrix:
                    # Create heatmap
                    sns.heatmap(np.array(activity_matrix).T,
                              xticklabels=time_labels[::max(1, len(time_labels)//10)],
                              yticklabels=['Avg Clusters'],
                              cmap='Greens',
                              ax=ax,
                              cbar_kws={'label': 'Average Clusters per Batch'})
                    ax.set_xlabel('Time')
                    ax.set_title('Cluster Activity Over Time')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for heatmap',
                           transform=ax.transAxes, ha='center', va='center', fontsize=14)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for heatmap',
                       transform=ax.transAxes, ha='center', va='center', fontsize=14)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating heatmap: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)

        plt.tight_layout()
        return fig

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values."""
        if not values:
            return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        }

    def _analyze_noise_ratio_trend(self, noise_ratios: List[float]) -> Dict[str, Any]:
        """Analyze noise ratio trend over time."""
        if len(noise_ratios) < 2:
            return {'trend': 'insufficient_data'}

        # Simple linear trend analysis
        x = np.arange(len(noise_ratios))
        coeffs = np.polyfit(x, noise_ratios, 1)
        slope = coeffs[0]

        trend_analysis = {
            'slope': float(slope),
            'trend_direction': 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable',
            'trend_strength': abs(slope),
            'r_squared': float(np.corrcoef(x, noise_ratios)[0, 1] ** 2) if len(noise_ratios) > 1 else 0
        }

        # Moving average analysis
        if len(noise_ratios) >= 5:
            window_size = min(5, len(noise_ratios) // 2)
            moving_avg = np.convolve(noise_ratios, np.ones(window_size)/window_size, mode='valid')
            trend_analysis['moving_average_trend'] = {
                'recent_avg': float(np.mean(moving_avg[-3:])) if len(moving_avg) >= 3 else float(moving_avg[-1]),
                'early_avg': float(np.mean(moving_avg[:3])) if len(moving_avg) >= 3 else float(moving_avg[0]),
                'change': float(np.mean(moving_avg[-3:]) - np.mean(moving_avg[:3])) if len(moving_avg) >= 6 else 0
            }

        return trend_analysis

    def _analyze_consecutive_high_noise(self, noise_ratios: List[float], threshold: float) -> Dict[str, Any]:
        """Analyze consecutive high noise periods."""
        consecutive_periods = []
        current_period_start = None
        current_period_length = 0

        for i, noise_ratio in enumerate(noise_ratios):
            if noise_ratio > threshold:
                if current_period_start is None:
                    current_period_start = i
                    current_period_length = 1
                else:
                    current_period_length += 1
            else:
                if current_period_start is not None:
                    consecutive_periods.append({
                        'start_index': current_period_start,
                        'length': current_period_length,
                        'max_noise_ratio': max(noise_ratios[current_period_start:current_period_start + current_period_length])
                    })
                    current_period_start = None
                    current_period_length = 0

        # Handle case where sequence ends with high noise
        if current_period_start is not None:
            consecutive_periods.append({
                'start_index': current_period_start,
                'length': current_period_length,
                'max_noise_ratio': max(noise_ratios[current_period_start:])
            })

        analysis = {
            'total_consecutive_periods': len(consecutive_periods),
            'max_consecutive_length': max([p['length'] for p in consecutive_periods]) if consecutive_periods else 0,
            'avg_consecutive_length': np.mean([p['length'] for p in consecutive_periods]) if consecutive_periods else 0,
            'consecutive_periods': consecutive_periods
        }

        return analysis

    def _assess_drift_risk(self, noise_ratios: List[float], threshold: float) -> Dict[str, Any]:
        """Assess drift risk based on noise ratio patterns."""
        if not noise_ratios:
            return {'risk_level': 'unknown', 'reason': 'no_data'}

        recent_ratios = noise_ratios[-10:] if len(noise_ratios) >= 10 else noise_ratios
        recent_avg = np.mean(recent_ratios)
        overall_avg = np.mean(noise_ratios)

        # Risk assessment criteria
        high_recent_avg = recent_avg > threshold * 0.8
        increasing_trend = recent_avg > overall_avg * 1.2
        high_volatility = np.std(recent_ratios) > threshold * 0.3
        frequent_threshold_breach = sum(1 for nr in recent_ratios if nr > threshold) > len(recent_ratios) * 0.3

        risk_factors = {
            'high_recent_average': high_recent_avg,
            'increasing_trend': increasing_trend,
            'high_volatility': high_volatility,
            'frequent_threshold_breach': frequent_threshold_breach
        }

        risk_score = sum(risk_factors.values())

        if risk_score >= 3:
            risk_level = 'high'
        elif risk_score >= 2:
            risk_level = 'medium'
        elif risk_score >= 1:
            risk_level = 'low'
        else:
            risk_level = 'minimal'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recent_average_noise_ratio': recent_avg,
            'overall_average_noise_ratio': overall_avg,
            'recommendation': self._get_drift_risk_recommendation(risk_level, risk_factors)
        }

    def _get_drift_risk_recommendation(self, risk_level: str, risk_factors: Dict[str, bool]) -> str:
        """Get recommendation based on drift risk assessment."""
        if risk_level == 'high':
            return "High drift risk detected. Consider immediate retraining or model adjustment."
        elif risk_level == 'medium':
            return "Medium drift risk. Monitor closely and prepare for potential retraining."
        elif risk_level == 'low':
            return "Low drift risk. Continue monitoring noise ratios."
        else:
            return "Minimal drift risk. Current model performance is stable."

    def _detect_drift_periods(self, noise_ratios: List[float],
                            timestamps: List[datetime],
                            threshold: float) -> List[Dict[str, Any]]:
        """Detect periods of sustained high noise ratios (drift periods)."""
        drift_periods = []
        in_drift = False
        drift_start = None
        drift_start_idx = None

        for i, (noise_ratio, timestamp) in enumerate(zip(noise_ratios, timestamps)):
            if noise_ratio > threshold and not in_drift:
                # Start of drift period
                in_drift = True
                drift_start = timestamp
                drift_start_idx = i
            elif noise_ratio <= threshold and in_drift:
                # End of drift period
                in_drift = False
                drift_periods.append({
                    'start_time': drift_start,
                    'end_time': timestamp,
                    'start_index': drift_start_idx,
                    'end_index': i,
                    'duration_seconds': (timestamp - drift_start).total_seconds(),
                    'max_noise_ratio': max(noise_ratios[drift_start_idx:i+1]),
                    'avg_noise_ratio': np.mean(noise_ratios[drift_start_idx:i+1])
                })

        # Handle case where sequence ends in drift
        if in_drift:
            drift_periods.append({
                'start_time': drift_start,
                'end_time': timestamps[-1],
                'start_index': drift_start_idx,
                'end_index': len(timestamps) - 1,
                'duration_seconds': (timestamps[-1] - drift_start).total_seconds(),
                'max_noise_ratio': max(noise_ratios[drift_start_idx:]),
                'avg_noise_ratio': np.mean(noise_ratios[drift_start_idx:])
            })

        return drift_periods

    def _analyze_drift_patterns(self, drift_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in drift periods."""
        if not drift_periods:
            return {'pattern_analysis': 'no_drift_periods'}

        durations = [period['duration_seconds'] for period in drift_periods]
        max_noise_ratios = [period['max_noise_ratio'] for period in drift_periods]

        return {
            'drift_duration_stats': self._calculate_stats(durations),
            'drift_intensity_stats': self._calculate_stats(max_noise_ratios),
            'drift_frequency_analysis': {
                'total_periods': len(drift_periods),
                'avg_time_between_drifts': self._calculate_avg_time_between_drifts(drift_periods)
            }
        }

    def _calculate_avg_time_between_drifts(self, drift_periods: List[Dict[str, Any]]) -> float:
        """Calculate average time between drift periods."""
        if len(drift_periods) < 2:
            return 0.0

        intervals = []
        for i in range(1, len(drift_periods)):
            interval = (drift_periods[i]['start_time'] - drift_periods[i-1]['end_time']).total_seconds()
            intervals.append(interval)

        return float(np.mean(intervals)) if intervals else 0.0

    def _analyze_drift_recovery(self, drift_periods: List[Dict[str, Any]],
                              noise_ratios: List[float],
                              timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze recovery patterns after drift periods."""
        if not drift_periods:
            return {'recovery_analysis': 'no_drift_periods'}

        recovery_times = []
        recovery_patterns = []

        for period in drift_periods:
            end_idx = period['end_index']

            # Look at next 10 batches after drift end (or until end of data)
            recovery_window = min(10, len(noise_ratios) - end_idx - 1)

            if recovery_window > 0:
                recovery_noise_ratios = noise_ratios[end_idx + 1:end_idx + 1 + recovery_window]

                # Find how long it takes to reach stable low noise
                stable_threshold = 0.1  # Consider stable if noise ratio < 0.1
                recovery_time = None

                for i, nr in enumerate(recovery_noise_ratios):
                    if nr < stable_threshold:
                        recovery_time = i + 1  # Number of batches to recover
                        break

                if recovery_time:
                    recovery_times.append(recovery_time)

                recovery_patterns.append({
                    'drift_end_time': period['end_time'],
                    'recovery_noise_ratios': recovery_noise_ratios,
                    'recovery_time_batches': recovery_time,
                    'immediate_recovery_ratio': recovery_noise_ratios[0] if recovery_noise_ratios else None
                })

        return {
            'avg_recovery_time_batches': np.mean(recovery_times) if recovery_times else None,
            'max_recovery_time_batches': max(recovery_times) if recovery_times else None,
            'recovery_success_rate': len(recovery_times) / len(drift_periods) if drift_periods else 0,
            'recovery_patterns': recovery_patterns
        }

    def _create_batch_processing_html_report(self, report: Dict[str, Any]) -> str:
        """Create HTML report for batch processing analysis."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Processing Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #0d4f01; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f5f5f5; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; }}
                .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Batch Processing Summary Report</h1>
                <p>Generated on: {report['timestamp']}</p>
            </div>

            <div class="section">
                <h2>Processing Overview</h2>
                {self._format_processing_overview_html(report.get('processing_overview', {}))}
            </div>

            <div class="section">
                <h2>Performance Analysis</h2>
                {self._format_performance_analysis_html(report.get('performance_analysis', {}))}
            </div>

            <div class="section">
                <h2>Noise Ratio Analysis</h2>
                {self._format_noise_analysis_html(report.get('noise_ratio_analysis', {}))}
            </div>

            <div class="section">
                <h2>Drift Detection Analysis</h2>
                {self._format_drift_analysis_html(report.get('drift_detection_analysis', {}))}
            </div>

            <div class="section">
                <h2>Visualizations</h2>
                {self._format_visualizations_html(report.get('visualizations', {}))}
            </div>
        </body>
        </html>
        """
        return html_content

    def _format_processing_overview_html(self, overview: Dict[str, Any]) -> str:
        """Format processing overview for HTML report."""
        if 'error' in overview:
            return f'<div class="warning">Error generating overview: {overview["error"]}</div>'

        return f"""
        <div class="metric">
            <strong>Total Batches:</strong> {overview.get('total_batches_processed', 0)}
        </div>
        <div class="metric">
            <strong>Total Samples:</strong> {overview.get('total_samples_processed', 0):,}
        </div>
        <div class="metric">
            <strong>Total Processing Time:</strong> {overview.get('total_processing_time_seconds', 0):.2f}s
        </div>
        <div class="metric">
            <strong>Average Throughput:</strong> {overview.get('average_throughput_samples_per_second', 0):.2f} samples/sec
        </div>
        <div class="metric">
            <strong>Average Batch Size:</strong> {overview.get('average_batch_size', 0):.1f}
        </div>
        """

    def _format_performance_analysis_html(self, analysis: Dict[str, Any]) -> str:
        """Format performance analysis for HTML report."""
        if 'error' in analysis:
            return f'<div class="warning">Error generating analysis: {analysis["error"]}</div>'

        pred_perf = analysis.get('prediction_performance', {})
        success_rate = pred_perf.get('overall_success_rate', 0)

        status_class = 'success' if success_rate > 0.9 else 'warning' if success_rate > 0.7 else 'warning'

        return f"""
        <div class="{status_class}">
            <strong>Overall Success Rate:</strong> {success_rate:.1%}
        </div>
        <div class="metric">
            <strong>Successful Predictions:</strong> {pred_perf.get('total_successful_predictions', 0):,}
        </div>
        <div class="metric">
            <strong>Failed Predictions:</strong> {pred_perf.get('total_failed_predictions', 0):,}
        </div>
        <div class="metric">
            <strong>Average Clusters per Batch:</strong> {analysis.get('cluster_analysis', {}).get('avg_clusters_per_batch', 0):.1f}
        </div>
        """

    def _format_noise_analysis_html(self, analysis: Dict[str, Any]) -> str:
        """Format noise ratio analysis for HTML report."""
        if 'error' in analysis:
            return f'<div class="warning">Error generating analysis: {analysis["error"]}</div>'

        threshold_analysis = analysis.get('threshold_analysis', {})
        high_noise_pct = threshold_analysis.get('high_noise_percentage', 0)

        status_class = 'warning' if high_noise_pct > 20 else 'success'

        return f"""
        <div class="{status_class}">
            <strong>High Noise Batches:</strong> {high_noise_pct:.1f}% above threshold
        </div>
        <div class="metric">
            <strong>Noise Threshold:</strong> {threshold_analysis.get('noise_threshold', 0):.3f}
        </div>
        <div class="metric">
            <strong>Average Noise Ratio:</strong> {analysis.get('noise_ratio_statistics', {}).get('mean', 0):.3f}
        </div>
        <div class="metric">
            <strong>Max Noise Ratio:</strong> {analysis.get('noise_ratio_statistics', {}).get('max', 0):.3f}
        </div>
        """

    def _format_drift_analysis_html(self, analysis: Dict[str, Any]) -> str:
        """Format drift detection analysis for HTML report."""
        if 'error' in analysis:
            return f'<div class="warning">Error generating analysis: {analysis["error"]}</div>'

        drift_stats = analysis.get('drift_statistics', {})
        drift_count = drift_stats.get('total_drift_periods', 0)

        status_class = 'warning' if drift_count > 0 else 'success'

        return f"""
        <div class="{status_class}">
            <strong>Drift Periods Detected:</strong> {drift_count}
        </div>
        <div class="metric">
            <strong>Total Drift Duration:</strong> {drift_stats.get('total_drift_duration_seconds', 0):.1f}s
        </div>
        <div class="metric">
            <strong>Average Drift Duration:</strong> {drift_stats.get('avg_drift_duration_seconds', 0):.1f}s
        </div>
        <div class="metric">
            <strong>Drift Frequency:</strong> {drift_stats.get('drift_frequency_per_hour', 0):.2f} per hour
        </div>
        """

    def _format_visualizations_html(self, visualizations: Dict[str, str]) -> str:
        """Format visualizations section for HTML report."""
        if 'error' in visualizations:
            return f'<div class="warning">Error generating visualizations: {visualizations["error"]}</div>'

        html = ""
        for viz_name, viz_path in visualizations.items():
            if viz_name != 'error':
                html += f"""
                <div class="visualization">
                    <h3>{viz_name.replace('_', ' ').title()}</h3>
                    <p>Visualization saved to: {viz_path}</p>
                </div>
                """

        return html if html else '<p>No visualizations generated.</p>'