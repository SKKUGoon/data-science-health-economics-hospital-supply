"""
Retraining comparison reporter for the MLflow Time Series Clustering Pipeline.

This module implements the RetrainingReporter class that generates
comprehensive reports comparing model performance before and after retraining,
including visualizations using seaborn with the specified color scheme.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .expanded_window_retrainer import RetrainingResult
from .retraining_trigger import RetrainingEvent, TriggerReason


logger = logging.getLogger(__name__)

# Color scheme: Black, White, Dark Grey, Dark Green (#0d4f01)
COLOR_SCHEME = {
    'primary': '#0d4f01',      # Dark Green
    'secondary': '#2d2d2d',    # Dark Grey
    'background': '#ffffff',   # White
    'text': '#000000',         # Black
    'accent': '#4a4a4a'        # Medium Grey
}


class RetrainingReporter:
    """
    Generates comprehensive retraining comparison reports and visualizations.

    This class creates detailed reports comparing model performance before
    and after retraining, including metrics analysis, improvement tracking,
    and visual comparisons using seaborn with the specified color scheme.
    """

    def __init__(self):
        """Initialize the retraining reporter."""
        # Set up matplotlib and seaborn styling
        self._setup_plotting_style()

        logger.info("RetrainingReporter initialized with custom color scheme")

    def _setup_plotting_style(self) -> None:
        """Set up consistent plotting style using the specified color scheme."""
        # Set seaborn style
        sns.set_style("whitegrid")

        # Set color palette
        colors = [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'],
                 COLOR_SCHEME['accent'], COLOR_SCHEME['text']]
        sns.set_palette(colors)

        # Set matplotlib parameters
        plt.rcParams.update({
            'figure.facecolor': COLOR_SCHEME['background'],
            'axes.facecolor': COLOR_SCHEME['background'],
            'text.color': COLOR_SCHEME['text'],
            'axes.labelcolor': COLOR_SCHEME['text'],
            'xtick.color': COLOR_SCHEME['text'],
            'ytick.color': COLOR_SCHEME['text'],
            'axes.edgecolor': COLOR_SCHEME['secondary'],
            'grid.color': COLOR_SCHEME['accent'],
            'grid.alpha': 0.3
        })

    def generate_retraining_comparison_report(self,
                                            retraining_result: RetrainingResult) -> Dict[str, Any]:
        """
        Generate comprehensive retraining comparison report.

        Args:
            retraining_result: Result from retraining operation

        Returns:
            Dictionary containing comprehensive report data
        """
        logger.info("Generating retraining comparison report for trigger: %s",
                   retraining_result.trigger_event.reason.value)

        report = {
            'summary': self._generate_summary_report(retraining_result),
            'clustering_comparison': self._generate_clustering_comparison(retraining_result),
            'performance_analysis': self._generate_performance_analysis(retraining_result),
            'improvement_metrics': self._generate_improvement_metrics(retraining_result),
            'trigger_analysis': self._generate_trigger_analysis(retraining_result),
            'recommendations': self._generate_recommendations(retraining_result),
            'timestamp': datetime.now(),
            'retraining_timestamp': retraining_result.timestamp
        }

        logger.info("Retraining comparison report generated successfully")
        return report

    def create_retraining_visualizations(self,
                                       retraining_result: RetrainingResult,
                                       save_path: str) -> Dict[str, str]:
        """
        Create before/after retraining visualizations using seaborn.

        Args:
            retraining_result: Result from retraining operation
            save_path: Directory path to save visualizations

        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("Creating retraining visualizations at: %s", save_path)

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        visualization_paths = {}

        try:
            # 1. Metrics comparison chart
            metrics_path = self._create_metrics_comparison_chart(
                retraining_result, save_dir / "metrics_comparison.png"
            )
            visualization_paths['metrics_comparison'] = str(metrics_path)

            # 2. Cluster distribution comparison
            cluster_dist_path = self._create_cluster_distribution_comparison(
                retraining_result, save_dir / "cluster_distribution_comparison.png"
            )
            visualization_paths['cluster_distribution'] = str(cluster_dist_path)

            # 3. Improvement analysis chart
            improvement_path = self._create_improvement_analysis_chart(
                retraining_result, save_dir / "improvement_analysis.png"
            )
            visualization_paths['improvement_analysis'] = str(improvement_path)

            # 4. Trigger analysis visualization
            trigger_path = self._create_trigger_analysis_visualization(
                retraining_result, save_dir / "trigger_analysis.png"
            )
            visualization_paths['trigger_analysis'] = str(trigger_path)

            # 5. Performance timeline (if multiple retraining results available)
            timeline_path = self._create_performance_timeline(
                [retraining_result], save_dir / "performance_timeline.png"
            )
            visualization_paths['performance_timeline'] = str(timeline_path)

            logger.info("Created %d retraining visualizations", len(visualization_paths))
            return visualization_paths

        except Exception as e:
            logger.error("Failed to create retraining visualizations: %s", str(e))
            return visualization_paths

    def generate_multi_retraining_analysis(self,
                                         retraining_results: List[RetrainingResult]) -> Dict[str, Any]:
        """
        Generate analysis across multiple retraining events.

        Args:
            retraining_results: List of retraining results to analyze

        Returns:
            Dictionary containing multi-retraining analysis
        """
        if not retraining_results:
            return {'error': 'No retraining results provided'}

        logger.info("Generating multi-retraining analysis for %d events", len(retraining_results))

        analysis = {
            'total_retrainings': len(retraining_results),
            'time_span': self._calculate_time_span(retraining_results),
            'trigger_frequency': self._analyze_trigger_frequency(retraining_results),
            'performance_trends': self._analyze_performance_trends(retraining_results),
            'improvement_patterns': self._analyze_improvement_patterns(retraining_results),
            'stability_analysis': self._analyze_model_stability(retraining_results),
            'recommendations': self._generate_multi_retraining_recommendations(retraining_results)
        }

        return analysis

    def _generate_summary_report(self, retraining_result: RetrainingResult) -> Dict[str, Any]:
        """Generate summary section of the retraining report."""
        trigger_event = retraining_result.trigger_event
        metrics = retraining_result.retraining_metrics

        summary = {
            'trigger_reason': trigger_event.reason.value,
            'trigger_timestamp': trigger_event.timestamp,
            'retraining_duration': metrics.get('retraining_duration_seconds', 0),
            'data_size': metrics.get('expanded_window_size', 0),
            'cluster_change': {
                'old_count': metrics.get('old_cluster_count', 0),
                'new_count': metrics.get('new_cluster_count', 0),
                'change': metrics.get('new_cluster_count', 0) - metrics.get('old_cluster_count', 0)
            },
            'noise_ratio_change': {
                'old_ratio': metrics.get('old_noise_ratio', 0),
                'new_ratio': metrics.get('new_noise_ratio', 0),
                'change': metrics.get('new_noise_ratio', 0) - metrics.get('old_noise_ratio', 0)
            },
            'stability_score': metrics.get('cluster_stability_score', 0)
        }

        return summary

    def _generate_clustering_comparison(self, retraining_result: RetrainingResult) -> Dict[str, Any]:
        """Generate clustering-specific comparison metrics."""
        comparison = retraining_result.model_comparison

        clustering_comparison = {
            'cluster_count_analysis': {
                'old_clusters': comparison['old_metrics'].get('n_clusters', 0),
                'new_clusters': comparison['new_metrics'].get('n_clusters', 0),
                'optimal_change': self._assess_cluster_count_change(comparison)
            },
            'noise_analysis': {
                'old_noise_ratio': comparison['old_metrics'].get('noise_ratio', 0),
                'new_noise_ratio': comparison['new_metrics'].get('noise_ratio', 0),
                'noise_reduction': comparison['old_metrics'].get('noise_ratio', 0) -
                                 comparison['new_metrics'].get('noise_ratio', 0)
            },
            'quality_metrics': {
                'silhouette_improvement': comparison.get('silhouette_improvement', 0),
                'calinski_harabasz_improvement': comparison.get('calinski_harabasz_improvement', 0),
                'davies_bouldin_improvement': comparison.get('davies_bouldin_improvement', 0)
            }
        }

        return clustering_comparison

    def _generate_performance_analysis(self, retraining_result: RetrainingResult) -> Dict[str, Any]:
        """Generate performance analysis section."""
        comparison = retraining_result.model_comparison

        performance_analysis = {
            'overall_improvement_ratio': comparison.get('overall_improvement_ratio', 0),
            'significant_improvements': len(comparison.get('improvements', {})),
            'significant_degradations': len(comparison.get('degradations', {})),
            'net_improvement_score': self._calculate_net_improvement_score(comparison),
            'performance_categories': {
                'clustering_quality': self._assess_clustering_quality_change(comparison),
                'noise_handling': self._assess_noise_handling_change(comparison),
                'model_stability': self._assess_model_stability_change(retraining_result)
            }
        }

        return performance_analysis

    def _generate_improvement_metrics(self, retraining_result: RetrainingResult) -> Dict[str, Any]:
        """Generate detailed improvement metrics."""
        comparison = retraining_result.model_comparison

        improvements = comparison.get('improvements', {})
        degradations = comparison.get('degradations', {})

        improvement_metrics = {
            'improvements': improvements,
            'degradations': degradations,
            'improvement_magnitude': sum(improvements.values()) if improvements else 0,
            'degradation_magnitude': sum(degradations.values()) if degradations else 0,
            'net_improvement': (sum(improvements.values()) if improvements else 0) -
                             (sum(degradations.values()) if degradations else 0),
            'improvement_categories': self._categorize_improvements(improvements, degradations)
        }

        return improvement_metrics

    def _generate_trigger_analysis(self, retraining_result: RetrainingResult) -> Dict[str, Any]:
        """Generate analysis of the trigger event."""
        trigger_event = retraining_result.trigger_event

        trigger_analysis = {
            'trigger_type': trigger_event.reason.value,
            'trigger_value': trigger_event.trigger_value,
            'threshold_value': trigger_event.threshold_value,
            'trigger_severity': self._assess_trigger_severity(trigger_event),
            'trigger_effectiveness': self._assess_trigger_effectiveness(retraining_result),
            'metadata': trigger_event.metadata or {}
        }

        return trigger_analysis

    def _generate_recommendations(self, retraining_result: RetrainingResult) -> List[str]:
        """Generate recommendations based on retraining results."""
        recommendations = []

        comparison = retraining_result.model_comparison
        metrics = retraining_result.retraining_metrics

        # Cluster count recommendations
        cluster_change = metrics.get('new_cluster_count', 0) - metrics.get('old_cluster_count', 0)
        if cluster_change > 2:
            recommendations.append(
                "Consider increasing min_cluster_size parameter to reduce cluster fragmentation"
            )
        elif cluster_change < -2:
            recommendations.append(
                "Consider decreasing min_cluster_size parameter to capture more granular patterns"
            )

        # Noise ratio recommendations
        noise_change = metrics.get('new_noise_ratio', 0) - metrics.get('old_noise_ratio', 0)
        if noise_change > 0.1:
            recommendations.append(
                "Noise ratio increased significantly; consider adjusting HDBSCAN parameters"
            )
        elif noise_change < -0.1:
            recommendations.append(
                "Good noise reduction achieved; current parameters are working well"
            )

        # Performance recommendations
        overall_improvement = comparison.get('overall_improvement_ratio', 0)
        if overall_improvement < 0.3:
            recommendations.append(
                "Limited improvement from retraining; consider adjusting trigger thresholds"
            )
        elif overall_improvement > 0.7:
            recommendations.append(
                "Excellent improvement from retraining; current configuration is optimal"
            )

        # Stability recommendations
        stability_score = metrics.get('cluster_stability_score', 0)
        if stability_score < 0.5:
            recommendations.append(
                "Low cluster stability; consider increasing expanding window size"
            )

        return recommendations

    def _create_metrics_comparison_chart(self,
                                       retraining_result: RetrainingResult,
                                       save_path: Path) -> Path:
        """Create metrics comparison chart."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison: Before vs After Retraining',
                    fontsize=16, color=COLOR_SCHEME['text'])

        comparison = retraining_result.model_comparison
        old_metrics = comparison['old_metrics']
        new_metrics = comparison['new_metrics']

        # 1. Cluster count comparison
        ax1 = axes[0, 0]
        clusters = ['Old Model', 'New Model']
        cluster_counts = [old_metrics.get('n_clusters', 0), new_metrics.get('n_clusters', 0)]
        bars1 = ax1.bar(clusters, cluster_counts, color=[COLOR_SCHEME['secondary'], COLOR_SCHEME['primary']])
        ax1.set_title('Number of Clusters')
        ax1.set_ylabel('Cluster Count')
        for bar, count in zip(bars1, cluster_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')

        # 2. Noise ratio comparison
        ax2 = axes[0, 1]
        noise_ratios = [old_metrics.get('noise_ratio', 0), new_metrics.get('noise_ratio', 0)]
        bars2 = ax2.bar(clusters, noise_ratios, color=[COLOR_SCHEME['secondary'], COLOR_SCHEME['primary']])
        ax2.set_title('Noise Ratio')
        ax2.set_ylabel('Noise Ratio')
        ax2.set_ylim(0, max(noise_ratios) * 1.2 if noise_ratios else 1)
        for bar, ratio in zip(bars2, noise_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ratio:.3f}', ha='center', va='bottom')

        # 3. Silhouette score comparison
        ax3 = axes[1, 0]
        silhouette_scores = [old_metrics.get('silhouette_score'), new_metrics.get('silhouette_score')]
        valid_scores = [score for score in silhouette_scores if score is not None]
        if valid_scores:
            bars3 = ax3.bar(clusters, [score if score is not None else 0 for score in silhouette_scores],
                           color=[COLOR_SCHEME['secondary'], COLOR_SCHEME['primary']])
            ax3.set_title('Silhouette Score')
            ax3.set_ylabel('Silhouette Score')
            for bar, score in zip(bars3, silhouette_scores):
                if score is not None:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No silhouette scores available', ha='center', va='center',
                    transform=ax3.transAxes)
            ax3.set_title('Silhouette Score (N/A)')

        # 4. Overall improvement ratio
        ax4 = axes[1, 1]
        improvement_ratio = comparison.get('overall_improvement_ratio', 0)
        colors = [COLOR_SCHEME['primary'] if improvement_ratio > 0.5 else COLOR_SCHEME['secondary']]
        bars4 = ax4.bar(['Overall Improvement'], [improvement_ratio], color=colors)
        ax4.set_title('Overall Improvement Ratio')
        ax4.set_ylabel('Improvement Ratio')
        ax4.set_ylim(0, 1)
        ax4.text(bars4[0].get_x() + bars4[0].get_width()/2, bars4[0].get_height() + 0.02,
                f'{improvement_ratio:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLOR_SCHEME['background'])
        plt.close()

        return save_path

    def _create_cluster_distribution_comparison(self,
                                              retraining_result: RetrainingResult,
                                              save_path: Path) -> Path:
        """Create cluster distribution comparison visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Cluster Size Distribution: Before vs After Retraining',
                    fontsize=16, color=COLOR_SCHEME['text'])

        reassignment = retraining_result.reassignment_summary
        cluster_sizes = reassignment.get('cluster_sizes', {})

        # Filter out noise points (-1) for distribution analysis
        valid_clusters = {k: v for k, v in cluster_sizes.items() if k != -1}

        if valid_clusters:
            cluster_ids = list(valid_clusters.keys())
            sizes = list(valid_clusters.values())

            # Left plot: Bar chart of cluster sizes
            bars = ax1.bar(range(len(cluster_ids)), sizes, color=COLOR_SCHEME['primary'])
            ax1.set_title('New Cluster Sizes')
            ax1.set_xlabel('Cluster ID')
            ax1.set_ylabel('Number of Points')
            ax1.set_xticks(range(len(cluster_ids)))
            ax1.set_xticklabels([f'C{cid}' for cid in cluster_ids])

            # Add value labels on bars
            for bar, size in zip(bars, sizes):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes) * 0.01,
                        str(size), ha='center', va='bottom')

            # Right plot: Distribution histogram
            ax2.hist(sizes, bins=min(10, len(sizes)), color=COLOR_SCHEME['primary'], alpha=0.7, edgecolor=COLOR_SCHEME['text'])
            ax2.set_title('Cluster Size Distribution')
            ax2.set_xlabel('Cluster Size')
            ax2.set_ylabel('Frequency')

            # Add statistics
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            ax2.axvline(mean_size, color=COLOR_SCHEME['secondary'], linestyle='--',
                       label=f'Mean: {mean_size:.1f}')
            ax2.legend()
        else:
            ax1.text(0.5, 0.5, 'No valid clusters found', ha='center', va='center',
                    transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No distribution data available', ha='center', va='center',
                    transform=ax2.transAxes)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLOR_SCHEME['background'])
        plt.close()

        return save_path

    def _create_improvement_analysis_chart(self,
                                         retraining_result: RetrainingResult,
                                         save_path: Path) -> Path:
        """Create improvement analysis chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Retraining Improvement Analysis', fontsize=16, color=COLOR_SCHEME['text'])

        comparison = retraining_result.model_comparison
        improvements = comparison.get('improvements', {})
        degradations = comparison.get('degradations', {})

        # Left plot: Improvements vs Degradations
        categories = ['Improvements', 'Degradations']
        counts = [len(improvements), len(degradations)]
        colors = [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary']]

        bars1 = ax1.bar(categories, counts, color=colors)
        ax1.set_title('Metric Changes Count')
        ax1.set_ylabel('Number of Metrics')

        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')

        # Right plot: Magnitude of changes
        if improvements or degradations:
            improvement_mag = sum(improvements.values()) if improvements else 0
            degradation_mag = sum(degradations.values()) if degradations else 0

            magnitudes = [improvement_mag, degradation_mag]
            bars2 = ax2.bar(categories, magnitudes, color=colors)
            ax2.set_title('Change Magnitude')
            ax2.set_ylabel('Total Change Magnitude')

            for bar, mag in zip(bars2, magnitudes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(magnitudes) * 0.01,
                        f'{mag:.3f}', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No change data available', ha='center', va='center',
                    transform=ax2.transAxes)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLOR_SCHEME['background'])
        plt.close()

        return save_path

    def _create_trigger_analysis_visualization(self,
                                             retraining_result: RetrainingResult,
                                             save_path: Path) -> Path:
        """Create trigger analysis visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Retraining Trigger Analysis', fontsize=16, color=COLOR_SCHEME['text'])

        trigger_event = retraining_result.trigger_event

        # Create a simple visualization showing trigger information
        trigger_info = [
            f"Trigger: {trigger_event.reason.value}",
            f"Timestamp: {trigger_event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Trigger Value: {trigger_event.trigger_value:.4f}" if trigger_event.trigger_value else "N/A",
            f"Threshold: {trigger_event.threshold_value:.4f}" if trigger_event.threshold_value else "N/A"
        ]

        # Create a text-based visualization
        ax.text(0.1, 0.7, '\n'.join(trigger_info), fontsize=12,
               verticalalignment='top', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_SCHEME['background'],
                        edgecolor=COLOR_SCHEME['primary']))

        # Add effectiveness assessment
        effectiveness = self._assess_trigger_effectiveness(retraining_result)
        effectiveness_text = f"Trigger Effectiveness: {effectiveness}"
        ax.text(0.1, 0.3, effectiveness_text, fontsize=12,
               transform=ax.transAxes, color=COLOR_SCHEME['primary'])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLOR_SCHEME['background'])
        plt.close()

        return save_path

    def _create_performance_timeline(self,
                                   retraining_results: List[RetrainingResult],
                                   save_path: Path) -> Path:
        """Create performance timeline visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('Performance Timeline', fontsize=16, color=COLOR_SCHEME['text'])

        if len(retraining_results) > 1:
            timestamps = [result.timestamp for result in retraining_results]
            improvement_ratios = [result.model_comparison.get('overall_improvement_ratio', 0)
                                for result in retraining_results]

            ax.plot(timestamps, improvement_ratios, marker='o',
                   color=COLOR_SCHEME['primary'], linewidth=2, markersize=8)
            ax.set_title('Overall Improvement Ratio Over Time')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Improvement Ratio')
            ax.grid(True, alpha=0.3)

            # Add horizontal line at 0.5 (neutral improvement)
            ax.axhline(y=0.5, color=COLOR_SCHEME['secondary'], linestyle='--', alpha=0.7)

        else:
            ax.text(0.5, 0.5, 'Insufficient data for timeline\n(Need multiple retraining events)',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLOR_SCHEME['background'])
        plt.close()

        return save_path

    # Helper methods for analysis
    def _assess_cluster_count_change(self, comparison: Dict[str, Any]) -> str:
        """Assess whether cluster count change is optimal."""
        old_count = comparison['old_metrics'].get('n_clusters', 0)
        new_count = comparison['new_metrics'].get('n_clusters', 0)
        change = new_count - old_count

        if abs(change) <= 1:
            return "Stable"
        elif change > 1:
            return "Increased (may indicate data complexity)"
        else:
            return "Decreased (may indicate better consolidation)"

    def _calculate_net_improvement_score(self, comparison: Dict[str, Any]) -> float:
        """Calculate net improvement score."""
        improvements = comparison.get('improvements', {})
        degradations = comparison.get('degradations', {})

        improvement_sum = sum(improvements.values()) if improvements else 0
        degradation_sum = sum(degradations.values()) if degradations else 0

        return improvement_sum - degradation_sum

    def _assess_clustering_quality_change(self, comparison: Dict[str, Any]) -> str:
        """Assess clustering quality change."""
        silhouette_change = comparison.get('silhouette_improvement', 0)

        if silhouette_change > 0.05:
            return "Significantly Improved"
        elif silhouette_change > 0.01:
            return "Slightly Improved"
        elif silhouette_change > -0.01:
            return "Stable"
        elif silhouette_change > -0.05:
            return "Slightly Degraded"
        else:
            return "Significantly Degraded"

    def _assess_noise_handling_change(self, comparison: Dict[str, Any]) -> str:
        """Assess noise handling change."""
        noise_change = comparison.get('noise_ratio_change', 0)

        if noise_change < -0.05:
            return "Significantly Better"
        elif noise_change < -0.01:
            return "Slightly Better"
        elif noise_change < 0.01:
            return "Stable"
        elif noise_change < 0.05:
            return "Slightly Worse"
        else:
            return "Significantly Worse"

    def _assess_model_stability_change(self, retraining_result: RetrainingResult) -> str:
        """Assess model stability change."""
        stability_score = retraining_result.retraining_metrics.get('cluster_stability_score', 0)

        if stability_score > 0.8:
            return "Highly Stable"
        elif stability_score > 0.6:
            return "Moderately Stable"
        elif stability_score > 0.4:
            return "Somewhat Unstable"
        else:
            return "Unstable"

    def _categorize_improvements(self, improvements: Dict[str, Any], degradations: Dict[str, Any]) -> Dict[str, str]:
        """Categorize improvements by type."""
        categories = {}

        for metric in improvements:
            if 'silhouette' in metric:
                categories['clustering_quality'] = 'Improved'
            elif 'noise' in metric:
                categories['noise_handling'] = 'Improved'

        for metric in degradations:
            if 'silhouette' in metric:
                categories['clustering_quality'] = 'Degraded'
            elif 'noise' in metric:
                categories['noise_handling'] = 'Degraded'

        return categories

    def _assess_trigger_severity(self, trigger_event: RetrainingEvent) -> str:
        """Assess the severity of the trigger event."""
        if trigger_event.reason == TriggerReason.MANUAL_COMMAND:
            return "Manual"
        elif trigger_event.trigger_value and trigger_event.threshold_value:
            ratio = trigger_event.trigger_value / trigger_event.threshold_value
            if ratio > 1.5:
                return "Severe"
            elif ratio > 1.2:
                return "Moderate"
            else:
                return "Mild"
        else:
            return "Unknown"

    def _assess_trigger_effectiveness(self, retraining_result: RetrainingResult) -> str:
        """Assess the effectiveness of the trigger."""
        improvement_ratio = retraining_result.model_comparison.get('overall_improvement_ratio', 0)

        if improvement_ratio > 0.7:
            return "Highly Effective"
        elif improvement_ratio > 0.5:
            return "Moderately Effective"
        elif improvement_ratio > 0.3:
            return "Somewhat Effective"
        else:
            return "Low Effectiveness"

    def _calculate_time_span(self, retraining_results: List[RetrainingResult]) -> Dict[str, Any]:
        """Calculate time span across retraining events."""
        if len(retraining_results) < 2:
            return {'total_days': 0, 'average_interval_days': 0}

        timestamps = [result.timestamp for result in retraining_results]
        timestamps.sort()

        total_span = (timestamps[-1] - timestamps[0]).days
        average_interval = total_span / (len(timestamps) - 1) if len(timestamps) > 1 else 0

        return {
            'total_days': total_span,
            'average_interval_days': average_interval,
            'first_retraining': timestamps[0],
            'last_retraining': timestamps[-1]
        }

    def _analyze_trigger_frequency(self, retraining_results: List[RetrainingResult]) -> Dict[str, Any]:
        """Analyze frequency of different trigger types."""
        trigger_counts = {}
        for result in retraining_results:
            reason = result.trigger_event.reason.value
            trigger_counts[reason] = trigger_counts.get(reason, 0) + 1

        return {
            'trigger_counts': trigger_counts,
            'most_common_trigger': max(trigger_counts.items(), key=lambda x: x[1])[0] if trigger_counts else None,
            'total_triggers': len(retraining_results)
        }

    def _analyze_performance_trends(self, retraining_results: List[RetrainingResult]) -> Dict[str, Any]:
        """Analyze performance trends across retraining events."""
        improvement_ratios = [result.model_comparison.get('overall_improvement_ratio', 0)
                            for result in retraining_results]

        if len(improvement_ratios) < 2:
            return {'trend': 'Insufficient data'}

        # Simple trend analysis
        recent_avg = np.mean(improvement_ratios[-3:]) if len(improvement_ratios) >= 3 else improvement_ratios[-1]
        early_avg = np.mean(improvement_ratios[:3]) if len(improvement_ratios) >= 3 else improvement_ratios[0]

        trend = "Improving" if recent_avg > early_avg else "Declining" if recent_avg < early_avg else "Stable"

        return {
            'trend': trend,
            'average_improvement': np.mean(improvement_ratios),
            'improvement_std': np.std(improvement_ratios),
            'recent_average': recent_avg,
            'early_average': early_avg
        }

    def _analyze_improvement_patterns(self, retraining_results: List[RetrainingResult]) -> Dict[str, Any]:
        """Analyze patterns in improvements."""
        successful_retrainings = [result for result in retraining_results
                                if result.model_comparison.get('overall_improvement_ratio', 0) > 0.5]

        return {
            'success_rate': len(successful_retrainings) / len(retraining_results) if retraining_results else 0,
            'successful_count': len(successful_retrainings),
            'total_count': len(retraining_results),
            'average_successful_improvement': np.mean([
                result.model_comparison.get('overall_improvement_ratio', 0)
                for result in successful_retrainings
            ]) if successful_retrainings else 0
        }

    def _analyze_model_stability(self, retraining_results: List[RetrainingResult]) -> Dict[str, Any]:
        """Analyze model stability across retraining events."""
        stability_scores = [result.retraining_metrics.get('cluster_stability_score', 0)
                          for result in retraining_results]

        return {
            'average_stability': np.mean(stability_scores) if stability_scores else 0,
            'stability_std': np.std(stability_scores) if stability_scores else 0,
            'min_stability': min(stability_scores) if stability_scores else 0,
            'max_stability': max(stability_scores) if stability_scores else 0,
            'stability_trend': self._calculate_stability_trend(stability_scores)
        }

    def _calculate_stability_trend(self, stability_scores: List[float]) -> str:
        """Calculate stability trend."""
        if len(stability_scores) < 2:
            return "Insufficient data"

        recent_avg = np.mean(stability_scores[-3:]) if len(stability_scores) >= 3 else stability_scores[-1]
        early_avg = np.mean(stability_scores[:3]) if len(stability_scores) >= 3 else stability_scores[0]

        if recent_avg > early_avg + 0.1:
            return "Improving"
        elif recent_avg < early_avg - 0.1:
            return "Declining"
        else:
            return "Stable"

    def _generate_multi_retraining_recommendations(self, retraining_results: List[RetrainingResult]) -> List[str]:
        """Generate recommendations based on multiple retraining events."""
        recommendations = []

        if not retraining_results:
            return ["No retraining data available for recommendations"]

        # Analyze success rate
        success_rate = len([r for r in retraining_results
                          if r.model_comparison.get('overall_improvement_ratio', 0) > 0.5]) / len(retraining_results)

        if success_rate < 0.3:
            recommendations.append("Low retraining success rate; consider adjusting trigger thresholds or parameters")
        elif success_rate > 0.8:
            recommendations.append("High retraining success rate; current configuration is working well")

        # Analyze trigger frequency
        trigger_analysis = self._analyze_trigger_frequency(retraining_results)
        most_common = trigger_analysis.get('most_common_trigger')

        if most_common == 'noise_threshold_exceeded':
            recommendations.append("Frequent noise-triggered retraining; consider optimizing HDBSCAN parameters")
        elif most_common == 'manual_command':
            recommendations.append("Frequent manual retraining; consider lowering automatic trigger thresholds")

        # Analyze stability trends
        stability_analysis = self._analyze_model_stability(retraining_results)
        if stability_analysis['average_stability'] < 0.5:
            recommendations.append("Low average model stability; consider increasing expanding window size")

        return recommendations