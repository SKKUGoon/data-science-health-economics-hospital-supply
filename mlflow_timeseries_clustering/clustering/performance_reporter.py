"""
Performance reporting for clustering results.

This module provides visualization and reporting capabilities for HDBSCAN
clustering results using seaborn with a custom color scheme.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from ..core.data_models import ClusteringResult
from ..core.exceptions import PipelineError


logger = logging.getLogger(__name__)

# Custom color scheme: Black, White, Dark Grey, Dark Green
CUSTOM_COLORS = {
    'black': '#000000',
    'white': '#FFFFFF',
    'dark_grey': '#404040',
    'dark_green': '#0d4f01'
}

# Color palette for visualizations
CLUSTER_PALETTE = [CUSTOM_COLORS['dark_green'], CUSTOM_COLORS['dark_grey'],
                  CUSTOM_COLORS['black'], '#2d7f02', '#606060', '#1a1a1a']


class ClusteringPerformanceReporter:
    """
    Performance reporter for clustering results with seaborn visualizations.

    This class generates comprehensive reports and visualizations for HDBSCAN
    clustering results using a custom color scheme.
    """

    def __init__(self):
        """Initialize the performance reporter."""
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = CUSTOM_COLORS['white']
        plt.rcParams['axes.facecolor'] = CUSTOM_COLORS['white']

        logger.info("ClusteringPerformanceReporter initialized")

    def generate_clustering_report(self,
                                 clustering_result: ClusteringResult,
                                 X: np.ndarray,
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive clustering performance report.

        Args:
            clustering_result: Results from HDBSCAN clustering
            X: Original input data used for clustering
            save_path: Optional path to save visualizations

        Returns:
            Dictionary containing report data and visualization paths
        """
        logger.info("Generating clustering performance report")

        try:
            report = {
                'metrics': clustering_result.metrics.copy(),
                'summary': self._generate_summary_stats(clustering_result),
                'visualizations': {}
            }

            # Generate visualizations
            if save_path:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)

                # Cluster distribution plot
                dist_path = self._create_cluster_distribution_plot(
                    clustering_result, save_dir / "cluster_distribution.png"
                )
                report['visualizations']['cluster_distribution'] = str(dist_path)

                # Silhouette analysis (if applicable)
                if clustering_result.metrics.get('silhouette_score') is not None:
                    silhouette_path = self._create_silhouette_analysis(
                        X, clustering_result, save_dir / "silhouette_analysis.png"
                    )
                    report['visualizations']['silhouette_analysis'] = str(silhouette_path)

                # Cluster scatter plot (for 2D visualization)
                if X.shape[1] >= 2:
                    scatter_path = self._create_cluster_scatter_plot(
                        X, clustering_result, save_dir / "cluster_scatter.png"
                    )
                    report['visualizations']['cluster_scatter'] = str(scatter_path)

                # Metrics summary plot
                metrics_path = self._create_metrics_summary_plot(
                    clustering_result.metrics, save_dir / "metrics_summary.png"
                )
                report['visualizations']['metrics_summary'] = str(metrics_path)

            logger.info("Clustering performance report generated successfully")
            return report

        except Exception as e:
            raise PipelineError(
                f"Failed to generate clustering report: {str(e)}",
                component="ClusteringReporter"
            ) from e

    def _generate_summary_stats(self, clustering_result: ClusteringResult) -> Dict[str, Any]:
        """
        Generate summary statistics for clustering results.

        Args:
            clustering_result: Clustering results

        Returns:
            Dictionary of summary statistics
        """
        labels = clustering_result.labels
        unique_labels = np.unique(labels)

        # Remove noise label if present
        cluster_labels = unique_labels[unique_labels != -1]

        summary = {
            'total_points': len(labels),
            'n_clusters': len(cluster_labels),
            'n_noise_points': np.sum(labels == -1),
            'noise_percentage': (np.sum(labels == -1) / len(labels)) * 100,
            'cluster_sizes': {}
        }

        # Calculate cluster sizes
        for label in cluster_labels:
            cluster_size = np.sum(labels == label)
            summary['cluster_sizes'][f'cluster_{label}'] = cluster_size

        # Add noise points if any
        if summary['n_noise_points'] > 0:
            summary['cluster_sizes']['noise_points'] = summary['n_noise_points']

        return summary

    def _create_cluster_distribution_plot(self,
                                        clustering_result: ClusteringResult,
                                        save_path: Path) -> Path:
        """
        Create cluster distribution bar plot.

        Args:
            clustering_result: Clustering results
            save_path: Path to save the plot

        Returns:
            Path where plot was saved
        """
        labels = clustering_result.labels
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Create DataFrame for plotting
        df_data = []
        for label, count in zip(unique_labels, counts):
            cluster_name = 'Noise' if label == -1 else f'Cluster {label}'
            df_data.append({'Cluster': cluster_name, 'Count': count})

        df = pd.DataFrame(df_data)

        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x='Cluster', y='Count',
                        palette=CLUSTER_PALETTE[:len(df)])

        plt.title('Cluster Distribution', fontsize=16, fontweight='bold',
                 color=CUSTOM_COLORS['black'])
        plt.xlabel('Cluster', fontsize=12, color=CUSTOM_COLORS['black'])
        plt.ylabel('Number of Points', fontsize=12, color=CUSTOM_COLORS['black'])

        # Add value labels on bars
        for i, v in enumerate(df['Count']):
            ax.text(i, v + max(counts) * 0.01, str(v),
                   ha='center', va='bottom', fontweight='bold',
                   color=CUSTOM_COLORS['black'])

        plt.xticks(rotation=45, color=CUSTOM_COLORS['black'])
        plt.yticks(color=CUSTOM_COLORS['black'])
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=CUSTOM_COLORS['white'])
        plt.close()

        logger.info("Cluster distribution plot saved to %s", save_path)
        return save_path

    def _create_silhouette_analysis(self,
                                  X: np.ndarray,
                                  clustering_result: ClusteringResult,
                                  save_path: Path) -> Path:
        """
        Create silhouette analysis plot.

        Args:
            X: Original input data
            clustering_result: Clustering results
            save_path: Path to save the plot

        Returns:
            Path where plot was saved
        """
        from sklearn.metrics import silhouette_samples

        labels = clustering_result.labels

        # Filter out noise points for silhouette analysis
        non_noise_mask = labels != -1
        X_non_noise = X[non_noise_mask]
        labels_non_noise = labels[non_noise_mask]

        if len(np.unique(labels_non_noise)) < 2:
            logger.warning("Insufficient clusters for silhouette analysis")
            # Create empty plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Insufficient clusters for silhouette analysis',
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=14, color=CUSTOM_COLORS['black'])
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=CUSTOM_COLORS['white'])
            plt.close()
            return save_path

        # Calculate silhouette scores
        sample_silhouette_values = silhouette_samples(X_non_noise, labels_non_noise)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        y_lower = 10
        unique_labels = np.unique(labels_non_noise)

        for i, label in enumerate(unique_labels):
            cluster_silhouette_values = sample_silhouette_values[labels_non_noise == label]
            cluster_silhouette_values.sort()

            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                           0, cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label),
                   color=CUSTOM_COLORS['black'], fontweight='bold')

            y_lower = y_upper + 10

        ax.set_xlabel('Silhouette Coefficient Values', color=CUSTOM_COLORS['black'])
        ax.set_ylabel('Cluster Label', color=CUSTOM_COLORS['black'])
        ax.set_title('Silhouette Analysis', fontsize=16, fontweight='bold',
                    color=CUSTOM_COLORS['black'])

        # Add average silhouette score line
        avg_score = clustering_result.metrics.get('silhouette_score', 0)
        ax.axvline(x=avg_score, color=CUSTOM_COLORS['dark_green'],
                  linestyle='--', linewidth=2,
                  label=f'Average Score: {avg_score:.3f}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=CUSTOM_COLORS['white'])
        plt.close()

        logger.info("Silhouette analysis plot saved to %s", save_path)
        return save_path

    def _create_cluster_scatter_plot(self,
                                   X: np.ndarray,
                                   clustering_result: ClusteringResult,
                                   save_path: Path) -> Path:
        """
        Create 2D scatter plot of clusters.

        Args:
            X: Original input data
            clustering_result: Clustering results
            save_path: Path to save the plot

        Returns:
            Path where plot was saved
        """
        labels = clustering_result.labels

        # Use first two dimensions for visualization
        X_plot = X[:, :2]

        plt.figure(figsize=(10, 8))

        # Plot each cluster
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Plot noise points
                mask = labels == label
                plt.scatter(X_plot[mask, 0], X_plot[mask, 1],
                          c=CUSTOM_COLORS['dark_grey'], marker='x', s=50,
                          alpha=0.6, label='Noise')
            else:
                # Plot cluster points
                mask = labels == label
                color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                plt.scatter(X_plot[mask, 0], X_plot[mask, 1],
                          c=color, s=50, alpha=0.7,
                          label=f'Cluster {label}')

        # Plot cluster centers if available
        if clustering_result.cluster_centers.size > 0:
            centers_2d = clustering_result.cluster_centers[:, :2]
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
                       c=CUSTOM_COLORS['black'], marker='*', s=200,
                       edgecolors=CUSTOM_COLORS['white'], linewidth=2,
                       label='Centers')

        plt.xlabel('Feature 1', color=CUSTOM_COLORS['black'])
        plt.ylabel('Feature 2', color=CUSTOM_COLORS['black'])
        plt.title('Cluster Visualization (First 2 Dimensions)',
                 fontsize=16, fontweight='bold', color=CUSTOM_COLORS['black'])
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=CUSTOM_COLORS['white'])
        plt.close()

        logger.info("Cluster scatter plot saved to %s", save_path)
        return save_path

    def _create_metrics_summary_plot(self,
                                   metrics: Dict[str, Any],
                                   save_path: Path) -> Path:
        """
        Create summary plot of clustering metrics.

        Args:
            metrics: Dictionary of clustering metrics
            save_path: Path to save the plot

        Returns:
            Path where plot was saved
        """
        # Prepare metrics for plotting
        plot_metrics = {}

        # Add basic metrics
        plot_metrics['Number of Clusters'] = metrics.get('n_clusters', 0)
        plot_metrics['Noise Points'] = metrics.get('n_noise_points', 0)
        plot_metrics['Total Points'] = metrics.get('total_points', 0)

        # Add quality metrics if available
        quality_metrics = {}
        if metrics.get('silhouette_score') is not None:
            quality_metrics['Silhouette Score'] = metrics['silhouette_score']
        if metrics.get('calinski_harabasz_score') is not None:
            quality_metrics['Calinski-Harabasz Score'] = metrics['calinski_harabasz_score']
        if metrics.get('davies_bouldin_score') is not None:
            quality_metrics['Davies-Bouldin Score'] = metrics['davies_bouldin_score']

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Basic metrics bar plot
        basic_df = pd.DataFrame(list(plot_metrics.items()),
                               columns=['Metric', 'Value'])
        sns.barplot(data=basic_df, x='Metric', y='Value',
                   palette=[CLUSTER_PALETTE[0]] * len(basic_df), ax=axes[0])
        axes[0].set_title('Basic Clustering Metrics', fontweight='bold',
                         color=CUSTOM_COLORS['black'])
        axes[0].tick_params(axis='x', rotation=45, colors=CUSTOM_COLORS['black'])
        axes[0].tick_params(axis='y', colors=CUSTOM_COLORS['black'])

        # Add value labels
        for i, v in enumerate(basic_df['Value']):
            axes[0].text(i, v + max(basic_df['Value']) * 0.01, str(v),
                        ha='center', va='bottom', fontweight='bold',
                        color=CUSTOM_COLORS['black'])

        # Quality metrics plot
        if quality_metrics:
            quality_df = pd.DataFrame(list(quality_metrics.items()),
                                    columns=['Metric', 'Score'])
            sns.barplot(data=quality_df, x='Metric', y='Score',
                       palette=[CLUSTER_PALETTE[1]] * len(quality_df), ax=axes[1])
            axes[1].set_title('Quality Metrics', fontweight='bold',
                             color=CUSTOM_COLORS['black'])
            axes[1].tick_params(axis='x', rotation=45, colors=CUSTOM_COLORS['black'])
            axes[1].tick_params(axis='y', colors=CUSTOM_COLORS['black'])

            # Add value labels
            for i, v in enumerate(quality_df['Score']):
                axes[1].text(i, v + (max(quality_df['Score']) - min(quality_df['Score'])) * 0.01,
                           f'{v:.3f}', ha='center', va='bottom', fontweight='bold',
                           color=CUSTOM_COLORS['black'])
        else:
            axes[1].text(0.5, 0.5, 'No quality metrics available',
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=12, color=CUSTOM_COLORS['black'])
            axes[1].set_title('Quality Metrics', fontweight='bold',
                             color=CUSTOM_COLORS['black'])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=CUSTOM_COLORS['white'])
        plt.close()

        logger.info("Metrics summary plot saved to %s", save_path)
        return save_path