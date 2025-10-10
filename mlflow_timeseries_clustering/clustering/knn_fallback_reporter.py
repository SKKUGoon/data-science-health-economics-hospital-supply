"""
kNN Fallback Performance Reporter for generating reports and visualizations.

This module implements the KNNFallbackReporter class that generates
comprehensive reports and visualizations for kNN fallback performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

from ..core.exceptions import PredictionError


logger = logging.getLogger(__name__)


class KNNFallbackReporter:
    """
    Reporter for kNN fallback model performance analysis and visualization.

    This class generates comprehensive reports and visualizations for kNN fallback
    performance, usage patterns, and accuracy analysis using seaborn with the
    specified color scheme.
    """

    # Color scheme: Black, White, Dark Grey, Dark Green
    COLOR_SCHEME = {
        'primary': '#000000',      # Black
        'secondary': '#FFFFFF',    # White
        'tertiary': '#404040',     # Dark Grey
        'accent': '#0d4f01'        # Dark Green
    }

    def __init__(self):
        """Initialize the kNN fallback reporter."""
        # Set seaborn style with custom color palette
        sns.set_style("whitegrid")
        self.custom_palette = [
            self.COLOR_SCHEME['primary'],
            self.COLOR_SCHEME['accent'],
            self.COLOR_SCHEME['tertiary']
        ]

        logger.info("KNNFallbackReporter initialized with custom color scheme")

    def generate_fallback_usage_report(self,
                                     fallback_metrics: Dict[str, Any],
                                     usage_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive fallback usage report.

        Args:
            fallback_metrics: Current fallback model metrics
            usage_history: Optional list of historical usage data

        Returns:
            Dictionary containing report data and visualization paths

        Raises:
            PredictionError: If report generation fails
        """
        logger.info("Generating kNN fallback usage report")

        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'model_status': {
                    'is_fitted': fallback_metrics.get('is_fitted', False),
                    'training_accuracy': fallback_metrics.get('training_accuracy'),
                    'training_data_size': fallback_metrics.get('training_data_size', 0),
                    'n_unique_classes': fallback_metrics.get('n_unique_classes', 0)
                },
                'usage_statistics': {
                    'fallback_usage_count': fallback_metrics.get('fallback_usage_count', 0),
                    'total_fallback_predictions': fallback_metrics.get('total_fallback_predictions', 0),
                    'avg_predictions_per_usage': fallback_metrics.get('avg_predictions_per_usage', 0.0)
                },
                'model_configuration': fallback_metrics.get('knn_params', {}),
                'class_information': {
                    'unique_classes': fallback_metrics.get('unique_classes', [])
                }
            }

            # Add usage trend analysis if history is available
            if usage_history:
                report_data['usage_trends'] = self._analyze_usage_trends(usage_history)

            # Generate summary statistics
            report_data['summary'] = self._generate_usage_summary(report_data)

            logger.info("kNN fallback usage report generated successfully")
            return report_data

        except Exception as e:
            raise PredictionError(
                f"Failed to generate kNN fallback usage report: {str(e)}",
                prediction_type="fallback_usage_report"
            ) from e

    def generate_fallback_accuracy_report(self,
                                        evaluation_metrics: Dict[str, Any],
                                        prediction_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate fallback accuracy analysis report.

        Args:
            evaluation_metrics: Evaluation metrics from test data
            prediction_history: Optional historical prediction data

        Returns:
            Dictionary containing accuracy report data

        Raises:
            PredictionError: If report generation fails
        """
        logger.info("Generating kNN fallback accuracy report")

        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'accuracy_metrics': {
                    'test_accuracy': evaluation_metrics.get('test_accuracy'),
                    'avg_test_confidence': evaluation_metrics.get('avg_test_confidence'),
                    'test_data_size': evaluation_metrics.get('test_data_size', 0),
                    'n_test_classes': evaluation_metrics.get('n_test_classes', 0)
                },
                'classification_performance': evaluation_metrics.get('classification_report', {}),
                'confusion_matrix': evaluation_metrics.get('confusion_matrix', [])
            }

            # Add confidence analysis
            if 'avg_test_confidence' in evaluation_metrics:
                report_data['confidence_analysis'] = self._analyze_prediction_confidence(evaluation_metrics)

            # Add historical accuracy trends if available
            if prediction_history:
                report_data['accuracy_trends'] = self._analyze_accuracy_trends(prediction_history)

            logger.info("kNN fallback accuracy report generated successfully")
            return report_data

        except Exception as e:
            raise PredictionError(
                f"Failed to generate kNN fallback accuracy report: {str(e)}",
                prediction_type="fallback_accuracy_report"
            ) from e

    def create_fallback_usage_visualizations(self,
                                           usage_data: Dict[str, Any],
                                           save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Create visualizations for fallback usage patterns.

        Args:
            usage_data: Usage data from fallback usage report
            save_path: Optional path to save visualizations

        Returns:
            Dictionary mapping visualization names to file paths

        Raises:
            PredictionError: If visualization creation fails
        """
        logger.info("Creating kNN fallback usage visualizations")

        try:
            visualization_paths = {}

            # Create usage statistics bar plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('kNN Fallback Usage Analysis', fontsize=16, color=self.COLOR_SCHEME['primary'])

            # Usage count over time (if trends available)
            if 'usage_trends' in usage_data:
                self._plot_usage_trends(axes[0, 0], usage_data['usage_trends'])
            else:
                self._plot_usage_summary(axes[0, 0], usage_data['usage_statistics'])

            # Predictions per usage distribution
            self._plot_predictions_distribution(axes[0, 1], usage_data['usage_statistics'])

            # Class distribution
            if usage_data['class_information']['unique_classes']:
                self._plot_class_distribution(axes[1, 0], usage_data['class_information'])
            else:
                axes[1, 0].text(0.5, 0.5, 'No class data available',
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Class Distribution')

            # Model configuration summary
            self._plot_model_config(axes[1, 1], usage_data['model_configuration'])

            plt.tight_layout()

            # Save visualization
            if save_path:
                usage_viz_path = f"{save_path}/knn_fallback_usage.png"
                plt.savefig(usage_viz_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                visualization_paths['usage_analysis'] = usage_viz_path

            plt.close()

            logger.info("kNN fallback usage visualizations created successfully")
            return visualization_paths

        except Exception as e:
            raise PredictionError(
                f"Failed to create kNN fallback usage visualizations: {str(e)}",
                prediction_type="fallback_usage_visualization"
            ) from e

    def create_fallback_accuracy_visualizations(self,
                                              accuracy_data: Dict[str, Any],
                                              save_path: Optional[str] = None) -> Dict[str, str]:
        """
        Create visualizations for fallback accuracy analysis.

        Args:
            accuracy_data: Accuracy data from fallback accuracy report
            save_path: Optional path to save visualizations

        Returns:
            Dictionary mapping visualization names to file paths

        Raises:
            PredictionError: If visualization creation fails
        """
        logger.info("Creating kNN fallback accuracy visualizations")

        try:
            visualization_paths = {}

            # Create accuracy analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('kNN Fallback Accuracy Analysis', fontsize=16, color=self.COLOR_SCHEME['primary'])

            # Confusion matrix heatmap
            if accuracy_data['confusion_matrix']:
                self._plot_confusion_matrix(axes[0, 0], accuracy_data['confusion_matrix'])
            else:
                axes[0, 0].text(0.5, 0.5, 'No confusion matrix data',
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Confusion Matrix')

            # Classification metrics bar plot
            if accuracy_data['classification_performance']:
                self._plot_classification_metrics(axes[0, 1], accuracy_data['classification_performance'])
            else:
                axes[0, 1].text(0.5, 0.5, 'No classification metrics',
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Classification Metrics')

            # Confidence analysis
            if 'confidence_analysis' in accuracy_data:
                self._plot_confidence_analysis(axes[1, 0], accuracy_data['confidence_analysis'])
            else:
                axes[1, 0].text(0.5, 0.5, 'No confidence data',
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Prediction Confidence')

            # Accuracy trends over time
            if 'accuracy_trends' in accuracy_data:
                self._plot_accuracy_trends(axes[1, 1], accuracy_data['accuracy_trends'])
            else:
                self._plot_accuracy_summary(axes[1, 1], accuracy_data['accuracy_metrics'])

            plt.tight_layout()

            # Save visualization
            if save_path:
                accuracy_viz_path = f"{save_path}/knn_fallback_accuracy.png"
                plt.savefig(accuracy_viz_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                visualization_paths['accuracy_analysis'] = accuracy_viz_path

            plt.close()

            logger.info("kNN fallback accuracy visualizations created successfully")
            return visualization_paths

        except Exception as e:
            raise PredictionError(
                f"Failed to create kNN fallback accuracy visualizations: {str(e)}",
                prediction_type="fallback_accuracy_visualization"
            ) from e

    def _analyze_usage_trends(self, usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze usage trends from historical data."""
        if not usage_history:
            return {}

        df = pd.DataFrame(usage_history)

        trends = {
            'total_usage_events': len(df),
            'avg_predictions_per_event': df.get('n_predictions', pd.Series()).mean(),
            'usage_frequency_trend': 'increasing' if len(df) > 1 and df.index[-1] > df.index[0] else 'stable'
        }

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            trends['time_span_hours'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600

        return trends

    def _analyze_accuracy_trends(self, prediction_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze accuracy trends from historical prediction data."""
        if not prediction_history:
            return {}

        df = pd.DataFrame(prediction_history)

        trends = {
            'total_evaluations': len(df),
            'avg_accuracy': df.get('accuracy', pd.Series()).mean(),
            'accuracy_std': df.get('accuracy', pd.Series()).std(),
            'accuracy_trend': 'improving' if len(df) > 1 and df.get('accuracy', pd.Series()).iloc[-1] > df.get('accuracy', pd.Series()).iloc[0] else 'stable'
        }

        return trends

    def _analyze_prediction_confidence(self, evaluation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction confidence metrics."""
        confidence_analysis = {
            'avg_confidence': evaluation_metrics.get('avg_test_confidence', 0.0),
            'confidence_category': 'high' if evaluation_metrics.get('avg_test_confidence', 0.0) > 0.8 else
                                 'medium' if evaluation_metrics.get('avg_test_confidence', 0.0) > 0.6 else 'low'
        }

        return confidence_analysis

    def _generate_usage_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for usage report."""
        usage_stats = report_data['usage_statistics']
        model_status = report_data['model_status']

        summary = {
            'model_health': 'healthy' if model_status['is_fitted'] and model_status['training_accuracy'] and model_status['training_accuracy'] > 0.7 else 'needs_attention',
            'usage_level': 'high' if usage_stats['fallback_usage_count'] > 100 else
                          'medium' if usage_stats['fallback_usage_count'] > 10 else 'low',
            'efficiency': usage_stats['avg_predictions_per_usage']
        }

        return summary

    def _plot_usage_trends(self, ax, trends_data: Dict[str, Any]):
        """Plot usage trends over time."""
        ax.bar(['Usage Events', 'Avg Predictions'],
               [trends_data.get('total_usage_events', 0), trends_data.get('avg_predictions_per_event', 0)],
               color=[self.COLOR_SCHEME['accent'], self.COLOR_SCHEME['tertiary']])
        ax.set_title('Usage Trends', color=self.COLOR_SCHEME['primary'])
        ax.set_ylabel('Count')

    def _plot_usage_summary(self, ax, usage_stats: Dict[str, Any]):
        """Plot usage summary statistics."""
        metrics = ['Usage Count', 'Total Predictions', 'Avg per Usage']
        values = [
            usage_stats.get('fallback_usage_count', 0),
            usage_stats.get('total_fallback_predictions', 0),
            usage_stats.get('avg_predictions_per_usage', 0.0)
        ]

        bars = ax.bar(metrics, values, color=self.custom_palette)
        ax.set_title('Usage Summary', color=self.COLOR_SCHEME['primary'])
        ax.set_ylabel('Count')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}', ha='center', va='bottom')

    def _plot_predictions_distribution(self, ax, usage_stats: Dict[str, Any]):
        """Plot predictions per usage distribution."""
        avg_predictions = usage_stats.get('avg_predictions_per_usage', 0.0)
        usage_count = usage_stats.get('fallback_usage_count', 0)

        # Create a simple distribution visualization
        ax.bar(['Avg Predictions per Usage'], [avg_predictions],
               color=self.COLOR_SCHEME['accent'])
        ax.set_title('Predictions Distribution', color=self.COLOR_SCHEME['primary'])
        ax.set_ylabel('Average Count')

        # Add text annotation
        ax.text(0, avg_predictions + avg_predictions*0.1,
               f'Based on {usage_count} usage events',
               ha='center', va='bottom')

    def _plot_class_distribution(self, ax, class_info: Dict[str, Any]):
        """Plot class distribution."""
        unique_classes = class_info.get('unique_classes', [])

        if unique_classes:
            class_counts = [1] * len(unique_classes)  # Equal distribution for visualization
            ax.pie(class_counts, labels=[f'Class {cls}' for cls in unique_classes],
                   colors=self.custom_palette[:len(unique_classes)], autopct='%1.1f%%')
            ax.set_title('Class Distribution', color=self.COLOR_SCHEME['primary'])
        else:
            ax.text(0.5, 0.5, 'No class data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Class Distribution')

    def _plot_model_config(self, ax, config: Dict[str, Any]):
        """Plot model configuration summary."""
        if not config:
            ax.text(0.5, 0.5, 'No configuration data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Configuration')
            return

        # Display key configuration parameters
        config_text = []
        for key, value in config.items():
            if key in ['n_neighbors', 'weights', 'algorithm']:
                config_text.append(f'{key}: {value}')

        ax.text(0.1, 0.9, '\n'.join(config_text),
               transform=ax.transAxes, verticalalignment='top',
               fontsize=10, color=self.COLOR_SCHEME['primary'])
        ax.set_title('Model Configuration', color=self.COLOR_SCHEME['primary'])
        ax.axis('off')

    def _plot_confusion_matrix(self, ax, confusion_matrix: List[List[int]]):
        """Plot confusion matrix heatmap."""
        cm_array = np.array(confusion_matrix)
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Greens', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_title('Confusion Matrix', color=self.COLOR_SCHEME['primary'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    def _plot_classification_metrics(self, ax, classification_report: Dict[str, Any]):
        """Plot classification metrics."""
        # Extract macro avg metrics
        macro_avg = classification_report.get('macro avg', {})
        metrics = ['precision', 'recall', 'f1-score']
        values = [macro_avg.get(metric, 0.0) for metric in metrics]

        bars = ax.bar(metrics, values, color=self.custom_palette)
        ax.set_title('Classification Metrics', color=self.COLOR_SCHEME['primary'])
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

    def _plot_confidence_analysis(self, ax, confidence_data: Dict[str, Any]):
        """Plot confidence analysis."""
        avg_confidence = confidence_data.get('avg_confidence', 0.0)
        category = confidence_data.get('confidence_category', 'unknown')

        # Create confidence gauge-like visualization
        ax.bar(['Avg Confidence'], [avg_confidence],
               color=self.COLOR_SCHEME['accent'])
        ax.set_title('Prediction Confidence', color=self.COLOR_SCHEME['primary'])
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 1)

        # Add category annotation
        ax.text(0, avg_confidence + 0.05, f'Category: {category}',
               ha='center', va='bottom')

    def _plot_accuracy_trends(self, ax, trends_data: Dict[str, Any]):
        """Plot accuracy trends over time."""
        metrics = ['Avg Accuracy', 'Accuracy Std']
        values = [
            trends_data.get('avg_accuracy', 0.0),
            trends_data.get('accuracy_std', 0.0)
        ]

        bars = ax.bar(metrics, values, color=[self.COLOR_SCHEME['accent'], self.COLOR_SCHEME['tertiary']])
        ax.set_title('Accuracy Trends', color=self.COLOR_SCHEME['primary'])
        ax.set_ylabel('Score')

        # Add trend annotation
        trend = trends_data.get('accuracy_trend', 'stable')
        ax.text(0.5, 0.9, f'Trend: {trend}', transform=ax.transAxes,
               ha='center', va='top')

    def _plot_accuracy_summary(self, ax, accuracy_metrics: Dict[str, Any]):
        """Plot accuracy summary."""
        test_accuracy = accuracy_metrics.get('test_accuracy', 0.0)
        avg_confidence = accuracy_metrics.get('avg_test_confidence', 0.0)

        metrics = ['Test Accuracy', 'Avg Confidence']
        values = [test_accuracy, avg_confidence]

        bars = ax.bar(metrics, values, color=self.custom_palette)
        ax.set_title('Accuracy Summary', color=self.COLOR_SCHEME['primary'])
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')