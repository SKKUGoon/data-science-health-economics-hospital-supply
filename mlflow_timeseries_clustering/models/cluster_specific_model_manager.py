"""
Cluster-specific model manager for time series and resource usage prediction.

This module implements the ClusterSpecificModelManager class that handles
LightGBM model training and prediction for each cluster identified by HDBSCAN.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import warnings
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..core.config import PipelineConfig
from ..core.exceptions import ModelTrainingError, PredictionError
from ..mlflow_integration.logging_utils import PipelineLogger
from ..mlflow_integration.artifact_manager import ArtifactManager


class ClusterSpecificModelManager:
    """
    Manages cluster-specific LightGBM models for time series and resource prediction.

    This class handles training separate LightGBM models for each cluster,
    tracks performance metrics, and provides prediction capabilities.
    """

    def __init__(self, config: PipelineConfig, artifact_manager: ArtifactManager):
        """
        Initialize the cluster-specific model manager.

        Args:
            config: Pipeline configuration containing LightGBM parameters
            artifact_manager: Artifact manager for saving/loading models
        """
        self.config = config
        self.artifact_manager = artifact_manager
        self.logger = PipelineLogger("ClusterSpecificModelManager")

        # Model storage
        self.timeseries_models: Dict[int, lgb.LGBMRegressor] = {}
        self.resource_models: Dict[int, lgb.LGBMRegressor] = {}

        # Performance tracking
        self.timeseries_metrics: Dict[int, Dict[str, float]] = {}
        self.resource_metrics: Dict[int, Dict[str, float]] = {}
        self.feature_importance: Dict[int, Dict[str, np.ndarray]] = {}

        # Training data tracking
        self.cluster_data_sizes: Dict[int, int] = {}
        self.training_history: List[Dict[str, Any]] = []

        # Minimum data requirements
        self.min_samples_per_cluster = 10
        self.min_features = 1

    def fit_timeseries_models(self, cluster_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[str, float]]:
        """
        Fit LightGBM time series models for each cluster.

        Args:
            cluster_data: Dictionary mapping cluster_id to {'X': features, 'y': targets}

        Returns:
            Dictionary mapping cluster_id to performance metrics

        Raises:
            ModelTrainingError: If model training fails
        """
        self.logger.info("Starting time series model training for all clusters")

        trained_models = {}
        all_metrics = {}

        for cluster_id, data in cluster_data.items():
            try:
                metrics = self._fit_single_timeseries_model(cluster_id, data['X'], data['y'])
                if metrics is not None:
                    trained_models[cluster_id] = self.timeseries_models[cluster_id]
                    all_metrics[cluster_id] = metrics

            except Exception as e:
                self.logger.error(f"Failed to train time series model for cluster {cluster_id}: {str(e)}")
                # Continue with other clusters rather than failing completely
                continue

        self.logger.info(f"Successfully trained time series models for {len(trained_models)} clusters")

        # Log overall training summary
        self._log_training_summary("timeseries", trained_models, all_metrics)

        return all_metrics

    def predict_timeseries(self, cluster_assignments: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make time series predictions for new data batches using cluster-specific models.

        Args:
            cluster_assignments: Cluster assignments for each data point
            X: Feature matrix for prediction

        Returns:
            Tuple of (predictions array, prediction metadata)

        Raises:
            PredictionError: If prediction fails
        """
        try:
            self.logger.info(f"Making time series predictions for {len(X)} samples")

            # Validate input data
            self._validate_prediction_data(cluster_assignments, X)

            # Initialize predictions array
            predictions = np.full(len(X), np.nan)
            prediction_metadata = {
                'cluster_predictions': {},
                'failed_predictions': [],
                'noise_point_predictions': 0,
                'successful_predictions': 0
            }

            # Get unique clusters in the batch
            unique_clusters = np.unique(cluster_assignments)

            for cluster_id in unique_clusters:
                # Skip noise points (cluster_id = -1)
                if cluster_id == -1:
                    noise_mask = cluster_assignments == -1
                    prediction_metadata['noise_point_predictions'] = np.sum(noise_mask)
                    self.logger.warning(f"Found {np.sum(noise_mask)} noise points, predictions will be NaN")
                    continue

                # Get data points for this cluster
                cluster_mask = cluster_assignments == cluster_id
                cluster_X = X[cluster_mask]

                if len(cluster_X) == 0:
                    continue

                try:
                    # Make predictions for this cluster
                    cluster_predictions = self._predict_single_cluster(cluster_id, cluster_X)
                    predictions[cluster_mask] = cluster_predictions

                    # Store metadata
                    prediction_metadata['cluster_predictions'][cluster_id] = {
                        'sample_count': len(cluster_X),
                        'mean_prediction': np.mean(cluster_predictions),
                        'std_prediction': np.std(cluster_predictions),
                        'min_prediction': np.min(cluster_predictions),
                        'max_prediction': np.max(cluster_predictions)
                    }

                    prediction_metadata['successful_predictions'] += len(cluster_X)

                except Exception as e:
                    self.logger.error(f"Failed to predict for cluster {cluster_id}: {str(e)}")
                    prediction_metadata['failed_predictions'].append({
                        'cluster_id': cluster_id,
                        'sample_count': len(cluster_X),
                        'error': str(e)
                    })

            # Log prediction summary
            self._log_prediction_summary(prediction_metadata)

            self.logger.info(f"Completed time series predictions: {prediction_metadata['successful_predictions']} successful")

            return predictions, prediction_metadata

        except Exception as e:
            raise PredictionError(
                f"Failed to make time series predictions: {str(e)}",
                prediction_type="timeseries",
                batch_size=len(X) if X is not None else 0
            )

    def _predict_single_cluster(self, cluster_id: int, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for a single cluster.

        Args:
            cluster_id: Cluster identifier
            X: Feature matrix for the cluster

        Returns:
            Predictions array

        Raises:
            PredictionError: If prediction fails
        """
        if cluster_id not in self.timeseries_models:
            raise PredictionError(
                f"No trained model available for cluster {cluster_id}",
                prediction_type="timeseries",
                cluster_id=cluster_id
            )

        model = self.timeseries_models[cluster_id]

        try:
            predictions = model.predict(X)
            return predictions

        except Exception as e:
            raise PredictionError(
                f"Model prediction failed for cluster {cluster_id}: {str(e)}",
                prediction_type="timeseries",
                cluster_id=cluster_id,
                batch_size=len(X)
            )

    def _validate_prediction_data(self, cluster_assignments: np.ndarray, X: np.ndarray):
        """
        Validate data for prediction.

        Args:
            cluster_assignments: Cluster assignments
            X: Feature matrix

        Raises:
            PredictionError: If validation fails
        """
        if cluster_assignments is None or X is None:
            raise PredictionError("Prediction data cannot be None", prediction_type="timeseries")

        if not isinstance(cluster_assignments, np.ndarray) or not isinstance(X, np.ndarray):
            raise PredictionError("Prediction data must be numpy arrays", prediction_type="timeseries")

        if len(cluster_assignments) != len(X):
            raise PredictionError(
                f"Cluster assignments and features must have same length: "
                f"assignments={len(cluster_assignments)}, features={len(X)}",
                prediction_type="timeseries"
            )

        if X.ndim != 2:
            raise PredictionError(
                f"Feature matrix must be 2D: got shape {X.shape}",
                prediction_type="timeseries"
            )

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise PredictionError(
                "Feature matrix contains NaN or infinite values",
                prediction_type="timeseries"
            )

    def track_prediction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                cluster_assignments: np.ndarray, step: Optional[int] = None):
        """
        Track prediction accuracy over time for monitoring.

        Args:
            y_true: True target values
            y_pred: Predicted values
            cluster_assignments: Cluster assignments for each prediction
            step: Optional step number for time series tracking
        """
        try:
            # Calculate overall accuracy metrics
            overall_metrics = self._calculate_timeseries_metrics(y_true, y_pred, cluster_id=-999)  # Use -999 for overall

            # Log overall metrics with step
            self.logger.log_metrics(overall_metrics, step=step, prefix="prediction_accuracy_overall")

            # Calculate per-cluster accuracy
            unique_clusters = np.unique(cluster_assignments)
            cluster_accuracies = {}

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue

                cluster_mask = cluster_assignments == cluster_id
                if np.sum(cluster_mask) == 0:
                    continue

                cluster_y_true = y_true[cluster_mask]
                cluster_y_pred = y_pred[cluster_mask]

                if len(cluster_y_true) > 0:
                    cluster_metrics = self._calculate_timeseries_metrics(cluster_y_true, cluster_y_pred, cluster_id)
                    cluster_accuracies[cluster_id] = cluster_metrics

                    # Log cluster-specific metrics
                    self.logger.log_metrics(
                        cluster_metrics,
                        step=step,
                        prefix=f"prediction_accuracy_cluster_{cluster_id}"
                    )

            # Store accuracy tracking
            accuracy_record = {
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'overall_metrics': overall_metrics,
                'cluster_accuracies': cluster_accuracies,
                'total_predictions': len(y_true),
                'clusters_evaluated': len(cluster_accuracies)
            }

            # Add to training history for tracking
            self.training_history.append({
                'type': 'prediction_accuracy',
                **accuracy_record
            })

            self.logger.info(f"Tracked prediction accuracy for step {step}: RMSE={overall_metrics['rmse']:.4f}")

        except Exception as e:
            self.logger.error(f"Failed to track prediction accuracy: {str(e)}")

    def create_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                               cluster_assignments: np.ndarray) -> Dict[str, Any]:
        """
        Create residual analysis for model validation.

        Args:
            y_true: True target values
            y_pred: Predicted values
            cluster_assignments: Cluster assignments

        Returns:
            Dictionary containing residual analysis results
        """
        try:
            residuals = y_true - y_pred

            analysis = {
                'overall_residuals': {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'min': np.min(residuals),
                    'max': np.max(residuals),
                    'median': np.median(residuals),
                    'q25': np.percentile(residuals, 25),
                    'q75': np.percentile(residuals, 75)
                },
                'cluster_residuals': {},
                'normality_tests': {},
                'outlier_analysis': {}
            }

            # Per-cluster residual analysis
            unique_clusters = np.unique(cluster_assignments)
            for cluster_id in unique_clusters:
                if cluster_id == -1:
                    continue

                cluster_mask = cluster_assignments == cluster_id
                cluster_residuals = residuals[cluster_mask]

                if len(cluster_residuals) > 0:
                    analysis['cluster_residuals'][cluster_id] = {
                        'mean': np.mean(cluster_residuals),
                        'std': np.std(cluster_residuals),
                        'min': np.min(cluster_residuals),
                        'max': np.max(cluster_residuals),
                        'count': len(cluster_residuals)
                    }

            # Outlier detection (using IQR method)
            q1, q3 = np.percentile(residuals, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = (residuals < lower_bound) | (residuals > upper_bound)
            analysis['outlier_analysis'] = {
                'outlier_count': np.sum(outliers),
                'outlier_percentage': np.sum(outliers) / len(residuals) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to create residual analysis: {str(e)}")
            return {'error': str(e)}

    def _log_prediction_summary(self, prediction_metadata: Dict[str, Any]):
        """
        Log prediction summary metrics.

        Args:
            prediction_metadata: Metadata from prediction operation
        """
        try:
            summary_metrics = {
                'total_clusters_predicted': len(prediction_metadata['cluster_predictions']),
                'successful_predictions': prediction_metadata['successful_predictions'],
                'failed_predictions': len(prediction_metadata['failed_predictions']),
                'noise_point_predictions': prediction_metadata['noise_point_predictions']
            }

            # Add cluster-specific summary stats
            if prediction_metadata['cluster_predictions']:
                cluster_means = [stats['mean_prediction'] for stats in prediction_metadata['cluster_predictions'].values()]
                summary_metrics.update({
                    'avg_cluster_prediction': np.mean(cluster_means),
                    'std_cluster_prediction': np.std(cluster_means),
                    'min_cluster_prediction': np.min(cluster_means),
                    'max_cluster_prediction': np.max(cluster_means)
                })

            self.logger.log_metrics(summary_metrics, prefix="prediction_summary")

        except Exception as e:
            self.logger.error(f"Failed to log prediction summary: {str(e)}")

    def generate_performance_report(self, y_true: Optional[np.ndarray] = None,
                                  y_pred: Optional[np.ndarray] = None,
                                  cluster_assignments: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with visualizations.

        Args:
            y_true: Optional true values for validation data
            y_pred: Optional predictions for validation data
            cluster_assignments: Optional cluster assignments for validation data

        Returns:
            Dictionary containing report data and visualization paths
        """
        try:
            self.logger.info("Generating time series model performance report")

            report = {
                'timestamp': datetime.now().isoformat(),
                'model_summary': self._generate_model_summary(),
                'performance_metrics': self.get_model_metrics("timeseries"),
                'visualizations': {},
                'feature_importance_summary': self._generate_feature_importance_summary()
            }

            # Add validation metrics if data provided
            if y_true is not None and y_pred is not None and cluster_assignments is not None:
                report['validation_analysis'] = self.create_residual_analysis(y_true, y_pred, cluster_assignments)

            # Generate visualizations
            report['visualizations'] = self._generate_performance_visualizations(
                y_true, y_pred, cluster_assignments
            )

            # Save report as artifact
            report_html = self._create_html_report(report)
            report_path = self.artifact_manager.save_report_artifact(
                report_content=report_html,
                filename="timeseries_performance_report.html",
                subfolder="timeseries_models"
            )

            report['report_path'] = report_path

            self.logger.info("Successfully generated time series performance report")
            return report

        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _generate_model_summary(self) -> Dict[str, Any]:
        """Generate summary of trained models."""
        return {
            'total_models_trained': len(self.timeseries_models),
            'clusters_with_models': list(self.timeseries_models.keys()),
            'total_samples_trained': sum(self.cluster_data_sizes.values()),
            'avg_samples_per_cluster': np.mean(list(self.cluster_data_sizes.values())) if self.cluster_data_sizes else 0,
            'min_samples_per_cluster': min(self.cluster_data_sizes.values()) if self.cluster_data_sizes else 0,
            'max_samples_per_cluster': max(self.cluster_data_sizes.values()) if self.cluster_data_sizes else 0,
            'model_parameters': self.config.lgb_timeseries_params
        }

    def _generate_feature_importance_summary(self) -> Dict[str, Any]:
        """Generate summary of feature importance across clusters."""
        if not self.feature_importance:
            return {'available': False}

        summary = {'available': True, 'cluster_summaries': {}}

        for cluster_id, importance_data in self.feature_importance.items():
            if 'timeseries' in importance_data:
                importance = importance_data['timeseries']
                top_5_indices = np.argsort(importance)[-5:][::-1]

                summary['cluster_summaries'][cluster_id] = {
                    'total_features': len(importance),
                    'top_5_features': {
                        f'feature_{idx}': float(importance[idx])
                        for idx in top_5_indices
                    },
                    'importance_stats': {
                        'mean': float(np.mean(importance)),
                        'std': float(np.std(importance)),
                        'max': float(np.max(importance)),
                        'min': float(np.min(importance))
                    }
                }

        return summary

    def _generate_performance_visualizations(self, y_true: Optional[np.ndarray] = None,
                                           y_pred: Optional[np.ndarray] = None,
                                           cluster_assignments: Optional[np.ndarray] = None) -> Dict[str, str]:
        """
        Generate performance visualizations using seaborn.

        Returns:
            Dictionary mapping visualization names to artifact paths
        """
        visualizations = {}

        try:
            # Set up the custom color palette
            colors = ['black', 'white', 'darkgray', '#0d4f01']  # Dark green
            sns.set_palette(colors)
            plt.style.use('default')

            # 1. Model Performance Summary Plot
            if self.timeseries_metrics:
                fig = self._create_model_performance_plot()
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="model_performance_summary.png",
                    subfolder="timeseries_models"
                )
                visualizations['model_performance_summary'] = viz_path
                plt.close(fig)

            # 2. Feature Importance Plot
            if self.feature_importance:
                fig = self._create_feature_importance_plot()
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="feature_importance.png",
                    subfolder="timeseries_models"
                )
                visualizations['feature_importance'] = viz_path
                plt.close(fig)

            # 3. Prediction vs Actual Plot (if validation data provided)
            if y_true is not None and y_pred is not None:
                fig = self._create_prediction_scatter_plot(y_true, y_pred, cluster_assignments)
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="prediction_vs_actual.png",
                    subfolder="timeseries_models"
                )
                visualizations['prediction_vs_actual'] = viz_path
                plt.close(fig)

            # 4. Residual Analysis Plot (if validation data provided)
            if y_true is not None and y_pred is not None:
                fig = self._create_residual_analysis_plot(y_true, y_pred, cluster_assignments)
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="residual_analysis.png",
                    subfolder="timeseries_models"
                )
                visualizations['residual_analysis'] = viz_path
                plt.close(fig)

            # 5. Cluster Performance Comparison
            if len(self.timeseries_metrics) > 1:
                fig = self._create_cluster_comparison_plot()
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="cluster_performance_comparison.png",
                    subfolder="timeseries_models"
                )
                visualizations['cluster_performance_comparison'] = viz_path
                plt.close(fig)

        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {str(e)}")
            visualizations['error'] = str(e)

        return visualizations

    def _create_model_performance_plot(self):
        """Create model performance summary plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Time Series Model Performance Summary', fontsize=16, fontweight='bold')

        # Extract metrics for all clusters
        cluster_ids = list(self.timeseries_metrics.keys())
        rmse_values = [self.timeseries_metrics[cid]['rmse'] for cid in cluster_ids]
        mae_values = [self.timeseries_metrics[cid]['mae'] for cid in cluster_ids]
        r2_values = [self.timeseries_metrics[cid]['r2_score'] for cid in cluster_ids]
        mape_values = [self.timeseries_metrics[cid].get('mape', 0) for cid in cluster_ids]

        # RMSE by cluster
        sns.barplot(x=cluster_ids, y=rmse_values, ax=axes[0, 0], color='#0d4f01')
        axes[0, 0].set_title('RMSE by Cluster')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('RMSE')

        # MAE by cluster
        sns.barplot(x=cluster_ids, y=mae_values, ax=axes[0, 1], color='darkgray')
        axes[0, 1].set_title('MAE by Cluster')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('MAE')

        # R² by cluster
        sns.barplot(x=cluster_ids, y=r2_values, ax=axes[1, 0], color='#0d4f01')
        axes[1, 0].set_title('R² Score by Cluster')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('R² Score')

        # MAPE by cluster
        valid_mape = [m for m in mape_values if not np.isinf(m)]
        if valid_mape:
            sns.barplot(x=[cid for cid, m in zip(cluster_ids, mape_values) if not np.isinf(m)],
                       y=valid_mape, ax=axes[1, 1], color='darkgray')
        axes[1, 1].set_title('MAPE by Cluster')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('MAPE (%)')

        plt.tight_layout()
        return fig

    def _create_feature_importance_plot(self):
        """Create feature importance visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Aggregate feature importance across clusters
        all_importance = []
        cluster_labels = []
        feature_indices = []

        for cluster_id, importance_data in self.feature_importance.items():
            if 'timeseries' in importance_data:
                importance = importance_data['timeseries']
                top_5_indices = np.argsort(importance)[-5:][::-1]

                for idx in top_5_indices:
                    all_importance.append(importance[idx])
                    cluster_labels.append(f'Cluster {cluster_id}')
                    feature_indices.append(f'Feature {idx}')

        if all_importance:
            # Create DataFrame for seaborn
            df = pd.DataFrame({
                'Importance': all_importance,
                'Cluster': cluster_labels,
                'Feature': feature_indices
            })

            # Create grouped bar plot
            sns.barplot(data=df, x='Feature', y='Importance', hue='Cluster', ax=ax)
            ax.set_title('Top Feature Importance by Cluster', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Importance')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def _create_prediction_scatter_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      cluster_assignments: Optional[np.ndarray] = None):
        """Create prediction vs actual scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if cluster_assignments is not None:
            # Color by cluster
            unique_clusters = np.unique(cluster_assignments)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

            for i, cluster_id in enumerate(unique_clusters):
                if cluster_id == -1:  # Skip noise points
                    continue
                mask = cluster_assignments == cluster_id
                ax.scatter(y_true[mask], y_pred[mask],
                          c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6)
        else:
            ax.scatter(y_true, y_pred, color='#0d4f01', alpha=0.6)

        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)

        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')

        if cluster_assignments is not None:
            ax.legend()

        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    def _create_residual_analysis_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     cluster_assignments: Optional[np.ndarray] = None):
        """Create residual analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')

        residuals = y_true - y_pred

        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, color='#0d4f01', alpha=0.6)
        axes[0, 0].axhline(y=0, color='black', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')

        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, color='darkgray', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')

        # Residuals by cluster (if available)
        if cluster_assignments is not None:
            unique_clusters = np.unique(cluster_assignments)
            cluster_residuals = []
            cluster_labels = []

            for cluster_id in unique_clusters:
                if cluster_id == -1:
                    continue
                mask = cluster_assignments == cluster_id
                cluster_residuals.extend(residuals[mask])
                cluster_labels.extend([f'Cluster {cluster_id}'] * np.sum(mask))

            if cluster_residuals:
                df = pd.DataFrame({'Residuals': cluster_residuals, 'Cluster': cluster_labels})
                sns.boxplot(data=df, x='Cluster', y='Residuals', ax=axes[1, 1])
                axes[1, 1].set_title('Residuals by Cluster')
                axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No cluster data available',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Residuals by Cluster')

        plt.tight_layout()
        return fig

    def _create_cluster_comparison_plot(self):
        """Create cluster performance comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for comparison
        metrics_data = []
        for cluster_id, metrics in self.timeseries_metrics.items():
            metrics_data.append({
                'Cluster': f'Cluster {cluster_id}',
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2_score'],
                'Samples': self.cluster_data_sizes.get(cluster_id, 0)
            })

        df = pd.DataFrame(metrics_data)

        # Create bubble plot: RMSE vs MAE, size = samples, color = R²
        scatter = ax.scatter(df['RMSE'], df['MAE'],
                           s=df['Samples']/10,  # Scale bubble size
                           c=df['R²'], cmap='RdYlGn', alpha=0.7)

        # Add cluster labels
        for i, row in df.iterrows():
            ax.annotate(row['Cluster'], (row['RMSE'], row['MAE']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('RMSE')
        ax.set_ylabel('MAE')
        ax.set_title('Cluster Performance Comparison\n(Bubble size = Sample count, Color = R² score)',
                    fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('R² Score')

        plt.tight_layout()
        return fig

    def _create_html_report(self, report_data: Dict[str, Any]) -> str:
        """Create HTML report from report data."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Time Series Model Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #0d4f01; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .error { color: red; }
                .success { color: #0d4f01; }
            </style>
        </head>
        <body>
            <h1>Time Series Model Performance Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>

            <h2>Model Summary</h2>
            <div class="metric">
                <p><strong>Total Models Trained:</strong> {total_models}</p>
                <p><strong>Total Samples:</strong> {total_samples}</p>
                <p><strong>Average Samples per Cluster:</strong> {avg_samples:.2f}</p>
            </div>

            <h2>Performance Metrics by Cluster</h2>
            <table>
                <tr>
                    <th>Cluster ID</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R² Score</th>
                    <th>MAPE (%)</th>
                    <th>Sample Count</th>
                </tr>
                {metrics_rows}
            </table>

            <h2>Feature Importance Summary</h2>
            {feature_importance_section}

            <h2>Visualizations</h2>
            {visualizations_section}

        </body>
        </html>
        """

        # Format metrics rows
        metrics_rows = ""
        if 'performance_metrics' in report_data:
            for cluster_id, metrics in report_data['performance_metrics'].items():
                sample_count = report_data['model_summary']['total_samples'] // len(report_data['performance_metrics'])
                mape = metrics.get('mape', 'N/A')
                mape_str = f"{mape:.2f}" if isinstance(mape, (int, float)) and not np.isinf(mape) else "N/A"

                metrics_rows += f"""
                <tr>
                    <td>{cluster_id}</td>
                    <td>{metrics['rmse']:.4f}</td>
                    <td>{metrics['mae']:.4f}</td>
                    <td>{metrics['r2_score']:.4f}</td>
                    <td>{mape_str}</td>
                    <td>{sample_count}</td>
                </tr>
                """

        # Format feature importance section
        feature_importance_section = ""
        if report_data['feature_importance_summary']['available']:
            feature_importance_section = "<p>Feature importance data is available for trained models.</p>"
        else:
            feature_importance_section = "<p>No feature importance data available.</p>"

        # Format visualizations section
        visualizations_section = ""
        if 'visualizations' in report_data and report_data['visualizations']:
            visualizations_section = "<ul>"
            for viz_name, viz_path in report_data['visualizations'].items():
                if viz_name != 'error':
                    visualizations_section += f"<li><strong>{viz_name.replace('_', ' ').title()}:</strong> {viz_path}</li>"
            visualizations_section += "</ul>"
        else:
            visualizations_section = "<p>No visualizations generated.</p>"

        return html_template.format(
            timestamp=report_data['timestamp'],
            total_models=report_data['model_summary']['total_models_trained'],
            total_samples=report_data['model_summary']['total_samples_trained'],
            avg_samples=report_data['model_summary']['avg_samples_per_cluster'],
            metrics_rows=metrics_rows,
            feature_importance_section=feature_importance_section,
            visualizations_section=visualizations_section
        )

    def _fit_single_timeseries_model(self, cluster_id: int, X: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Fit a single LightGBM time series model for a specific cluster.

        Args:
            cluster_id: Cluster identifier
            X: Feature matrix for the cluster
            y: Target values for the cluster

        Returns:
            Performance metrics dictionary, or None if training skipped

        Raises:
            ModelTrainingError: If model training fails
        """
        try:
            # Validate input data
            self._validate_training_data(X, y, cluster_id, "timeseries")

            # Check if we have sufficient data
            if len(X) < self.min_samples_per_cluster:
                self.handle_insufficient_data_warning(cluster_id, len(X), "timeseries")
                return None

            self.logger.info(f"Training time series model for cluster {cluster_id} with {len(X)} samples")

            # Store cluster data size
            self.cluster_data_sizes[cluster_id] = len(X)

            # Create and configure LightGBM model
            model = lgb.LGBMRegressor(**self.config.lgb_timeseries_params)

            # Fit the model
            model.fit(
                X, y,
                eval_set=[(X, y)],
                eval_names=['train'],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            # Store the trained model
            self.timeseries_models[cluster_id] = model

            # Calculate performance metrics
            y_pred = model.predict(X)
            metrics = self._calculate_timeseries_metrics(y, y_pred, cluster_id)

            # Store metrics
            self.timeseries_metrics[cluster_id] = metrics

            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[cluster_id] = {
                    'timeseries': model.feature_importances_
                }

            # Log model information
            self.logger.log_model_info(
                model_type="LightGBM_TimeSeries",
                model_params=self.config.lgb_timeseries_params,
                training_metrics=metrics,
                cluster_id=cluster_id
            )

            # Save model artifact
            self.artifact_manager.save_model_artifact(
                model=model,
                artifact_name="timeseries_model",
                component="lightgbm_timeseries",
                cluster_id=cluster_id
            )

            self.logger.info(f"Successfully trained time series model for cluster {cluster_id}")
            return metrics

        except Exception as e:
            raise ModelTrainingError(
                f"Failed to train time series model for cluster {cluster_id}: {str(e)}",
                model_type="LightGBM_TimeSeries",
                cluster_id=cluster_id,
                training_data_size=len(X) if X is not None else 0
            )

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray, cluster_id: int, model_type: str):
        """
        Validate training data for model fitting.

        Args:
            X: Feature matrix
            y: Target values
            cluster_id: Cluster identifier
            model_type: Type of model being trained

        Raises:
            ModelTrainingError: If data validation fails
        """
        if X is None or y is None:
            raise ModelTrainingError(
                f"Training data cannot be None for cluster {cluster_id}",
                model_type=model_type,
                cluster_id=cluster_id
            )

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ModelTrainingError(
                f"Training data must be numpy arrays for cluster {cluster_id}",
                model_type=model_type,
                cluster_id=cluster_id
            )

        if len(X) == 0 or len(y) == 0:
            raise ModelTrainingError(
                f"Training data cannot be empty for cluster {cluster_id}",
                model_type=model_type,
                cluster_id=cluster_id
            )

        if len(X) != len(y):
            raise ModelTrainingError(
                f"Feature matrix and target vector must have same length for cluster {cluster_id}: "
                f"X.shape[0]={len(X)}, y.shape[0]={len(y)}",
                model_type=model_type,
                cluster_id=cluster_id
            )

        if X.ndim != 2:
            raise ModelTrainingError(
                f"Feature matrix must be 2D for cluster {cluster_id}: got shape {X.shape}",
                model_type=model_type,
                cluster_id=cluster_id
            )

        if y.ndim != 1:
            raise ModelTrainingError(
                f"Target vector must be 1D for cluster {cluster_id}: got shape {y.shape}",
                model_type=model_type,
                cluster_id=cluster_id
            )

        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ModelTrainingError(
                f"Feature matrix contains NaN or infinite values for cluster {cluster_id}",
                model_type=model_type,
                cluster_id=cluster_id
            )

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ModelTrainingError(
                f"Target vector contains NaN or infinite values for cluster {cluster_id}",
                model_type=model_type,
                cluster_id=cluster_id
            )

    def _calculate_timeseries_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, cluster_id: int) -> Dict[str, float]:
        """
        Calculate comprehensive time series performance metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            cluster_id: Cluster identifier

        Returns:
            Dictionary of performance metrics
        """
        try:
            metrics = {}

            # Basic regression metrics
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)

            # Mean Absolute Percentage Error (MAPE)
            # Handle division by zero
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics['mape'] = mape
            else:
                metrics['mape'] = float('inf')
                self.logger.warning(f"Cannot calculate MAPE for cluster {cluster_id}: all true values are zero")

            # Additional metrics
            residuals = y_true - y_pred
            metrics['mean_residual'] = np.mean(residuals)
            metrics['std_residual'] = np.std(residuals)
            metrics['max_error'] = np.max(np.abs(residuals))

            # Explained variance
            metrics['explained_variance'] = 1 - (np.var(residuals) / np.var(y_true))

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate metrics for cluster {cluster_id}: {str(e)}")
            # Return basic metrics if calculation fails
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2_score': -float('inf'),
                'mape': float('inf')
            }

    def _log_training_summary(self, model_type: str, trained_models: Dict[int, Any],
                            all_metrics: Dict[int, Dict[str, float]]):
        """
        Log summary of training results.

        Args:
            model_type: Type of models trained
            trained_models: Dictionary of successfully trained models
            all_metrics: Dictionary of performance metrics
        """
        if not all_metrics:
            self.logger.warning(f"No {model_type} models were successfully trained")
            return

        # Calculate summary statistics
        summary_metrics = {}
        for metric_name in ['rmse', 'mae', 'r2_score', 'mape']:
            values = [metrics.get(metric_name, float('nan')) for metrics in all_metrics.values()]
            valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]

            if valid_values:
                summary_metrics[f'avg_{metric_name}'] = np.mean(valid_values)
                summary_metrics[f'std_{metric_name}'] = np.std(valid_values)
                summary_metrics[f'min_{metric_name}'] = np.min(valid_values)
                summary_metrics[f'max_{metric_name}'] = np.max(valid_values)

        # Add training summary info
        summary_metrics['total_clusters_trained'] = len(trained_models)
        summary_metrics['total_samples_trained'] = sum(self.cluster_data_sizes.get(cid, 0) for cid in trained_models.keys())
        summary_metrics['avg_samples_per_cluster'] = summary_metrics['total_samples_trained'] / len(trained_models) if trained_models else 0

        # Log summary metrics
        self.logger.log_metrics(summary_metrics, prefix=f"{model_type}_training_summary")

        # Store training history
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'clusters_trained': list(trained_models.keys()),
            'summary_metrics': summary_metrics
        }
        self.training_history.append(training_record)

    def get_model_metrics(self, model_type: str = "timeseries") -> Dict[int, Dict[str, float]]:
        """
        Get performance metrics for trained models.

        Args:
            model_type: Type of model metrics to retrieve

        Returns:
            Dictionary mapping cluster_id to metrics
        """
        if model_type == "timeseries":
            return self.timeseries_metrics.copy()
        elif model_type == "resource":
            return self.resource_metrics.copy()
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

    def get_feature_importance(self, cluster_id: int, model_type: str = "timeseries") -> Optional[np.ndarray]:
        """
        Get feature importance for a specific cluster and model type.

        Args:
            cluster_id: Cluster identifier
            model_type: Type of model

        Returns:
            Feature importance array, or None if not available
        """
        if cluster_id in self.feature_importance:
            return self.feature_importance[cluster_id].get(model_type)
        return None

    def log_feature_importance(self, cluster_id: int, feature_names: Optional[List[str]] = None):
        """
        Log feature importance for a specific cluster to MLflow.

        Args:
            cluster_id: Cluster identifier
            feature_names: Optional list of feature names
        """
        if cluster_id not in self.feature_importance:
            self.logger.warning(f"No feature importance data available for cluster {cluster_id}")
            return

        try:
            for model_type, importance in self.feature_importance[cluster_id].items():
                # Log feature importance as metrics
                for i, imp_value in enumerate(importance):
                    feature_name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
                    metric_name = f"cluster_{cluster_id}_{model_type}_importance_{feature_name}"
                    self.logger.log_metrics({metric_name: float(imp_value)})

                # Log top features
                top_indices = np.argsort(importance)[-5:][::-1]  # Top 5 features
                top_features = {}
                for rank, idx in enumerate(top_indices):
                    feature_name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
                    top_features[f"top_{rank+1}_feature"] = feature_name
                    top_features[f"top_{rank+1}_importance"] = float(importance[idx])

                self.logger.log_parameters(top_features, prefix=f"cluster_{cluster_id}_{model_type}")

        except Exception as e:
            self.logger.error(f"Failed to log feature importance for cluster {cluster_id}: {str(e)}")

    def handle_insufficient_data_warning(self, cluster_id: int, data_size: int, model_type: str):
        """
        Handle and log warnings for clusters with insufficient data.

        Args:
            cluster_id: Cluster identifier
            data_size: Size of available data
            model_type: Type of model being trained
        """
        warning_msg = (
            f"Insufficient data for {model_type} model training in cluster {cluster_id}: "
            f"{data_size} samples (minimum required: {self.min_samples_per_cluster})"
        )

        # Log warning
        self.logger.warning(warning_msg)

        # Log as MLflow parameter for tracking
        warning_params = {
            f"cluster_{cluster_id}_insufficient_data": True,
            f"cluster_{cluster_id}_data_size": data_size,
            f"cluster_{cluster_id}_min_required": self.min_samples_per_cluster,
            f"cluster_{cluster_id}_model_type": model_type
        }
        self.logger.log_parameters(warning_params, prefix="warnings")

        # Store in training history
        warning_record = {
            'timestamp': datetime.now().isoformat(),
            'cluster_id': cluster_id,
            'warning_type': 'insufficient_data',
            'data_size': data_size,
            'min_required': self.min_samples_per_cluster,
            'model_type': model_type
        }
        self.training_history.append(warning_record)

    def fit_resource_models(self, cluster_data: Dict[int, Dict[str, np.ndarray]],
                           resource_data: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[str, float]]:
        """
        Fit LightGBM resource usage models for each cluster.

        Args:
            cluster_data: Dictionary mapping cluster_id to {'X': features, 'y': targets}
            resource_data: Dictionary mapping cluster_id to resource usage metrics
                          Expected format: {'processing_time': array, 'memory_usage': array, 'complexity': array}

        Returns:
            Dictionary mapping cluster_id to resource prediction metrics

        Raises:
            ModelTrainingError: If model training fails
        """
        self.logger.info("Starting resource usage model training for all clusters")

        trained_models = {}
        all_metrics = {}

        for cluster_id, data in cluster_data.items():
            if cluster_id not in resource_data:
                self.logger.warning(f"No resource data available for cluster {cluster_id}, skipping")
                continue

            try:
                metrics = self._fit_single_resource_model(cluster_id, data['X'], resource_data[cluster_id])
                if metrics is not None:
                    trained_models[cluster_id] = self.resource_models[cluster_id]
                    all_metrics[cluster_id] = metrics

            except Exception as e:
                self.logger.error(f"Failed to train resource model for cluster {cluster_id}: {str(e)}")
                # Continue with other clusters rather than failing completely
                continue

        self.logger.info(f"Successfully trained resource models for {len(trained_models)} clusters")

        # Log overall training summary
        self._log_training_summary("resource", trained_models, all_metrics)

        return all_metrics

    def predict_resources(self, cluster_assignments: np.ndarray, X: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Make resource usage predictions for new data batches using cluster-specific models.

        Args:
            cluster_assignments: Cluster assignments for each data point
            X: Feature matrix for prediction

        Returns:
            Tuple of (resource predictions dict, prediction metadata)

        Raises:
            PredictionError: If prediction fails
        """
        try:
            self.logger.info(f"Making resource usage predictions for {len(X)} samples")

            # Validate input data
            self._validate_prediction_data(cluster_assignments, X)

            # Initialize predictions dictionaries
            resource_predictions = {
                'processing_time': np.full(len(X), np.nan),
                'memory_usage': np.full(len(X), np.nan),
                'computational_complexity': np.full(len(X), np.nan)
            }

            prediction_metadata = {
                'cluster_predictions': {},
                'failed_predictions': [],
                'noise_point_predictions': 0,
                'successful_predictions': 0
            }

            # Get unique clusters in the batch
            unique_clusters = np.unique(cluster_assignments)

            for cluster_id in unique_clusters:
                # Skip noise points (cluster_id = -1)
                if cluster_id == -1:
                    noise_mask = cluster_assignments == -1
                    prediction_metadata['noise_point_predictions'] = np.sum(noise_mask)
                    self.logger.warning(f"Found {np.sum(noise_mask)} noise points, resource predictions will be NaN")
                    continue

                # Get data points for this cluster
                cluster_mask = cluster_assignments == cluster_id
                cluster_X = X[cluster_mask]

                if len(cluster_X) == 0:
                    continue

                try:
                    # Make resource predictions for this cluster
                    cluster_resource_pred = self._predict_single_cluster_resources(cluster_id, cluster_X)

                    # Assign predictions
                    for resource_type, predictions in cluster_resource_pred.items():
                        if resource_type in resource_predictions:
                            resource_predictions[resource_type][cluster_mask] = predictions

                    # Store metadata
                    prediction_metadata['cluster_predictions'][cluster_id] = {
                        'sample_count': len(cluster_X),
                        'resource_predictions': {
                            resource_type: {
                                'mean': np.mean(predictions),
                                'std': np.std(predictions),
                                'min': np.min(predictions),
                                'max': np.max(predictions)
                            }
                            for resource_type, predictions in cluster_resource_pred.items()
                        }
                    }

                    prediction_metadata['successful_predictions'] += len(cluster_X)

                except Exception as e:
                    self.logger.error(f"Failed to predict resources for cluster {cluster_id}: {str(e)}")
                    prediction_metadata['failed_predictions'].append({
                        'cluster_id': cluster_id,
                        'sample_count': len(cluster_X),
                        'error': str(e)
                    })

            # Log prediction summary
            self._log_resource_prediction_summary(prediction_metadata)

            self.logger.info(f"Completed resource predictions: {prediction_metadata['successful_predictions']} successful")

            return resource_predictions, prediction_metadata

        except Exception as e:
            raise PredictionError(
                f"Failed to make resource usage predictions: {str(e)}",
                prediction_type="resource",
                batch_size=len(X) if X is not None else 0
            )

    def predict_resources_single_cluster(self, cluster_id: int, X: np.ndarray) -> Dict[str, float]:
        """
        Make resource usage predictions for a single cluster.

        Args:
            cluster_id: Cluster identifier
            X: Feature matrix for the cluster

        Returns:
            Dictionary containing resource predictions

        Raises:
            PredictionError: If prediction fails
        """
        if cluster_id not in self.resource_models:
            raise PredictionError(
                f"No trained resource model available for cluster {cluster_id}",
                prediction_type="resource",
                cluster_id=cluster_id
            )

        try:
            cluster_resource_pred = self._predict_single_cluster_resources(cluster_id, X)

            # Calculate average predictions for the batch
            avg_predictions = {}
            for resource_type, predictions in cluster_resource_pred.items():
                avg_predictions[resource_type] = float(np.mean(predictions))

            return avg_predictions

        except Exception as e:
            raise PredictionError(
                f"Resource prediction failed for cluster {cluster_id}: {str(e)}",
                prediction_type="resource",
                cluster_id=cluster_id,
                batch_size=len(X)
            )

    def track_resource_prediction_accuracy(self, actual_resources: Dict[str, np.ndarray],
                                         predicted_resources: Dict[str, np.ndarray],
                                         cluster_assignments: np.ndarray, step: Optional[int] = None):
        """
        Track resource prediction accuracy over time for monitoring.

        Args:
            actual_resources: Dictionary of actual resource usage values
            predicted_resources: Dictionary of predicted resource usage values
            cluster_assignments: Cluster assignments for each prediction
            step: Optional step number for time series tracking
        """
        try:
            # Calculate overall accuracy metrics for each resource type
            overall_metrics = {}
            for resource_type in actual_resources.keys():
                if resource_type in predicted_resources:
                    actual = actual_resources[resource_type]
                    predicted = predicted_resources[resource_type]

                    # Filter out NaN values
                    valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
                    if np.sum(valid_mask) > 0:
                        actual_valid = actual[valid_mask]
                        predicted_valid = predicted[valid_mask]

                        metrics = self._calculate_resource_metrics(actual_valid, predicted_valid, resource_type)
                        overall_metrics.update({f"{resource_type}_{k}": v for k, v in metrics.items()})

            # Log overall metrics with step
            if overall_metrics:
                self.logger.log_metrics(overall_metrics, step=step, prefix="resource_prediction_accuracy_overall")

            # Calculate per-cluster accuracy
            unique_clusters = np.unique(cluster_assignments)
            cluster_accuracies = {}

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue

                cluster_mask = cluster_assignments == cluster_id
                if np.sum(cluster_mask) == 0:
                    continue

                cluster_metrics = {}
                for resource_type in actual_resources.keys():
                    if resource_type in predicted_resources:
                        actual = actual_resources[resource_type][cluster_mask]
                        predicted = predicted_resources[resource_type][cluster_mask]

                        # Filter out NaN values
                        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
                        if np.sum(valid_mask) > 0:
                            actual_valid = actual[valid_mask]
                            predicted_valid = predicted[valid_mask]

                            metrics = self._calculate_resource_metrics(actual_valid, predicted_valid, resource_type)
                            cluster_metrics.update({f"{resource_type}_{k}": v for k, v in metrics.items()})

                if cluster_metrics:
                    cluster_accuracies[cluster_id] = cluster_metrics

                    # Log cluster-specific metrics
                    self.logger.log_metrics(
                        cluster_metrics,
                        step=step,
                        prefix=f"resource_prediction_accuracy_cluster_{cluster_id}"
                    )

            # Store accuracy tracking
            accuracy_record = {
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'overall_metrics': overall_metrics,
                'cluster_accuracies': cluster_accuracies,
                'total_predictions': len(cluster_assignments),
                'clusters_evaluated': len(cluster_accuracies)
            }

            # Add to training history for tracking
            self.training_history.append({
                'type': 'resource_prediction_accuracy',
                **accuracy_record
            })

            self.logger.info(f"Tracked resource prediction accuracy for step {step}")

        except Exception as e:
            self.logger.error(f"Failed to track resource prediction accuracy: {str(e)}")

    def _fit_single_resource_model(self, cluster_id: int, X: np.ndarray,
                                 resource_data: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
        """
        Fit a single LightGBM resource usage model for a specific cluster.

        Args:
            cluster_id: Cluster identifier
            X: Feature matrix for the cluster
            resource_data: Dictionary containing resource usage targets

        Returns:
            Performance metrics dictionary, or None if training skipped

        Raises:
            ModelTrainingError: If model training fails
        """
        try:
            # Validate input data
            self._validate_resource_training_data(X, resource_data, cluster_id)

            # Check if we have sufficient data
            if len(X) < self.min_samples_per_cluster:
                self.handle_insufficient_data_warning(cluster_id, len(X), "resource")
                return None

            self.logger.info(f"Training resource usage model for cluster {cluster_id} with {len(X)} samples")

            # Create combined target vector from resource metrics
            # We'll predict a composite resource score and then decompose it
            y_resource = self._create_resource_target(resource_data)

            # Create and configure LightGBM model
            model = lgb.LGBMRegressor(**self.config.lgb_resource_params)

            # Fit the model
            model.fit(
                X, y_resource,
                eval_set=[(X, y_resource)],
                eval_names=['train'],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            # Store the trained model
            self.resource_models[cluster_id] = model

            # Calculate performance metrics
            y_pred = model.predict(X)
            metrics = self._calculate_resource_metrics(y_resource, y_pred, "composite_resource_score")

            # Store metrics
            self.resource_metrics[cluster_id] = metrics

            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                if cluster_id not in self.feature_importance:
                    self.feature_importance[cluster_id] = {}
                self.feature_importance[cluster_id]['resource'] = model.feature_importances_

            # Log model information
            self.logger.log_model_info(
                model_type="LightGBM_Resource",
                model_params=self.config.lgb_resource_params,
                training_metrics=metrics,
                cluster_id=cluster_id
            )

            # Save model artifact
            self.artifact_manager.save_model_artifact(
                model=model,
                artifact_name="resource_model",
                component="lightgbm_resource",
                cluster_id=cluster_id
            )

            self.logger.info(f"Successfully trained resource usage model for cluster {cluster_id}")
            return metrics

        except Exception as e:
            raise ModelTrainingError(
                f"Failed to train resource usage model for cluster {cluster_id}: {str(e)}",
                model_type="LightGBM_Resource",
                cluster_id=cluster_id,
                training_data_size=len(X) if X is not None else 0
            )

    def _predict_single_cluster_resources(self, cluster_id: int, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make resource predictions for a single cluster.

        Args:
            cluster_id: Cluster identifier
            X: Feature matrix for the cluster

        Returns:
            Dictionary mapping resource types to predictions

        Raises:
            PredictionError: If prediction fails
        """
        if cluster_id not in self.resource_models:
            raise PredictionError(
                f"No trained resource model available for cluster {cluster_id}",
                prediction_type="resource",
                cluster_id=cluster_id
            )

        model = self.resource_models[cluster_id]

        try:
            # Get composite resource score predictions
            composite_predictions = model.predict(X)

            # Decompose into individual resource predictions
            # This is a simplified approach - in practice, you might train separate models
            # or use multi-output regression
            resource_predictions = {
                'processing_time': composite_predictions * 0.4,  # 40% weight for processing time
                'memory_usage': composite_predictions * 0.35,    # 35% weight for memory usage
                'computational_complexity': composite_predictions * 0.25  # 25% weight for complexity
            }

            return resource_predictions

        except Exception as e:
            raise PredictionError(
                f"Resource model prediction failed for cluster {cluster_id}: {str(e)}",
                prediction_type="resource",
                cluster_id=cluster_id,
                batch_size=len(X)
            )

    def _validate_resource_training_data(self, X: np.ndarray, resource_data: Dict[str, np.ndarray], cluster_id: int):
        """
        Validate resource training data for model fitting.

        Args:
            X: Feature matrix
            resource_data: Dictionary of resource usage data
            cluster_id: Cluster identifier

        Raises:
            ModelTrainingError: If data validation fails
        """
        # Validate feature matrix
        self._validate_training_data(X, np.zeros(len(X)), cluster_id, "resource")  # Use dummy y for validation

        # Validate resource data
        if not resource_data:
            raise ModelTrainingError(
                f"Resource data cannot be empty for cluster {cluster_id}",
                model_type="LightGBM_Resource",
                cluster_id=cluster_id
            )

        required_resources = ['processing_time', 'memory_usage', 'computational_complexity']
        for resource_type in required_resources:
            if resource_type not in resource_data:
                raise ModelTrainingError(
                    f"Missing required resource type '{resource_type}' for cluster {cluster_id}",
                    model_type="LightGBM_Resource",
                    cluster_id=cluster_id
                )

            resource_values = resource_data[resource_type]
            if not isinstance(resource_values, np.ndarray):
                raise ModelTrainingError(
                    f"Resource data '{resource_type}' must be numpy array for cluster {cluster_id}",
                    model_type="LightGBM_Resource",
                    cluster_id=cluster_id
                )

            if len(resource_values) != len(X):
                raise ModelTrainingError(
                    f"Resource data '{resource_type}' length must match feature matrix for cluster {cluster_id}: "
                    f"resource_length={len(resource_values)}, X_length={len(X)}",
                    model_type="LightGBM_Resource",
                    cluster_id=cluster_id
                )

            if np.any(np.isnan(resource_values)) or np.any(np.isinf(resource_values)):
                raise ModelTrainingError(
                    f"Resource data '{resource_type}' contains NaN or infinite values for cluster {cluster_id}",
                    model_type="LightGBM_Resource",
                    cluster_id=cluster_id
                )

            if np.any(resource_values < 0):
                raise ModelTrainingError(
                    f"Resource data '{resource_type}' contains negative values for cluster {cluster_id}",
                    model_type="LightGBM_Resource",
                    cluster_id=cluster_id
                )

    def _create_resource_target(self, resource_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create a composite resource target from individual resource metrics.

        Args:
            resource_data: Dictionary containing resource usage arrays

        Returns:
            Composite resource target array
        """
        # Normalize each resource metric to [0, 1] range
        normalized_resources = {}

        for resource_type, values in resource_data.items():
            if len(values) > 0:
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val > min_val:
                    normalized_resources[resource_type] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_resources[resource_type] = np.zeros_like(values)

        # Create weighted composite score
        weights = {
            'processing_time': 0.4,
            'memory_usage': 0.35,
            'computational_complexity': 0.25
        }

        composite_score = np.zeros(len(next(iter(resource_data.values()))))

        for resource_type, weight in weights.items():
            if resource_type in normalized_resources:
                composite_score += weight * normalized_resources[resource_type]

        return composite_score

    def _calculate_resource_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, resource_type: str) -> Dict[str, float]:
        """
        Calculate resource prediction performance metrics.

        Args:
            y_true: True resource values
            y_pred: Predicted resource values
            resource_type: Type of resource being evaluated

        Returns:
            Dictionary of performance metrics
        """
        try:
            metrics = {}

            # Basic regression metrics
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)

            # Mean Absolute Percentage Error (MAPE)
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                metrics['mape'] = mape
            else:
                metrics['mape'] = float('inf')

            # Resource-specific metrics
            residuals = y_true - y_pred
            metrics['mean_residual'] = np.mean(residuals)
            metrics['std_residual'] = np.std(residuals)
            metrics['max_error'] = np.max(np.abs(residuals))

            # Explained variance
            metrics['explained_variance'] = 1 - (np.var(residuals) / np.var(y_true))

            # Resource efficiency metrics
            if resource_type in ['processing_time', 'memory_usage', 'computational_complexity']:
                # Lower values are better for these metrics
                metrics['efficiency_score'] = 1.0 / (1.0 + metrics['mae'])  # Higher is better

                # Prediction bias (positive means over-prediction)
                metrics['prediction_bias'] = np.mean(y_pred - y_true)

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate resource metrics for {resource_type}: {str(e)}")
            # Return basic metrics if calculation fails
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2_score': -float('inf'),
                'mape': float('inf')
            }

    def _log_resource_prediction_summary(self, prediction_metadata: Dict[str, Any]):
        """
        Log resource prediction summary metrics.

        Args:
            prediction_metadata: Metadata from resource prediction operation
        """
        try:
            summary_metrics = {
                'total_clusters_predicted': len(prediction_metadata['cluster_predictions']),
                'successful_predictions': prediction_metadata['successful_predictions'],
                'failed_predictions': len(prediction_metadata['failed_predictions']),
                'noise_point_predictions': prediction_metadata['noise_point_predictions']
            }

            # Add resource-specific summary stats
            if prediction_metadata['cluster_predictions']:
                for resource_type in ['processing_time', 'memory_usage', 'computational_complexity']:
                    resource_means = []
                    for cluster_stats in prediction_metadata['cluster_predictions'].values():
                        if resource_type in cluster_stats['resource_predictions']:
                            resource_means.append(cluster_stats['resource_predictions'][resource_type]['mean'])

                    if resource_means:
                        summary_metrics.update({
                            f'avg_predicted_{resource_type}': np.mean(resource_means),
                            f'std_predicted_{resource_type}': np.std(resource_means),
                            f'min_predicted_{resource_type}': np.min(resource_means),
                            f'max_predicted_{resource_type}': np.max(resource_means)
                        })

            self.logger.log_metrics(summary_metrics, prefix="resource_prediction_summary")

        except Exception as e:
            self.logger.error(f"Failed to log resource prediction summary: {str(e)}")

    def generate_resource_optimization_recommendations(self,
                                                    predicted_resources: Dict[str, np.ndarray],
                                                    cluster_assignments: np.ndarray,
                                                    actual_resources: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Generate resource optimization recommendations based on predictions and actual usage.

        Args:
            predicted_resources: Dictionary of predicted resource usage values
            cluster_assignments: Cluster assignments for each prediction
            actual_resources: Optional dictionary of actual resource usage values

        Returns:
            Dictionary containing optimization recommendations
        """
        try:
            self.logger.info("Generating resource optimization recommendations")

            recommendations = {
                'timestamp': datetime.now().isoformat(),
                'cluster_recommendations': {},
                'overall_recommendations': {},
                'optimization_opportunities': []
            }

            # Analyze per-cluster resource usage
            unique_clusters = np.unique(cluster_assignments)
            cluster_stats = {}

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue

                cluster_mask = cluster_assignments == cluster_id
                cluster_recs = {
                    'cluster_id': cluster_id,
                    'sample_count': np.sum(cluster_mask),
                    'resource_analysis': {},
                    'recommendations': []
                }

                # Analyze each resource type
                for resource_type in predicted_resources.keys():
                    predicted = predicted_resources[resource_type][cluster_mask]
                    valid_predicted = predicted[~np.isnan(predicted)]

                    if len(valid_predicted) > 0:
                        resource_stats = {
                            'mean_predicted': np.mean(valid_predicted),
                            'std_predicted': np.std(valid_predicted),
                            'max_predicted': np.max(valid_predicted),
                            'min_predicted': np.min(valid_predicted),
                            'percentile_95': np.percentile(valid_predicted, 95),
                            'percentile_75': np.percentile(valid_predicted, 75)
                        }

                        # Add actual vs predicted comparison if available
                        if actual_resources and resource_type in actual_resources:
                            actual = actual_resources[resource_type][cluster_mask]
                            valid_actual = actual[~np.isnan(actual)]

                            if len(valid_actual) > 0 and len(valid_actual) == len(valid_predicted):
                                resource_stats.update({
                                    'mean_actual': np.mean(valid_actual),
                                    'prediction_error': np.mean(np.abs(valid_predicted - valid_actual)),
                                    'prediction_bias': np.mean(valid_predicted - valid_actual),
                                    'accuracy_score': 1.0 - (np.mean(np.abs(valid_predicted - valid_actual)) / np.mean(valid_actual))
                                })

                        cluster_recs['resource_analysis'][resource_type] = resource_stats

                        # Generate specific recommendations
                        recommendations_list = self._generate_resource_recommendations(
                            resource_type, resource_stats, cluster_id
                        )
                        cluster_recs['recommendations'].extend(recommendations_list)

                cluster_stats[cluster_id] = cluster_recs
                recommendations['cluster_recommendations'][cluster_id] = cluster_recs

            # Generate overall recommendations
            recommendations['overall_recommendations'] = self._generate_overall_recommendations(cluster_stats)

            # Identify optimization opportunities
            recommendations['optimization_opportunities'] = self._identify_optimization_opportunities(
                cluster_stats, predicted_resources, cluster_assignments
            )

            # Log recommendations summary
            self._log_optimization_recommendations(recommendations)

            self.logger.info("Successfully generated resource optimization recommendations")
            return recommendations

        except Exception as e:
            self.logger.error(f"Failed to generate resource optimization recommendations: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def track_actual_vs_predicted_resources(self,
                                          actual_resources: Dict[str, np.ndarray],
                                          predicted_resources: Dict[str, np.ndarray],
                                          cluster_assignments: np.ndarray,
                                          batch_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Track actual vs predicted resource usage for continuous monitoring.

        Args:
            actual_resources: Dictionary of actual resource usage values
            predicted_resources: Dictionary of predicted resource usage values
            cluster_assignments: Cluster assignments for each prediction
            batch_id: Optional batch identifier for tracking

        Returns:
            Dictionary containing tracking results
        """
        try:
            self.logger.info(f"Tracking actual vs predicted resources for batch {batch_id}")

            tracking_results = {
                'timestamp': datetime.now().isoformat(),
                'batch_id': batch_id,
                'overall_tracking': {},
                'cluster_tracking': {},
                'drift_indicators': {}
            }

            # Overall tracking across all clusters
            overall_metrics = {}
            for resource_type in actual_resources.keys():
                if resource_type in predicted_resources:
                    actual = actual_resources[resource_type]
                    predicted = predicted_resources[resource_type]

                    # Filter out NaN values
                    valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
                    if np.sum(valid_mask) > 0:
                        actual_valid = actual[valid_mask]
                        predicted_valid = predicted[valid_mask]

                        metrics = self._calculate_tracking_metrics(actual_valid, predicted_valid, resource_type)
                        overall_metrics[resource_type] = metrics

            tracking_results['overall_tracking'] = overall_metrics

            # Per-cluster tracking
            unique_clusters = np.unique(cluster_assignments)
            cluster_tracking = {}

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue

                cluster_mask = cluster_assignments == cluster_id
                cluster_metrics = {}

                for resource_type in actual_resources.keys():
                    if resource_type in predicted_resources:
                        actual = actual_resources[resource_type][cluster_mask]
                        predicted = predicted_resources[resource_type][cluster_mask]

                        # Filter out NaN values
                        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
                        if np.sum(valid_mask) > 0:
                            actual_valid = actual[valid_mask]
                            predicted_valid = predicted[valid_mask]

                            metrics = self._calculate_tracking_metrics(actual_valid, predicted_valid, resource_type)
                            cluster_metrics[resource_type] = metrics

                if cluster_metrics:
                    cluster_tracking[cluster_id] = cluster_metrics

            tracking_results['cluster_tracking'] = cluster_tracking

            # Detect drift indicators
            tracking_results['drift_indicators'] = self._detect_resource_drift(
                overall_metrics, cluster_tracking
            )

            # Store tracking history
            tracking_record = {
                'type': 'resource_tracking',
                **tracking_results
            }
            self.training_history.append(tracking_record)

            # Log tracking metrics
            self._log_resource_tracking_metrics(tracking_results)

            self.logger.info(f"Successfully tracked actual vs predicted resources for batch {batch_id}")
            return tracking_results

        except Exception as e:
            self.logger.error(f"Failed to track actual vs predicted resources: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'batch_id': batch_id
            }

    def _generate_resource_recommendations(self, resource_type: str, resource_stats: Dict[str, float],
                                         cluster_id: int) -> List[Dict[str, Any]]:
        """
        Generate specific recommendations for a resource type and cluster.

        Args:
            resource_type: Type of resource (processing_time, memory_usage, etc.)
            resource_stats: Statistics for the resource
            cluster_id: Cluster identifier

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        # High resource usage recommendations
        if resource_type == 'processing_time':
            if resource_stats['percentile_95'] > resource_stats['mean_predicted'] * 2:
                recommendations.append({
                    'type': 'performance_optimization',
                    'priority': 'high',
                    'message': f"Cluster {cluster_id} shows high processing time variance. Consider optimizing algorithms or increasing compute resources.",
                    'metric': 'processing_time_variance',
                    'value': resource_stats['std_predicted']
                })

        elif resource_type == 'memory_usage':
            if resource_stats['max_predicted'] > resource_stats['mean_predicted'] * 3:
                recommendations.append({
                    'type': 'memory_optimization',
                    'priority': 'high',
                    'message': f"Cluster {cluster_id} has high memory usage spikes. Consider memory optimization or batch size reduction.",
                    'metric': 'memory_usage_spike',
                    'value': resource_stats['max_predicted']
                })

        elif resource_type == 'computational_complexity':
            if resource_stats['mean_predicted'] > 0.8:  # Assuming normalized scale
                recommendations.append({
                    'type': 'complexity_optimization',
                    'priority': 'medium',
                    'message': f"Cluster {cluster_id} has high computational complexity. Consider algorithm simplification.",
                    'metric': 'computational_complexity',
                    'value': resource_stats['mean_predicted']
                })

        # Prediction accuracy recommendations
        if 'accuracy_score' in resource_stats:
            if resource_stats['accuracy_score'] < 0.7:
                recommendations.append({
                    'type': 'model_improvement',
                    'priority': 'medium',
                    'message': f"Resource prediction accuracy for {resource_type} in cluster {cluster_id} is low. Consider model retraining.",
                    'metric': 'prediction_accuracy',
                    'value': resource_stats['accuracy_score']
                })

        # Prediction bias recommendations
        if 'prediction_bias' in resource_stats:
            if abs(resource_stats['prediction_bias']) > resource_stats['mean_actual'] * 0.2:
                bias_direction = "over-predicting" if resource_stats['prediction_bias'] > 0 else "under-predicting"
                recommendations.append({
                    'type': 'prediction_bias',
                    'priority': 'medium',
                    'message': f"Model is {bias_direction} {resource_type} for cluster {cluster_id}. Consider bias correction.",
                    'metric': 'prediction_bias',
                    'value': resource_stats['prediction_bias']
                })

        return recommendations

    def _generate_overall_recommendations(self, cluster_stats: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate overall recommendations across all clusters.

        Args:
            cluster_stats: Statistics for all clusters

        Returns:
            Dictionary of overall recommendations
        """
        overall_recs = {
            'resource_distribution': {},
            'cluster_efficiency': {},
            'recommendations': []
        }

        if not cluster_stats:
            return overall_recs

        # Analyze resource distribution across clusters
        for resource_type in ['processing_time', 'memory_usage', 'computational_complexity']:
            resource_means = []
            for cluster_data in cluster_stats.values():
                if resource_type in cluster_data['resource_analysis']:
                    resource_means.append(cluster_data['resource_analysis'][resource_type]['mean_predicted'])

            if resource_means:
                overall_recs['resource_distribution'][resource_type] = {
                    'mean_across_clusters': np.mean(resource_means),
                    'std_across_clusters': np.std(resource_means),
                    'min_cluster_usage': np.min(resource_means),
                    'max_cluster_usage': np.max(resource_means),
                    'usage_variance': np.var(resource_means)
                }

                # Generate recommendations based on distribution
                if np.std(resource_means) > np.mean(resource_means) * 0.5:
                    overall_recs['recommendations'].append({
                        'type': 'load_balancing',
                        'priority': 'high',
                        'message': f"High variance in {resource_type} across clusters suggests load balancing opportunities.",
                        'metric': f'{resource_type}_variance',
                        'value': np.std(resource_means)
                    })

        # Cluster efficiency analysis
        cluster_efficiencies = []
        for cluster_id, cluster_data in cluster_stats.items():
            efficiency_score = 0
            resource_count = 0

            for resource_type in cluster_data['resource_analysis']:
                stats = cluster_data['resource_analysis'][resource_type]
                if 'accuracy_score' in stats:
                    efficiency_score += stats['accuracy_score']
                    resource_count += 1

            if resource_count > 0:
                cluster_efficiencies.append(efficiency_score / resource_count)

        if cluster_efficiencies:
            overall_recs['cluster_efficiency'] = {
                'mean_efficiency': np.mean(cluster_efficiencies),
                'min_efficiency': np.min(cluster_efficiencies),
                'max_efficiency': np.max(cluster_efficiencies),
                'efficiency_std': np.std(cluster_efficiencies)
            }

            if np.mean(cluster_efficiencies) < 0.8:
                overall_recs['recommendations'].append({
                    'type': 'overall_optimization',
                    'priority': 'high',
                    'message': "Overall resource prediction efficiency is low. Consider pipeline optimization.",
                    'metric': 'mean_efficiency',
                    'value': np.mean(cluster_efficiencies)
                })

        return overall_recs

    def _identify_optimization_opportunities(self, cluster_stats: Dict[int, Dict[str, Any]],
                                           predicted_resources: Dict[str, np.ndarray],
                                           cluster_assignments: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify specific optimization opportunities.

        Args:
            cluster_stats: Statistics for all clusters
            predicted_resources: Predicted resource usage
            cluster_assignments: Cluster assignments

        Returns:
            List of optimization opportunities
        """
        opportunities = []

        # Identify clusters with consistently high resource usage
        high_usage_clusters = []
        for cluster_id, cluster_data in cluster_stats.items():
            high_usage_count = 0
            for resource_type in cluster_data['resource_analysis']:
                stats = cluster_data['resource_analysis'][resource_type]
                if stats['mean_predicted'] > stats.get('percentile_75', stats['mean_predicted']):
                    high_usage_count += 1

            if high_usage_count >= 2:  # High usage in at least 2 resource types
                high_usage_clusters.append(cluster_id)

        if high_usage_clusters:
            opportunities.append({
                'type': 'cluster_optimization',
                'priority': 'high',
                'description': f"Clusters {high_usage_clusters} show consistently high resource usage",
                'clusters_affected': high_usage_clusters,
                'recommendation': "Consider cluster-specific optimization or resource allocation"
            })

        # Identify resource imbalances
        resource_imbalances = {}
        for resource_type in predicted_resources.keys():
            cluster_means = []
            for cluster_id in np.unique(cluster_assignments):
                if cluster_id != -1 and cluster_id in cluster_stats:
                    if resource_type in cluster_stats[cluster_id]['resource_analysis']:
                        cluster_means.append(cluster_stats[cluster_id]['resource_analysis'][resource_type]['mean_predicted'])

            if len(cluster_means) > 1:
                imbalance_ratio = np.max(cluster_means) / np.min(cluster_means) if np.min(cluster_means) > 0 else float('inf')
                if imbalance_ratio > 3.0:  # 3x difference between clusters
                    resource_imbalances[resource_type] = imbalance_ratio

        if resource_imbalances:
            opportunities.append({
                'type': 'resource_balancing',
                'priority': 'medium',
                'description': f"Resource imbalances detected: {resource_imbalances}",
                'imbalances': resource_imbalances,
                'recommendation': "Consider workload redistribution or cluster rebalancing"
            })

        return opportunities

    def _calculate_tracking_metrics(self, actual: np.ndarray, predicted: np.ndarray, resource_type: str) -> Dict[str, float]:
        """
        Calculate tracking metrics for actual vs predicted comparison.

        Args:
            actual: Actual resource values
            predicted: Predicted resource values
            resource_type: Type of resource

        Returns:
            Dictionary of tracking metrics
        """
        metrics = {}

        # Basic accuracy metrics
        metrics['mae'] = mean_absolute_error(actual, predicted)
        metrics['rmse'] = np.sqrt(mean_squared_error(actual, predicted))
        metrics['r2_score'] = r2_score(actual, predicted)

        # Percentage-based metrics
        if np.mean(actual) > 0:
            metrics['mape'] = np.mean(np.abs((actual - predicted) / actual)) * 100
            metrics['accuracy_percentage'] = (1 - metrics['mae'] / np.mean(actual)) * 100
        else:
            metrics['mape'] = float('inf')
            metrics['accuracy_percentage'] = 0.0

        # Bias and variance
        metrics['bias'] = np.mean(predicted - actual)
        metrics['variance'] = np.var(predicted - actual)

        # Correlation
        if len(actual) > 1:
            correlation = np.corrcoef(actual, predicted)[0, 1]
            metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics['correlation'] = 0.0

        # Resource-specific metrics
        if resource_type in ['processing_time', 'memory_usage']:
            # Efficiency metrics (lower is better for these resources)
            metrics['efficiency_improvement'] = (np.mean(actual) - np.mean(predicted)) / np.mean(actual) * 100
        elif resource_type == 'computational_complexity':
            # Complexity reduction potential
            metrics['complexity_reduction_potential'] = np.mean(actual) - np.mean(predicted)

        return metrics

    def _detect_resource_drift(self, overall_metrics: Dict[str, Dict[str, float]],
                             cluster_tracking: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """
        Detect drift indicators in resource predictions.

        Args:
            overall_metrics: Overall tracking metrics
            cluster_tracking: Per-cluster tracking metrics

        Returns:
            Dictionary of drift indicators
        """
        drift_indicators = {
            'overall_drift': {},
            'cluster_drift': {},
            'drift_alerts': []
        }

        # Overall drift detection
        for resource_type, metrics in overall_metrics.items():
            drift_score = 0

            # High MAPE indicates drift
            if metrics.get('mape', 0) > 20:  # 20% MAPE threshold
                drift_score += 1

            # Low correlation indicates drift
            if metrics.get('correlation', 1) < 0.7:
                drift_score += 1

            # High bias indicates systematic drift
            if abs(metrics.get('bias', 0)) > np.abs(metrics.get('mae', 1)) * 0.5:
                drift_score += 1

            drift_indicators['overall_drift'][resource_type] = {
                'drift_score': drift_score,
                'drift_level': 'high' if drift_score >= 2 else 'medium' if drift_score == 1 else 'low'
            }

            if drift_score >= 2:
                drift_indicators['drift_alerts'].append({
                    'type': 'overall_drift',
                    'resource_type': resource_type,
                    'severity': 'high',
                    'message': f"High drift detected in {resource_type} predictions"
                })

        # Per-cluster drift detection
        for cluster_id, cluster_metrics in cluster_tracking.items():
            cluster_drift = {}
            for resource_type, metrics in cluster_metrics.items():
                drift_score = 0

                if metrics.get('mape', 0) > 25:  # Higher threshold for individual clusters
                    drift_score += 1
                if metrics.get('correlation', 1) < 0.6:
                    drift_score += 1
                if abs(metrics.get('bias', 0)) > np.abs(metrics.get('mae', 1)) * 0.6:
                    drift_score += 1

                cluster_drift[resource_type] = {
                    'drift_score': drift_score,
                    'drift_level': 'high' if drift_score >= 2 else 'medium' if drift_score == 1 else 'low'
                }

                if drift_score >= 2:
                    drift_indicators['drift_alerts'].append({
                        'type': 'cluster_drift',
                        'cluster_id': cluster_id,
                        'resource_type': resource_type,
                        'severity': 'high',
                        'message': f"High drift detected in {resource_type} predictions for cluster {cluster_id}"
                    })

            drift_indicators['cluster_drift'][cluster_id] = cluster_drift

        return drift_indicators

    def _log_optimization_recommendations(self, recommendations: Dict[str, Any]):
        """
        Log optimization recommendations to MLflow.

        Args:
            recommendations: Recommendations dictionary
        """
        try:
            # Log summary metrics
            summary_metrics = {
                'total_clusters_analyzed': len(recommendations['cluster_recommendations']),
                'total_recommendations': sum(len(cluster_data['recommendations'])
                                           for cluster_data in recommendations['cluster_recommendations'].values()),
                'optimization_opportunities': len(recommendations['optimization_opportunities'])
            }

            # Count recommendations by priority
            priority_counts = {'high': 0, 'medium': 0, 'low': 0}
            for cluster_data in recommendations['cluster_recommendations'].values():
                for rec in cluster_data['recommendations']:
                    priority = rec.get('priority', 'medium')
                    priority_counts[priority] += 1

            summary_metrics.update({
                'high_priority_recommendations': priority_counts['high'],
                'medium_priority_recommendations': priority_counts['medium'],
                'low_priority_recommendations': priority_counts['low']
            })

            self.logger.log_metrics(summary_metrics, prefix="resource_optimization")

        except Exception as e:
            self.logger.error(f"Failed to log optimization recommendations: {str(e)}")

    def _log_resource_tracking_metrics(self, tracking_results: Dict[str, Any]):
        """
        Log resource tracking metrics to MLflow.

        Args:
            tracking_results: Tracking results dictionary
        """
        try:
            # Log overall tracking metrics
            for resource_type, metrics in tracking_results['overall_tracking'].items():
                metric_dict = {f"{resource_type}_{k}": v for k, v in metrics.items()}
                self.logger.log_metrics(metric_dict, prefix="resource_tracking_overall")

            # Log drift indicators
            drift_summary = {
                'total_drift_alerts': len(tracking_results['drift_indicators']['drift_alerts']),
                'high_drift_resources': sum(1 for alert in tracking_results['drift_indicators']['drift_alerts']
                                          if alert['severity'] == 'high')
            }
            self.logger.log_metrics(drift_summary, prefix="resource_drift")

        except Exception as e:
            self.logger.error(f"Failed to log resource tracking metrics: {str(e)}")

    def generate_resource_usage_report(self,
                                      predicted_resources: Optional[Dict[str, np.ndarray]] = None,
                                      actual_resources: Optional[Dict[str, np.ndarray]] = None,
                                      cluster_assignments: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive resource usage report with visualizations.

        Args:
            predicted_resources: Optional predicted resource usage values
            actual_resources: Optional actual resource usage values
            cluster_assignments: Optional cluster assignments

        Returns:
            Dictionary containing report data and visualization paths
        """
        try:
            self.logger.info("Generating resource usage performance report")

            report = {
                'timestamp': datetime.now().isoformat(),
                'model_summary': self._generate_resource_model_summary(),
                'performance_metrics': self.get_model_metrics("resource"),
                'visualizations': {},
                'feature_importance_summary': self._generate_resource_feature_importance_summary()
            }

            # Add validation analysis if data provided
            if predicted_resources is not None and actual_resources is not None and cluster_assignments is not None:
                report['validation_analysis'] = self._create_resource_validation_analysis(
                    predicted_resources, actual_resources, cluster_assignments
                )

            # Generate optimization recommendations if data provided
            if predicted_resources is not None and cluster_assignments is not None:
                report['optimization_recommendations'] = self.generate_resource_optimization_recommendations(
                    predicted_resources, cluster_assignments, actual_resources
                )

            # Generate visualizations
            report['visualizations'] = self._generate_resource_usage_visualizations(
                predicted_resources, actual_resources, cluster_assignments
            )

            # Save report as artifact
            report_html = self._create_resource_html_report(report)
            report_path = self.artifact_manager.save_report_artifact(
                report_content=report_html,
                filename="resource_usage_report.html",
                subfolder="resource_models"
            )

            report['report_path'] = report_path

            self.logger.info("Successfully generated resource usage performance report")
            return report

        except Exception as e:
            self.logger.error(f"Failed to generate resource usage report: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _generate_resource_model_summary(self) -> Dict[str, Any]:
        """Generate summary of trained resource models."""
        return {
            'total_resource_models_trained': len(self.resource_models),
            'clusters_with_resource_models': list(self.resource_models.keys()),
            'total_samples_trained': sum(self.cluster_data_sizes.values()),
            'avg_samples_per_cluster': np.mean(list(self.cluster_data_sizes.values())) if self.cluster_data_sizes else 0,
            'min_samples_per_cluster': min(self.cluster_data_sizes.values()) if self.cluster_data_sizes else 0,
            'max_samples_per_cluster': max(self.cluster_data_sizes.values()) if self.cluster_data_sizes else 0,
            'resource_model_parameters': self.config.lgb_resource_params
        }

    def _generate_resource_feature_importance_summary(self) -> Dict[str, Any]:
        """Generate summary of feature importance for resource models."""
        if not self.feature_importance:
            return {'available': False}

        summary = {'available': True, 'cluster_summaries': {}}

        for cluster_id, importance_data in self.feature_importance.items():
            if 'resource' in importance_data:
                importance = importance_data['resource']
                top_5_indices = np.argsort(importance)[-5:][::-1]

                summary['cluster_summaries'][cluster_id] = {
                    'total_features': len(importance),
                    'top_5_features': {
                        f'feature_{idx}': float(importance[idx])
                        for idx in top_5_indices
                    },
                    'importance_stats': {
                        'mean': float(np.mean(importance)),
                        'std': float(np.std(importance)),
                        'max': float(np.max(importance)),
                        'min': float(np.min(importance))
                    }
                }

        return summary

    def _create_resource_validation_analysis(self,
                                           predicted_resources: Dict[str, np.ndarray],
                                           actual_resources: Dict[str, np.ndarray],
                                           cluster_assignments: np.ndarray) -> Dict[str, Any]:
        """
        Create validation analysis for resource predictions.

        Args:
            predicted_resources: Predicted resource usage values
            actual_resources: Actual resource usage values
            cluster_assignments: Cluster assignments

        Returns:
            Dictionary containing validation analysis results
        """
        try:
            analysis = {
                'overall_analysis': {},
                'cluster_analysis': {},
                'resource_type_analysis': {}
            }

            # Overall analysis across all resource types
            for resource_type in predicted_resources.keys():
                if resource_type in actual_resources:
                    predicted = predicted_resources[resource_type]
                    actual = actual_resources[resource_type]

                    # Filter out NaN values
                    valid_mask = ~(np.isnan(predicted) | np.isnan(actual))
                    if np.sum(valid_mask) > 0:
                        predicted_valid = predicted[valid_mask]
                        actual_valid = actual[valid_mask]

                        analysis['resource_type_analysis'][resource_type] = {
                            'correlation': np.corrcoef(predicted_valid, actual_valid)[0, 1] if len(predicted_valid) > 1 else 0,
                            'mae': mean_absolute_error(actual_valid, predicted_valid),
                            'rmse': np.sqrt(mean_squared_error(actual_valid, predicted_valid)),
                            'r2_score': r2_score(actual_valid, predicted_valid),
                            'bias': np.mean(predicted_valid - actual_valid),
                            'prediction_efficiency': 1.0 - (np.mean(np.abs(predicted_valid - actual_valid)) / np.mean(actual_valid))
                        }

            # Per-cluster analysis
            unique_clusters = np.unique(cluster_assignments)
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue

                cluster_mask = cluster_assignments == cluster_id
                cluster_analysis = {}

                for resource_type in predicted_resources.keys():
                    if resource_type in actual_resources:
                        predicted = predicted_resources[resource_type][cluster_mask]
                        actual = actual_resources[resource_type][cluster_mask]

                        # Filter out NaN values
                        valid_mask = ~(np.isnan(predicted) | np.isnan(actual))
                        if np.sum(valid_mask) > 0:
                            predicted_valid = predicted[valid_mask]
                            actual_valid = actual[valid_mask]

                            if len(predicted_valid) > 1:
                                cluster_analysis[resource_type] = {
                                    'sample_count': len(predicted_valid),
                                    'correlation': np.corrcoef(predicted_valid, actual_valid)[0, 1],
                                    'mae': mean_absolute_error(actual_valid, predicted_valid),
                                    'bias': np.mean(predicted_valid - actual_valid),
                                    'accuracy_percentage': (1 - mean_absolute_error(actual_valid, predicted_valid) / np.mean(actual_valid)) * 100
                                }

                if cluster_analysis:
                    analysis['cluster_analysis'][cluster_id] = cluster_analysis

            # Overall summary
            if analysis['resource_type_analysis']:
                overall_mae = np.mean([metrics['mae'] for metrics in analysis['resource_type_analysis'].values()])
                overall_r2 = np.mean([metrics['r2_score'] for metrics in analysis['resource_type_analysis'].values()])

                analysis['overall_analysis'] = {
                    'average_mae_across_resources': overall_mae,
                    'average_r2_across_resources': overall_r2,
                    'total_resource_types_analyzed': len(analysis['resource_type_analysis']),
                    'total_clusters_analyzed': len(analysis['cluster_analysis'])
                }

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to create resource validation analysis: {str(e)}")
            return {'error': str(e)}

    def _generate_resource_usage_visualizations(self,
                                              predicted_resources: Optional[Dict[str, np.ndarray]] = None,
                                              actual_resources: Optional[Dict[str, np.ndarray]] = None,
                                              cluster_assignments: Optional[np.ndarray] = None) -> Dict[str, str]:
        """
        Generate resource usage visualizations using seaborn.

        Returns:
            Dictionary mapping visualization names to artifact paths
        """
        visualizations = {}

        try:
            # Set up the custom color palette
            colors = ['black', 'white', 'darkgray', '#0d4f01']  # Dark green
            sns.set_palette(colors)
            plt.style.use('default')

            # 1. Resource Model Performance Summary Plot
            if self.resource_metrics:
                fig = self._create_resource_model_performance_plot()
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="resource_model_performance_summary.png",
                    subfolder="resource_models"
                )
                visualizations['resource_model_performance_summary'] = viz_path
                plt.close(fig)

            # 2. Resource Feature Importance Plot
            if self.feature_importance:
                fig = self._create_resource_feature_importance_plot()
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="resource_feature_importance.png",
                    subfolder="resource_models"
                )
                visualizations['resource_feature_importance'] = viz_path
                plt.close(fig)

            # 3. Resource Usage Distribution Plot
            if predicted_resources is not None and cluster_assignments is not None:
                fig = self._create_resource_usage_distribution_plot(predicted_resources, cluster_assignments)
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="resource_usage_distribution.png",
                    subfolder="resource_models"
                )
                visualizations['resource_usage_distribution'] = viz_path
                plt.close(fig)

            # 4. Predicted vs Actual Resource Usage (if actual data provided)
            if predicted_resources is not None and actual_resources is not None and cluster_assignments is not None:
                fig = self._create_resource_prediction_comparison_plot(
                    predicted_resources, actual_resources, cluster_assignments
                )
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="resource_prediction_comparison.png",
                    subfolder="resource_models"
                )
                visualizations['resource_prediction_comparison'] = viz_path
                plt.close(fig)

            # 5. Resource Usage Trends by Cluster
            if predicted_resources is not None and cluster_assignments is not None:
                fig = self._create_resource_trends_by_cluster_plot(predicted_resources, cluster_assignments)
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="resource_trends_by_cluster.png",
                    subfolder="resource_models"
                )
                visualizations['resource_trends_by_cluster'] = viz_path
                plt.close(fig)

            # 6. Resource Optimization Opportunities
            if predicted_resources is not None and cluster_assignments is not None:
                fig = self._create_resource_optimization_plot(predicted_resources, cluster_assignments)
                viz_path = self.artifact_manager.save_visualization(
                    figure=fig,
                    filename="resource_optimization_opportunities.png",
                    subfolder="resource_models"
                )
                visualizations['resource_optimization_opportunities'] = viz_path
                plt.close(fig)

        except Exception as e:
            self.logger.error(f"Failed to generate resource usage visualizations: {str(e)}")
            visualizations['error'] = str(e)

        return visualizations

    def _create_resource_model_performance_plot(self):
        """Create resource model performance summary plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Resource Model Performance Summary', fontsize=16, fontweight='bold')

        # Extract metrics for all clusters
        cluster_ids = list(self.resource_metrics.keys())
        rmse_values = [self.resource_metrics[cid]['rmse'] for cid in cluster_ids]
        mae_values = [self.resource_metrics[cid]['mae'] for cid in cluster_ids]
        r2_values = [self.resource_metrics[cid]['r2_score'] for cid in cluster_ids]
        efficiency_values = [self.resource_metrics[cid].get('efficiency_score', 0) for cid in cluster_ids]

        # RMSE by cluster
        sns.barplot(x=cluster_ids, y=rmse_values, ax=axes[0, 0], color='#0d4f01')
        axes[0, 0].set_title('RMSE by Cluster')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('RMSE')

        # MAE by cluster
        sns.barplot(x=cluster_ids, y=mae_values, ax=axes[0, 1], color='darkgray')
        axes[0, 1].set_title('MAE by Cluster')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('MAE')

        # R² by cluster
        sns.barplot(x=cluster_ids, y=r2_values, ax=axes[1, 0], color='#0d4f01')
        axes[1, 0].set_title('R² Score by Cluster')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('R² Score')

        # Efficiency Score by cluster
        sns.barplot(x=cluster_ids, y=efficiency_values, ax=axes[1, 1], color='darkgray')
        axes[1, 1].set_title('Efficiency Score by Cluster')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Efficiency Score')

        plt.tight_layout()
        return fig

    def _create_resource_feature_importance_plot(self):
        """Create resource feature importance visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Aggregate feature importance across clusters for resource models
        all_importance = []
        cluster_labels = []
        feature_indices = []

        for cluster_id, importance_data in self.feature_importance.items():
            if 'resource' in importance_data:
                importance = importance_data['resource']
                top_5_indices = np.argsort(importance)[-5:][::-1]

                for idx in top_5_indices:
                    all_importance.append(importance[idx])
                    cluster_labels.append(f'Cluster {cluster_id}')
                    feature_indices.append(f'Feature {idx}')

        if all_importance:
            # Create DataFrame for seaborn
            df = pd.DataFrame({
                'Importance': all_importance,
                'Cluster': cluster_labels,
                'Feature': feature_indices
            })

            # Create grouped bar plot
            sns.barplot(data=df, x='Feature', y='Importance', hue='Cluster', ax=ax)
            ax.set_title('Top Resource Model Feature Importance by Cluster', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Importance')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def _create_resource_usage_distribution_plot(self, predicted_resources: Dict[str, np.ndarray],
                                               cluster_assignments: np.ndarray):
        """Create resource usage distribution plot."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Resource Usage Distribution by Cluster', fontsize=16, fontweight='bold')

        resource_types = ['processing_time', 'memory_usage', 'computational_complexity']

        for i, resource_type in enumerate(resource_types):
            if resource_type in predicted_resources:
                # Create data for violin plot
                data_for_plot = []
                cluster_labels = []

                unique_clusters = np.unique(cluster_assignments)
                for cluster_id in unique_clusters:
                    if cluster_id != -1:  # Skip noise points
                        cluster_mask = cluster_assignments == cluster_id
                        resource_values = predicted_resources[resource_type][cluster_mask]
                        valid_values = resource_values[~np.isnan(resource_values)]

                        if len(valid_values) > 0:
                            data_for_plot.extend(valid_values)
                            cluster_labels.extend([f'Cluster {cluster_id}'] * len(valid_values))

                if data_for_plot:
                    df = pd.DataFrame({
                        'Resource_Usage': data_for_plot,
                        'Cluster': cluster_labels
                    })

                    sns.violinplot(data=df, x='Cluster', y='Resource_Usage', ax=axes[i], color='#0d4f01')
                    axes[i].set_title(f'{resource_type.replace("_", " ").title()}')
                    axes[i].set_xlabel('Cluster')
                    axes[i].set_ylabel('Predicted Usage')
                    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        return fig

    def _create_resource_prediction_comparison_plot(self, predicted_resources: Dict[str, np.ndarray],
                                                  actual_resources: Dict[str, np.ndarray],
                                                  cluster_assignments: np.ndarray):
        """Create predicted vs actual resource usage comparison plot."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Predicted vs Actual Resource Usage', fontsize=16, fontweight='bold')

        resource_types = ['processing_time', 'memory_usage', 'computational_complexity']

        for i, resource_type in enumerate(resource_types):
            if resource_type in predicted_resources and resource_type in actual_resources:
                predicted = predicted_resources[resource_type]
                actual = actual_resources[resource_type]

                # Filter out NaN values
                valid_mask = ~(np.isnan(predicted) | np.isnan(actual))
                if np.sum(valid_mask) > 0:
                    predicted_valid = predicted[valid_mask]
                    actual_valid = actual[valid_mask]
                    cluster_valid = cluster_assignments[valid_mask]

                    # Color by cluster
                    unique_clusters = np.unique(cluster_valid)
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

                    for j, cluster_id in enumerate(unique_clusters):
                        if cluster_id == -1:  # Skip noise points
                            continue
                        mask = cluster_valid == cluster_id
                        axes[i].scatter(actual_valid[mask], predicted_valid[mask],
                                      c=[colors[j]], label=f'Cluster {cluster_id}', alpha=0.6)

                    # Perfect prediction line
                    min_val = min(np.min(actual_valid), np.min(predicted_valid))
                    max_val = max(np.max(actual_valid), np.max(predicted_valid))
                    axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)

                    axes[i].set_xlabel('Actual Usage')
                    axes[i].set_ylabel('Predicted Usage')
                    axes[i].set_title(f'{resource_type.replace("_", " ").title()}')

                    # Add R² annotation
                    r2 = r2_score(actual_valid, predicted_valid)
                    axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    def _create_resource_trends_by_cluster_plot(self, predicted_resources: Dict[str, np.ndarray],
                                              cluster_assignments: np.ndarray):
        """Create resource usage trends by cluster plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate average resource usage per cluster
        unique_clusters = np.unique(cluster_assignments)
        cluster_data = []

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_mask = cluster_assignments == cluster_id
            cluster_resources = {}

            for resource_type in predicted_resources.keys():
                resource_values = predicted_resources[resource_type][cluster_mask]
                valid_values = resource_values[~np.isnan(resource_values)]

                if len(valid_values) > 0:
                    cluster_resources[resource_type] = np.mean(valid_values)
                else:
                    cluster_resources[resource_type] = 0

            cluster_resources['cluster_id'] = cluster_id
            cluster_data.append(cluster_resources)

        if cluster_data:
            df = pd.DataFrame(cluster_data)

            # Melt the dataframe for seaborn
            df_melted = df.melt(id_vars=['cluster_id'],
                               value_vars=['processing_time', 'memory_usage', 'computational_complexity'],
                               var_name='Resource_Type', value_name='Average_Usage')

            # Create grouped bar plot
            sns.barplot(data=df_melted, x='cluster_id', y='Average_Usage', hue='Resource_Type', ax=ax)
            ax.set_title('Average Resource Usage by Cluster', fontsize=14, fontweight='bold')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Average Predicted Usage')
            ax.legend(title='Resource Type')

        plt.tight_layout()
        return fig

    def _create_resource_optimization_plot(self, predicted_resources: Dict[str, np.ndarray],
                                         cluster_assignments: np.ndarray):
        """Create resource optimization opportunities plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Resource Optimization Analysis', fontsize=16, fontweight='bold')

        # Calculate cluster-wise statistics
        unique_clusters = np.unique(cluster_assignments)
        cluster_stats = {}

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_mask = cluster_assignments == cluster_id
            stats = {'cluster_id': cluster_id}

            for resource_type in predicted_resources.keys():
                resource_values = predicted_resources[resource_type][cluster_mask]
                valid_values = resource_values[~np.isnan(resource_values)]

                if len(valid_values) > 0:
                    stats[f'{resource_type}_mean'] = np.mean(valid_values)
                    stats[f'{resource_type}_std'] = np.std(valid_values)
                    stats[f'{resource_type}_max'] = np.max(valid_values)
                else:
                    stats[f'{resource_type}_mean'] = 0
                    stats[f'{resource_type}_std'] = 0
                    stats[f'{resource_type}_max'] = 0

            cluster_stats[cluster_id] = stats

        if cluster_stats:
            df = pd.DataFrame(list(cluster_stats.values()))

            # Resource usage variance by cluster
            variance_data = []
            for resource_type in ['processing_time', 'memory_usage', 'computational_complexity']:
                for _, row in df.iterrows():
                    variance_data.append({
                        'cluster_id': row['cluster_id'],
                        'resource_type': resource_type,
                        'variance': row[f'{resource_type}_std']
                    })

            if variance_data:
                variance_df = pd.DataFrame(variance_data)
                sns.barplot(data=variance_df, x='cluster_id', y='variance', hue='resource_type', ax=axes[0, 0])
                axes[0, 0].set_title('Resource Usage Variance by Cluster')
                axes[0, 0].set_xlabel('Cluster ID')
                axes[0, 0].set_ylabel('Standard Deviation')

            # Resource efficiency comparison
            if len(df) > 1:
                # Calculate efficiency score (lower resource usage = higher efficiency)
                df['efficiency_score'] = 1.0 / (1.0 + df['processing_time_mean'] + df['memory_usage_mean'] + df['computational_complexity_mean'])

                sns.barplot(data=df, x='cluster_id', y='efficiency_score', ax=axes[0, 1], color='#0d4f01')
                axes[0, 1].set_title('Cluster Efficiency Score')
                axes[0, 1].set_xlabel('Cluster ID')
                axes[0, 1].set_ylabel('Efficiency Score (Higher is Better)')

            # Resource usage correlation heatmap
            resource_cols = [col for col in df.columns if '_mean' in col]
            if len(resource_cols) > 1:
                corr_matrix = df[resource_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=axes[1, 0])
                axes[1, 0].set_title('Resource Usage Correlation')

            # Optimization opportunities (high usage clusters)
            if len(df) > 1:
                # Identify clusters with high resource usage
                high_usage_threshold = df[['processing_time_mean', 'memory_usage_mean', 'computational_complexity_mean']].mean().mean()
                df['high_usage'] = (df['processing_time_mean'] + df['memory_usage_mean'] + df['computational_complexity_mean']) > high_usage_threshold

                usage_summary = df.groupby('high_usage').size()

                # Create labels and colors that match the actual data
                labels = []
                colors = []
                values = []

                for high_usage, count in usage_summary.items():
                    if high_usage:
                        labels.append('High Usage')
                        colors.append('darkgray')
                    else:
                        labels.append('Low Usage')
                        colors.append('#0d4f01')
                    values.append(count)

                if values:  # Only create pie chart if we have data
                    axes[1, 1].pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
                    axes[1, 1].set_title('Clusters by Resource Usage Level')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        return fig

    def _create_resource_html_report(self, report_data: Dict[str, Any]) -> str:
        """
        Create HTML report for resource usage analysis.

        Args:
            report_data: Report data dictionary

        Returns:
            HTML report string
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Resource Usage Model Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #0d4f01; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .error { color: red; }
                .success { color: #0d4f01; }
                .recommendation { background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }
                .high-priority { border-left-color: #dc3545; }
                .medium-priority { border-left-color: #ffc107; }
                .low-priority { border-left-color: #28a745; }
            </style>
        </head>
        <body>
            <h1>Resource Usage Model Performance Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>

            <h2>Resource Model Summary</h2>
            <div class="metric">
                <p><strong>Total Resource Models Trained:</strong> {total_models}</p>
                <p><strong>Total Samples:</strong> {total_samples}</p>
                <p><strong>Average Samples per Cluster:</strong> {avg_samples:.2f}</p>
            </div>

            <h2>Performance Metrics by Cluster</h2>
            <table>
                <tr>
                    <th>Cluster ID</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R² Score</th>
                    <th>Efficiency Score</th>
                    <th>Sample Count</th>
                </tr>
                {metrics_rows}
            </table>

            <h2>Feature Importance Summary</h2>
            {feature_importance_section}

            <h2>Optimization Recommendations</h2>
            {recommendations_section}

            <h2>Visualizations</h2>
            {visualizations_section}

        </body>
        </html>
        """

        # Format metrics rows
        metrics_rows = ""
        if 'performance_metrics' in report_data:
            for cluster_id, metrics in report_data['performance_metrics'].items():
                sample_count = report_data['model_summary']['total_samples_trained'] // len(report_data['performance_metrics'])
                efficiency = metrics.get('efficiency_score', 'N/A')
                efficiency_str = f"{efficiency:.4f}" if isinstance(efficiency, (int, float)) else "N/A"

                metrics_rows += f"""
                <tr>
                    <td>{cluster_id}</td>
                    <td>{metrics['rmse']:.4f}</td>
                    <td>{metrics['mae']:.4f}</td>
                    <td>{metrics['r2_score']:.4f}</td>
                    <td>{efficiency_str}</td>
                    <td>{sample_count}</td>
                </tr>
                """

        # Format feature importance section
        feature_importance_section = ""
        if report_data['feature_importance_summary']['available']:
            feature_importance_section = "<p>Feature importance data is available for trained resource models.</p>"
        else:
            feature_importance_section = "<p>No feature importance data available.</p>"

        # Format recommendations section
        recommendations_section = ""
        if 'optimization_recommendations' in report_data and 'cluster_recommendations' in report_data['optimization_recommendations']:
            recommendations_section = "<h3>Cluster-Specific Recommendations</h3>"
            for cluster_id, cluster_data in report_data['optimization_recommendations']['cluster_recommendations'].items():
                if cluster_data['recommendations']:
                    recommendations_section += f"<h4>Cluster {cluster_id}</h4>"
                    for rec in cluster_data['recommendations']:
                        priority_class = f"{rec.get('priority', 'medium')}-priority"
                        recommendations_section += f"""
                        <div class="recommendation {priority_class}">
                            <strong>{rec.get('type', 'General').replace('_', ' ').title()}:</strong> {rec.get('message', 'No message')}
                        </div>
                        """
        else:
            recommendations_section = "<p>No optimization recommendations available.</p>"

        # Format visualizations section
        visualizations_section = ""
        if 'visualizations' in report_data and report_data['visualizations']:
            visualizations_section = "<ul>"
            for viz_name, viz_path in report_data['visualizations'].items():
                if viz_name != 'error':
                    visualizations_section += f"<li><strong>{viz_name.replace('_', ' ').title()}:</strong> {viz_path}</li>"
            visualizations_section += "</ul>"
        else:
            visualizations_section = "<p>No visualizations generated.</p>"

        return html_template.format(
            timestamp=report_data['timestamp'],
            total_models=report_data['model_summary']['total_resource_models_trained'],
            total_samples=report_data['model_summary']['total_samples_trained'],
            avg_samples=report_data['model_summary']['avg_samples_per_cluster'],
            metrics_rows=metrics_rows,
            feature_importance_section=feature_importance_section,
            recommendations_section=recommendations_section,
            visualizations_section=visualizations_section
        )

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.

        Returns:
            Dictionary containing training summary information
        """
        return {
            'timeseries_models_count': len(self.timeseries_models),
            'resource_models_count': len(self.resource_models),
            'cluster_data_sizes': self.cluster_data_sizes.copy(),
            'training_history': self.training_history.copy(),
            'total_clusters_with_data': len(self.cluster_data_sizes),
            'total_samples_processed': sum(self.cluster_data_sizes.values()),
            'feature_importance_available': list(self.feature_importance.keys())
        }