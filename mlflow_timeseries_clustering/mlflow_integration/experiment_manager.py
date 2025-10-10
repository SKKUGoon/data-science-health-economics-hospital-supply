"""
MLflow experiment management for the Time Series Clustering Pipeline.

This module handles MLflow experiment creation, run management, and nested
run hierarchy for the pipeline components.
"""

import mlflow
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import logging

from ..core.exceptions import MLflowIntegrationError
from ..core.config import PipelineConfig
from .artifact_manager import ArtifactManager


class ExperimentManager:
    """
    Manages MLflow experiments and runs for the Time Series Clustering Pipeline.

    This class handles experiment creation, nested run management, and provides
    context managers for different pipeline phases.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the experiment manager.

        Args:
            config: Pipeline configuration containing MLflow settings
        """
        self.config = config
        self.experiment_name = config.experiment_name
        self.hospital_profile_id = config.hospital_profile_id
        self.logger = logging.getLogger(__name__)

        # Track active runs
        self._active_runs: Dict[str, str] = {}
        self._parent_run_id: Optional[str] = None

        # Initialize artifact manager
        self.artifact_manager = ArtifactManager(config)

        # Initialize experiment
        self._setup_experiment()

    def _setup_experiment(self):
        """Set up the MLflow experiment."""
        try:
            # Create or get existing experiment
            experiment_name = self.experiment_name
            if self.hospital_profile_id:
                experiment_name = f"{experiment_name}-{self.hospital_profile_id}"

            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    tags={
                        "pipeline_type": "timeseries_clustering",
                        "hospital_profile_id": self.hospital_profile_id or "default",
                        "version": "1.0.0"
                    }
                )
                self.logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                self.logger.info(f"Using existing MLflow experiment: {experiment_name}")

            # Set the experiment
            mlflow.set_experiment(experiment_name)
            self.experiment_id = experiment_id

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to setup MLflow experiment: {str(e)}",
                operation="setup_experiment",
                experiment_name=experiment_name
            )

    @contextmanager
    def start_parent_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Start a parent run for the entire pipeline execution.

        Args:
            run_name: Name for the parent run
            tags: Optional tags for the run

        Yields:
            MLflow run object
        """
        try:
            default_tags = {
                "pipeline_phase": "parent",
                "hospital_profile_id": self.hospital_profile_id or "default"
            }
            if tags:
                default_tags.update(tags)

            with mlflow.start_run(run_name=run_name, tags=default_tags) as run:
                self._parent_run_id = run.info.run_id
                self.logger.info(f"Started parent run: {run_name} (ID: {self._parent_run_id})")

                # Log pipeline configuration
                self._log_pipeline_config()

                yield run

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to start parent run: {str(e)}",
                operation="start_parent_run",
                run_id=self._parent_run_id
            )
        finally:
            self._parent_run_id = None

    @contextmanager
    def start_nested_run(self, run_name: str, component: str,
                        tags: Optional[Dict[str, str]] = None):
        """
        Start a nested run for a specific pipeline component.

        Args:
            run_name: Name for the nested run
            component: Component name (e.g., 'hdbscan', 'lightgbm_timeseries')
            tags: Optional tags for the run

        Yields:
            MLflow run object
        """
        if self._parent_run_id is None:
            raise MLflowIntegrationError(
                "Cannot start nested run without active parent run",
                operation="start_nested_run"
            )

        try:
            default_tags = {
                "pipeline_phase": "nested",
                "component": component,
                "hospital_profile_id": self.hospital_profile_id or "default"
            }
            if tags:
                default_tags.update(tags)

            with mlflow.start_run(
                run_name=run_name,
                nested=True,
                tags=default_tags
            ) as run:
                nested_run_id = run.info.run_id
                self._active_runs[component] = nested_run_id
                self.logger.info(
                    f"Started nested run: {run_name} for component {component} "
                    f"(ID: {nested_run_id})"
                )

                yield run

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to start nested run for component {component}: {str(e)}",
                operation="start_nested_run"
            )
        finally:
            if component in self._active_runs:
                del self._active_runs[component]

    def _log_pipeline_config(self):
        """Log pipeline configuration parameters to MLflow."""
        try:
            config_dict = self.config.to_dict()

            # Log parameters in organized groups
            for param_group, params in config_dict.items():
                if isinstance(params, dict):
                    for param_name, param_value in params.items():
                        mlflow.log_param(f"{param_group}.{param_name}", param_value)
                else:
                    mlflow.log_param(param_group, params)

            self.logger.info("Logged pipeline configuration to MLflow")

        except Exception as e:
            self.logger.warning(f"Failed to log pipeline configuration: {str(e)}")

    def get_active_run_id(self, component: Optional[str] = None) -> Optional[str]:
        """
        Get the active run ID for a component or parent run.

        Args:
            component: Component name, or None for parent run

        Returns:
            Run ID if active, None otherwise
        """
        if component is None:
            return self._parent_run_id
        return self._active_runs.get(component)

    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get information about the current experiment.

        Returns:
            Dictionary containing experiment information
        """
        try:
            experiment = mlflow.get_experiment(self.experiment_id)
            return {
                "experiment_id": experiment.experiment_id,
                "experiment_name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
                "tags": experiment.tags
            }
        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to get experiment info: {str(e)}",
                operation="get_experiment_info",
                experiment_name=self.experiment_name
            )

    def list_runs(self, component: Optional[str] = None,
                 max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List runs for the experiment, optionally filtered by component.

        Args:
            component: Filter by component name
            max_results: Maximum number of runs to return

        Returns:
            List of run information dictionaries
        """
        try:
            filter_string = ""
            if component:
                filter_string = f"tags.component = '{component}'"

            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=["start_time DESC"]
            )

            return runs.to_dict('records') if not runs.empty else []

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to list runs: {str(e)}",
                operation="list_runs",
                experiment_name=self.experiment_name
            )

    def register_pipeline_model(self, model_components: Dict[str, str],
                              model_version: str = "1.0.0",
                              stage: str = "None") -> str:
        """
        Register all pipeline components as a unified model in MLflow Model Registry.

        Args:
            model_components: Dictionary mapping component names to artifact paths
            model_version: Version string for the model
            stage: Model stage (None, Staging, Production, Archived)

        Returns:
            Registered model name
        """
        try:
            model_name = f"{self.config.model_registry_prefix}-{self.hospital_profile_id}"

            # Create model registry entry if it doesn't exist
            client = mlflow.tracking.MlflowClient()
            try:
                client.get_registered_model(model_name)
                self.logger.info(f"Using existing registered model: {model_name}")
            except mlflow.exceptions.RestException:
                client.create_registered_model(
                    name=model_name,
                    tags={
                        "pipeline_type": "timeseries_clustering",
                        "hospital_profile_id": self.hospital_profile_id,
                        "components": ",".join(model_components.keys())
                    },
                    description=f"Time Series Clustering Pipeline for {self.hospital_profile_id}"
                )
                self.logger.info(f"Created new registered model: {model_name}")

            # Register current run as a model version
            if self._parent_run_id:
                model_version_info = client.create_model_version(
                    name=model_name,
                    source=f"runs:/{self._parent_run_id}",
                    run_id=self._parent_run_id,
                    tags={
                        "version": model_version,
                        "hospital_profile_id": self.hospital_profile_id,
                        "components": ",".join(model_components.keys())
                    },
                    description=f"Pipeline version {model_version} with components: {', '.join(model_components.keys())}"
                )

                # Set model stage if specified
                if stage != "None":
                    client.transition_model_version_stage(
                        name=model_name,
                        version=model_version_info.version,
                        stage=stage
                    )

                self.logger.info(
                    f"Registered model version {model_version_info.version} "
                    f"for {model_name} in stage {stage}"
                )

                return model_name

            else:
                raise MLflowIntegrationError(
                    "Cannot register model without active parent run",
                    operation="register_pipeline_model"
                )

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to register pipeline model: {str(e)}",
                operation="register_pipeline_model",
                details={"model_components": list(model_components.keys())}
            )

    def get_latest_model_version(self, stage: str = "Production") -> Optional[Dict[str, Any]]:
        """
        Get the latest model version from the registry.

        Args:
            stage: Model stage to filter by

        Returns:
            Dictionary containing model version information
        """
        try:
            model_name = f"{self.config.model_registry_prefix}-{self.hospital_profile_id}"
            client = mlflow.tracking.MlflowClient()

            try:
                model_versions = client.get_latest_versions(
                    name=model_name,
                    stages=[stage] if stage != "None" else None
                )

                if model_versions:
                    latest_version = model_versions[0]
                    return {
                        "name": latest_version.name,
                        "version": latest_version.version,
                        "stage": latest_version.current_stage,
                        "run_id": latest_version.run_id,
                        "source": latest_version.source,
                        "tags": latest_version.tags,
                        "description": latest_version.description
                    }
                else:
                    self.logger.info(f"No model versions found for {model_name} in stage {stage}")
                    return None

            except mlflow.exceptions.RestException:
                self.logger.info(f"Registered model {model_name} not found")
                return None

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to get latest model version: {str(e)}",
                operation="get_latest_model_version",
                details={"stage": stage}
            )

    def compare_model_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions and their performance metrics.

        Args:
            version1: First model version to compare
            version2: Second model version to compare

        Returns:
            Dictionary containing comparison results
        """
        try:
            model_name = f"{self.config.model_registry_prefix}-{self.hospital_profile_id}"
            client = mlflow.tracking.MlflowClient()

            # Get model version details
            mv1 = client.get_model_version(model_name, version1)
            mv2 = client.get_model_version(model_name, version2)

            # Get run metrics for comparison
            run1_metrics = client.get_run(mv1.run_id).data.metrics
            run2_metrics = client.get_run(mv2.run_id).data.metrics

            # Calculate metric differences
            metric_comparison = {}
            all_metrics = set(run1_metrics.keys()) | set(run2_metrics.keys())

            for metric in all_metrics:
                val1 = run1_metrics.get(metric, 0)
                val2 = run2_metrics.get(metric, 0)
                metric_comparison[metric] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": val2 - val1,
                    "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }

            comparison_result = {
                "model_name": model_name,
                "version1": {
                    "version": version1,
                    "run_id": mv1.run_id,
                    "stage": mv1.current_stage,
                    "creation_timestamp": mv1.creation_timestamp
                },
                "version2": {
                    "version": version2,
                    "run_id": mv2.run_id,
                    "stage": mv2.current_stage,
                    "creation_timestamp": mv2.creation_timestamp
                },
                "metric_comparison": metric_comparison
            }

            self.logger.info(f"Compared model versions {version1} and {version2}")
            return comparison_result

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to compare model versions: {str(e)}",
                operation="compare_model_versions",
                details={"version1": version1, "version2": version2}
            )

    def transition_model_stage(self, version: str, new_stage: str,
                             archive_existing: bool = True) -> None:
        """
        Transition a model version to a new stage.

        Args:
            version: Model version to transition
            new_stage: New stage (Staging, Production, Archived)
            archive_existing: Whether to archive existing models in the target stage
        """
        try:
            model_name = f"{self.config.model_registry_prefix}-{self.hospital_profile_id}"
            client = mlflow.tracking.MlflowClient()

            # Archive existing models in the target stage if requested
            if archive_existing and new_stage in ["Staging", "Production"]:
                existing_versions = client.get_latest_versions(
                    name=model_name,
                    stages=[new_stage]
                )

                for existing_version in existing_versions:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=existing_version.version,
                        stage="Archived"
                    )
                    self.logger.info(
                        f"Archived existing model version {existing_version.version}"
                    )

            # Transition the specified version to the new stage
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=new_stage
            )

            self.logger.info(
                f"Transitioned model version {version} to stage {new_stage}"
            )

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to transition model stage: {str(e)}",
                operation="transition_model_stage",
                details={"version": version, "new_stage": new_stage}
            )

    def rollback_model(self, target_version: str) -> None:
        """
        Rollback to a previous model version by promoting it to Production.

        Args:
            target_version: Version to rollback to
        """
        try:
            model_name = f"{self.config.model_registry_prefix}-{self.hospital_profile_id}"
            client = mlflow.tracking.MlflowClient()

            # Get current production model
            current_production = client.get_latest_versions(
                name=model_name,
                stages=["Production"]
            )

            # Archive current production model
            if current_production:
                for prod_version in current_production:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=prod_version.version,
                        stage="Archived"
                    )
                    self.logger.info(
                        f"Archived current production version {prod_version.version}"
                    )

            # Promote target version to production
            client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production"
            )

            self.logger.info(f"Rolled back to model version {target_version}")

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to rollback model: {str(e)}",
                operation="rollback_model",
                details={"target_version": target_version}
            )

    def list_model_versions(self) -> List[Dict[str, Any]]:
        """
        List all versions of the pipeline model.

        Returns:
            List of model version information dictionaries
        """
        try:
            model_name = f"{self.config.model_registry_prefix}-{self.hospital_profile_id}"
            client = mlflow.tracking.MlflowClient()

            try:
                model_versions = client.search_model_versions(f"name='{model_name}'")

                version_list = []
                for version in model_versions:
                    version_info = {
                        "name": version.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "run_id": version.run_id,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                        "tags": version.tags,
                        "description": version.description
                    }
                    version_list.append(version_info)

                self.logger.info(f"Listed {len(version_list)} model versions for {model_name}")
                return version_list

            except mlflow.exceptions.RestException:
                self.logger.info(f"Registered model {model_name} not found")
                return []

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to list model versions: {str(e)}",
                operation="list_model_versions"
            )

    def log_pipeline_metrics(self, metrics: Dict[str, float],
                           step: Optional[int] = None) -> None:
        """
        Log metrics for the current pipeline run.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series metrics
        """
        try:
            for metric_name, metric_value in metrics.items():
                if step is not None:
                    mlflow.log_metric(metric_name, metric_value, step=step)
                else:
                    mlflow.log_metric(metric_name, metric_value)

            self.logger.info(f"Logged {len(metrics)} pipeline metrics")

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to log pipeline metrics: {str(e)}",
                operation="log_pipeline_metrics"
            )

    def log_pipeline_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log parameters for the current pipeline run.

        Args:
            parameters: Dictionary of parameters to log
        """
        try:
            for param_name, param_value in parameters.items():
                mlflow.log_param(param_name, param_value)

            self.logger.info(f"Logged {len(parameters)} pipeline parameters")

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to log pipeline parameters: {str(e)}",
                operation="log_pipeline_parameters"
            )

    def set_pipeline_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for the current pipeline run.

        Args:
            tags: Dictionary of tags to set
        """
        try:
            mlflow.set_tags(tags)
            self.logger.info(f"Set {len(tags)} pipeline tags")

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to set pipeline tags: {str(e)}",
                operation="set_pipeline_tags"
            )

    def create_child_experiment(self, child_name: str,
                              tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create a child experiment for specific pipeline phases.

        Args:
            child_name: Name for the child experiment
            tags: Optional tags for the child experiment

        Returns:
            Child experiment ID
        """
        try:
            child_experiment_name = f"{self.experiment_name}-{child_name}"
            if self.hospital_profile_id:
                child_experiment_name = f"{child_experiment_name}-{self.hospital_profile_id}"

            default_tags = {
                "parent_experiment": self.experiment_name,
                "pipeline_phase": child_name,
                "hospital_profile_id": self.hospital_profile_id or "default"
            }
            if tags:
                default_tags.update(tags)

            child_experiment_id = mlflow.create_experiment(
                name=child_experiment_name,
                tags=default_tags
            )

            self.logger.info(f"Created child experiment: {child_experiment_name}")
            return child_experiment_id

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to create child experiment: {str(e)}",
                operation="create_child_experiment",
                details={"child_name": child_name}
            )

    def get_run_metrics_history(self, run_id: str,
                              metric_name: str) -> List[Dict[str, Any]]:
        """
        Get the history of a specific metric for a run.

        Args:
            run_id: MLflow run ID
            metric_name: Name of the metric

        Returns:
            List of metric history entries
        """
        try:
            client = mlflow.tracking.MlflowClient()
            metric_history = client.get_metric_history(run_id, metric_name)

            history_list = []
            for metric in metric_history:
                history_entry = {
                    "timestamp": metric.timestamp,
                    "step": metric.step,
                    "value": metric.value
                }
                history_list.append(history_entry)

            return history_list

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to get metric history: {str(e)}",
                operation="get_run_metrics_history",
                details={"run_id": run_id, "metric_name": metric_name}
            )

    def search_runs_by_metrics(self, metric_filters: Dict[str, str],
                             max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search runs based on metric criteria.

        Args:
            metric_filters: Dictionary of metric filters (e.g., {"rmse": "< 0.1"})
            max_results: Maximum number of runs to return

        Returns:
            List of matching run information
        """
        try:
            # Build filter string
            filter_conditions = []
            for metric, condition in metric_filters.items():
                filter_conditions.append(f"metrics.{metric} {condition}")

            filter_string = " and ".join(filter_conditions)

            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=["start_time DESC"]
            )

            return runs.to_dict('records') if not runs.empty else []

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to search runs by metrics: {str(e)}",
                operation="search_runs_by_metrics",
                details={"metric_filters": metric_filters}
            )

    def get_best_run(self, metric_name: str,
                    mode: str = "min") -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a specific metric.

        Args:
            metric_name: Name of the metric to optimize
            mode: Optimization mode ("min" or "max")

        Returns:
            Best run information or None if no runs found
        """
        try:
            order_direction = "ASC" if mode == "min" else "DESC"
            order_by = [f"metrics.{metric_name} {order_direction}"]

            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=order_by,
                max_results=1
            )

            if not runs.empty:
                best_run = runs.iloc[0].to_dict()
                self.logger.info(
                    f"Found best run for {metric_name} ({mode}): {best_run['run_id']}"
                )
                return best_run
            else:
                self.logger.info(f"No runs found for metric {metric_name}")
                return None

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to get best run: {str(e)}",
                operation="get_best_run",
                details={"metric_name": metric_name, "mode": mode}
            )

    def archive_experiment(self) -> None:
        """Archive the current experiment."""
        try:
            client = mlflow.tracking.MlflowClient()
            client.delete_experiment(self.experiment_id)
            self.logger.info(f"Archived experiment: {self.experiment_name}")

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to archive experiment: {str(e)}",
                operation="archive_experiment"
            )

    def restore_experiment(self) -> None:
        """Restore an archived experiment."""
        try:
            client = mlflow.tracking.MlflowClient()
            client.restore_experiment(self.experiment_id)
            self.logger.info(f"Restored experiment: {self.experiment_name}")

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to restore experiment: {str(e)}",
                operation="restore_experiment"
            )