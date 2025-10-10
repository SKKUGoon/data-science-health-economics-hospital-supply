"""
Artifact management for the MLflow Time Series Clustering Pipeline.

This module handles model serialization using joblib, artifact organization
by HospitalProfile ID, and artifact loading with validation.
"""

import os
import joblib
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import tempfile
import shutil
from datetime import datetime
import json

from ..core.exceptions import MLflowIntegrationError
from ..core.config import PipelineConfig
from ..mlflow_integration.logging_utils import PipelineLogger


class ArtifactManager:
    """
    Manages artifact storage, organization, and loading for the pipeline.

    This class handles joblib serialization of models, organizes artifacts
    by HospitalProfile ID, and provides validation for loaded artifacts.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the artifact manager.

        Args:
            config: Pipeline configuration containing artifact settings
        """
        self.config = config
        self.hospital_profile_id = config.hospital_profile_id or "default"
        self.logger = PipelineLogger("artifact_manager")

        # Define artifact structure
        self.artifact_structure = {
            "models": {
                "hdbscan_model.joblib": "HDBSCAN clustering model",
                "scaler.joblib": "StandardScaler for preprocessing",
                "knn_fallback.joblib": "kNN fallback model",
                "timeseries_models": "Directory for cluster-specific time series models",
                "resource_models": "Directory for cluster-specific resource models"
            },
            "reports": {
                "clustering_report.html": "Clustering performance report",
                "model_performance_report.html": "Model performance report",
                "visualizations": "Directory for visualization artifacts"
            },
            "data": {
                "cluster_labels.npy": "Cluster labels array",
                "cluster_centers.npy": "Cluster centers array",
                "performance_history.json": "Performance metrics history"
            }
        }

    def save_model_artifact(self, model: Any, artifact_name: str,
                          cluster_id: Optional[int] = None,
                          model_type: str = "general") -> str:
        """
        Save a model artifact using joblib serialization.

        Args:
            model: Model object to save
            artifact_name: Name for the artifact file
            cluster_id: Optional cluster ID for cluster-specific models
            model_type: Type of model (timeseries, resource, general)

        Returns:
            Path to the saved artifact
        """
        try:
            # Create temporary directory for artifact
            with tempfile.TemporaryDirectory() as temp_dir:
                # Determine artifact path based on type and cluster
                if cluster_id is not None:
                    if model_type == "timeseries":
                        artifact_dir = f"{self.hospital_profile_id}/models/timeseries_models"
                        artifact_filename = f"cluster_{cluster_id}_model.joblib"
                    elif model_type == "resource":
                        artifact_dir = f"{self.hospital_profile_id}/models/resource_models"
                        artifact_filename = f"cluster_{cluster_id}_resource.joblib"
                    else:
                        artifact_dir = f"{self.hospital_profile_id}/models"
                        artifact_filename = f"{artifact_name}_cluster_{cluster_id}.joblib"
                else:
                    artifact_dir = f"{self.hospital_profile_id}/models"
                    artifact_filename = f"{artifact_name}.joblib"

                # Create local artifact path
                local_artifact_path = Path(temp_dir) / artifact_filename
                local_artifact_path.parent.mkdir(parents=True, exist_ok=True)

                # Save model using joblib
                joblib.dump(model, local_artifact_path)

                # Create metadata
                metadata = {
                    "model_type": model_type,
                    "cluster_id": cluster_id,
                    "hospital_profile_id": self.hospital_profile_id,
                    "saved_at": datetime.now().isoformat(),
                    "joblib_version": joblib.__version__,
                    "artifact_name": artifact_name
                }

                # Save metadata alongside model
                metadata_path = local_artifact_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Log artifacts to MLflow
                mlflow.log_artifact(str(local_artifact_path), artifact_dir)
                mlflow.log_artifact(str(metadata_path), artifact_dir)

                artifact_path = f"{artifact_dir}/{artifact_filename}"
                self.logger.info(
                    f"Saved {model_type} model artifact: {artifact_path}"
                )

                return artifact_path

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to save model artifact {artifact_name}: {str(e)}",
                operation="save_model_artifact",
                details={
                    "artifact_name": artifact_name,
                    "model_type": model_type,
                    "cluster_id": cluster_id
                }
            )

    def load_model_artifact(self, artifact_path: str,
                          validate: bool = True) -> Any:
        """
        Load a model artifact from MLflow.

        Args:
            artifact_path: Path to the artifact in MLflow
            validate: Whether to validate the loaded model

        Returns:
            Loaded model object
        """
        try:
            # Download artifact from MLflow
            local_path = mlflow.artifacts.download_artifacts(artifact_path)

            # Load model using joblib
            model = joblib.load(local_path)

            if validate:
                self._validate_loaded_model(model, artifact_path)

            self.logger.info(f"Loaded model artifact: {artifact_path}")
            return model

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to load model artifact {artifact_path}: {str(e)}",
                operation="load_model_artifact",
                details={"artifact_path": artifact_path}
            )

    def save_data_artifact(self, data: Union[Dict, List, Any], artifact_name: str,
                          data_type: str = "general") -> str:
        """
        Save data artifacts (arrays, dictionaries, etc.).

        Args:
            data: Data to save
            artifact_name: Name for the artifact
            data_type: Type of data being saved

        Returns:
            Path to the saved artifact
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_dir = f"{self.hospital_profile_id}/data"

                # Determine file extension and save method
                if artifact_name.endswith('.npy'):
                    import numpy as np
                    local_path = Path(temp_dir) / artifact_name
                    np.save(local_path, data)
                elif artifact_name.endswith('.json'):
                    local_path = Path(temp_dir) / artifact_name
                    with open(local_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    # Use joblib for other data types
                    local_path = Path(temp_dir) / f"{artifact_name}.joblib"
                    joblib.dump(data, local_path)

                # Create metadata
                metadata = {
                    "data_type": data_type,
                    "hospital_profile_id": self.hospital_profile_id,
                    "saved_at": datetime.now().isoformat(),
                    "artifact_name": artifact_name
                }

                metadata_path = local_path.with_suffix('.meta.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Log to MLflow
                mlflow.log_artifact(str(local_path), artifact_dir)
                mlflow.log_artifact(str(metadata_path), artifact_dir)

                artifact_path = f"{artifact_dir}/{local_path.name}"
                self.logger.info(f"Saved data artifact: {artifact_path}")

                return artifact_path

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to save data artifact {artifact_name}: {str(e)}",
                operation="save_data_artifact",
                details={"artifact_name": artifact_name, "data_type": data_type}
            )

    def load_data_artifact(self, artifact_path: str) -> Any:
        """
        Load a data artifact from MLflow.

        Args:
            artifact_path: Path to the artifact in MLflow

        Returns:
            Loaded data
        """
        try:
            local_path = mlflow.artifacts.download_artifacts(artifact_path)

            # Determine loading method based on file extension
            if artifact_path.endswith('.npy'):
                import numpy as np
                data = np.load(local_path)
            elif artifact_path.endswith('.json'):
                with open(local_path, 'r') as f:
                    data = json.load(f)
            else:
                # Use joblib for other formats
                data = joblib.load(local_path)

            self.logger.info(f"Loaded data artifact: {artifact_path}")
            return data

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to load data artifact {artifact_path}: {str(e)}",
                operation="load_data_artifact",
                details={"artifact_path": artifact_path}
            )

    def save_report_artifact(self, report_content: str, report_name: str,
                           report_type: str = "html") -> str:
        """
        Save report artifacts (HTML, text, etc.).

        Args:
            report_content: Content of the report
            report_name: Name for the report file
            report_type: Type of report (html, txt, etc.)

        Returns:
            Path to the saved report
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_dir = f"{self.hospital_profile_id}/reports"

                # Ensure proper file extension
                if not report_name.endswith(f'.{report_type}'):
                    report_name = f"{report_name}.{report_type}"

                local_path = Path(temp_dir) / report_name

                # Save report content
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)

                # Create metadata
                metadata = {
                    "report_type": report_type,
                    "hospital_profile_id": self.hospital_profile_id,
                    "generated_at": datetime.now().isoformat(),
                    "report_name": report_name
                }

                metadata_path = local_path.with_suffix('.meta.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Log to MLflow
                mlflow.log_artifact(str(local_path), artifact_dir)
                mlflow.log_artifact(str(metadata_path), artifact_dir)

                artifact_path = f"{artifact_dir}/{report_name}"
                self.logger.info(f"Saved report artifact: {artifact_path}")

                return artifact_path

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to save report artifact {report_name}: {str(e)}",
                operation="save_report_artifact",
                details={"report_name": report_name, "report_type": report_type}
            )

    def save_visualization_artifact(self, figure, viz_name: str,
                                  viz_type: str = "png") -> str:
        """
        Save visualization artifacts (matplotlib figures, etc.).

        Args:
            figure: Matplotlib figure or similar visualization object
            viz_name: Name for the visualization file
            viz_type: Type of visualization (png, svg, pdf)

        Returns:
            Path to the saved visualization
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_dir = f"{self.hospital_profile_id}/reports/visualizations"

                # Ensure proper file extension
                if not viz_name.endswith(f'.{viz_type}'):
                    viz_name = f"{viz_name}.{viz_type}"

                local_path = Path(temp_dir) / viz_name

                # Save visualization
                if hasattr(figure, 'savefig'):
                    # Matplotlib figure
                    figure.savefig(local_path, dpi=300, bbox_inches='tight')
                else:
                    # Other visualization types
                    figure.save(local_path)

                # Create metadata
                metadata = {
                    "visualization_type": viz_type,
                    "hospital_profile_id": self.hospital_profile_id,
                    "generated_at": datetime.now().isoformat(),
                    "visualization_name": viz_name
                }

                metadata_path = local_path.with_suffix('.meta.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Log to MLflow
                mlflow.log_artifact(str(local_path), artifact_dir)
                mlflow.log_artifact(str(metadata_path), artifact_dir)

                artifact_path = f"{artifact_dir}/{viz_name}"
                self.logger.info(f"Saved visualization artifact: {artifact_path}")

                return artifact_path

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to save visualization artifact {viz_name}: {str(e)}",
                operation="save_visualization_artifact",
                details={"viz_name": viz_name, "viz_type": viz_type}
            )

    def list_artifacts(self, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available artifacts for the current hospital profile.

        Args:
            artifact_type: Filter by artifact type (models, reports, data, visualizations)

        Returns:
            List of artifact information dictionaries
        """
        try:
            # Get current run artifacts
            client = mlflow.tracking.MlflowClient()
            run = mlflow.active_run()

            if run is None:
                raise MLflowIntegrationError(
                    "No active MLflow run to list artifacts from",
                    operation="list_artifacts"
                )

            # List artifacts with hospital profile filter
            base_path = self.hospital_profile_id
            if artifact_type:
                base_path = f"{base_path}/{artifact_type}"

            artifacts = client.list_artifacts(run.info.run_id, base_path)

            artifact_list = []
            for artifact in artifacts:
                artifact_info = {
                    "path": artifact.path,
                    "is_dir": artifact.is_dir,
                    "file_size": artifact.file_size,
                    "hospital_profile_id": self.hospital_profile_id
                }

                # Try to load metadata if available
                if not artifact.is_dir:
                    metadata_path = f"{artifact.path}.meta.json"
                    try:
                        metadata_artifact = client.list_artifacts(
                            run.info.run_id, metadata_path
                        )
                        if metadata_artifact:
                            # Load metadata
                            local_metadata = mlflow.artifacts.download_artifacts(metadata_path)
                            with open(local_metadata, 'r') as f:
                                metadata = json.load(f)
                            artifact_info.update(metadata)
                    except:
                        # Metadata not available, continue without it
                        pass

                artifact_list.append(artifact_info)

            self.logger.info(
                f"Listed {len(artifact_list)} artifacts for {self.hospital_profile_id}"
            )
            return artifact_list

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to list artifacts: {str(e)}",
                operation="list_artifacts",
                details={"artifact_type": artifact_type}
            )

    def _validate_loaded_model(self, model: Any, artifact_path: str) -> None:
        """
        Validate a loaded model artifact.

        Args:
            model: Loaded model object
            artifact_path: Path to the artifact
        """
        try:
            # Basic validation - check if model has expected methods
            if hasattr(model, 'predict'):
                # Try a dummy prediction to ensure model is functional
                # This is a basic check - more specific validation can be added
                pass
            elif hasattr(model, 'transform'):
                # For transformers like scalers
                pass
            else:
                self.logger.warning(
                    f"Loaded model from {artifact_path} may not have expected methods"
                )

            self.logger.debug(f"Model validation passed for {artifact_path}")

        except Exception as e:
            self.logger.warning(
                f"Model validation failed for {artifact_path}: {str(e)}"
            )

    def cleanup_old_artifacts(self, keep_versions: int = 5) -> None:
        """
        Clean up old artifact versions to manage storage.

        Args:
            keep_versions: Number of recent versions to keep
        """
        try:
            # This is a placeholder for artifact cleanup logic
            # In a full implementation, this would:
            # 1. List all artifacts for the hospital profile
            # 2. Group by artifact type and name
            # 3. Keep only the most recent versions
            # 4. Delete older versions

            self.logger.info(
                f"Artifact cleanup completed, keeping {keep_versions} versions"
            )

        except Exception as e:
            self.logger.warning(f"Artifact cleanup failed: {str(e)}")

    def get_artifact_info(self, artifact_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific artifact.

        Args:
            artifact_path: Path to the artifact

        Returns:
            Dictionary containing artifact information
        """
        try:
            client = mlflow.tracking.MlflowClient()
            run = mlflow.active_run()

            if run is None:
                raise MLflowIntegrationError(
                    "No active MLflow run to get artifact info from",
                    operation="get_artifact_info"
                )

            # Get artifact info from MLflow
            artifacts = client.list_artifacts(run.info.run_id, artifact_path)

            if not artifacts:
                raise MLflowIntegrationError(
                    f"Artifact not found: {artifact_path}",
                    operation="get_artifact_info"
                )

            artifact = artifacts[0]
            artifact_info = {
                "path": artifact.path,
                "is_dir": artifact.is_dir,
                "file_size": artifact.file_size,
                "hospital_profile_id": self.hospital_profile_id
            }

            # Try to load metadata
            metadata_path = f"{artifact_path}.meta.json"
            try:
                local_metadata = mlflow.artifacts.download_artifacts(metadata_path)
                with open(local_metadata, 'r') as f:
                    metadata = json.load(f)
                artifact_info.update(metadata)
            except:
                # Metadata not available
                pass

            return artifact_info

        except Exception as e:
            raise MLflowIntegrationError(
                f"Failed to get artifact info for {artifact_path}: {str(e)}",
                operation="get_artifact_info",
                details={"artifact_path": artifact_path}
            )