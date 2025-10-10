"""
Core data models for the MLflow Time Series Clustering Pipeline.

This module defines the primary data structures used throughout the pipeline
for representing clustering results, batch processing outcomes, and performance reports.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class ClusteringResult:
    """
    Result of HDBSCAN clustering operation.

    Attributes:
        labels: Cluster labels for each data point (-1 for noise points)
        cluster_centers: Computed cluster centers
        noise_ratio: Ratio of noise points to total points
        metrics: Dictionary containing clustering quality metrics
    """
    labels: np.ndarray
    cluster_centers: np.ndarray
    noise_ratio: float
    metrics: Dict[str, Any]

    def __post_init__(self):
        """Validate clustering result data."""
        if not isinstance(self.labels, np.ndarray):
            raise ValueError("labels must be a numpy array")
        if not isinstance(self.cluster_centers, np.ndarray):
            raise ValueError("cluster_centers must be a numpy array")
        if not 0.0 <= self.noise_ratio <= 1.0:
            raise ValueError("noise_ratio must be between 0.0 and 1.0")
        if not isinstance(self.metrics, dict):
            raise ValueError("metrics must be a dictionary")


@dataclass
class BatchResult:
    """
    Result of processing a single data batch.

    Attributes:
        cluster_assignments: Cluster assignments for batch data points
        timeseries_predictions: Time series predictions for each data point
        resource_predictions: Resource usage predictions per cluster
        noise_ratio: Ratio of noise points in this batch
        processing_time: Time taken to process this batch in seconds
    """
    cluster_assignments: np.ndarray
    timeseries_predictions: np.ndarray
    resource_predictions: Dict[int, Dict[str, float]]
    noise_ratio: float
    processing_time: float

    def __post_init__(self):
        """Validate batch result data."""
        if not isinstance(self.cluster_assignments, np.ndarray):
            raise ValueError("cluster_assignments must be a numpy array")
        if not isinstance(self.timeseries_predictions, np.ndarray):
            raise ValueError("timeseries_predictions must be a numpy array")
        if not isinstance(self.resource_predictions, dict):
            raise ValueError("resource_predictions must be a dictionary")
        if not 0.0 <= self.noise_ratio <= 1.0:
            raise ValueError("noise_ratio must be between 0.0 and 1.0")
        if self.processing_time < 0:
            raise ValueError("processing_time must be non-negative")


@dataclass
class PerformanceReport:
    """
    Comprehensive performance report for the pipeline.

    Attributes:
        clustering_metrics: Metrics related to clustering performance
        model_metrics: Metrics for time series and resource usage models
        visualizations: Dictionary of visualization file paths or data
        timestamp: When this report was generated
        model_version: Version identifier for the models used
    """
    clustering_metrics: Dict[str, Any]
    model_metrics: Dict[str, Any]
    visualizations: Dict[str, Any]
    timestamp: datetime
    model_version: str

    def __post_init__(self):
        """Validate performance report data."""
        if not isinstance(self.clustering_metrics, dict):
            raise ValueError("clustering_metrics must be a dictionary")
        if not isinstance(self.model_metrics, dict):
            raise ValueError("model_metrics must be a dictionary")
        if not isinstance(self.visualizations, dict):
            raise ValueError("visualizations must be a dictionary")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        if not isinstance(self.model_version, str) or not self.model_version:
            raise ValueError("model_version must be a non-empty string")