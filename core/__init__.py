"""
Core module providing shared infrastructure for the MLflow Time Series Clustering Pipeline.

This module serves as the foundation for the entire project, providing:

**Configuration Management:**
- `PipelineConfig`: Unified Pydantic-based configuration with validation
- Centralized parameter management for HDBSCAN, LightGBM, and kNN models
- MLflow integration settings and hospital profile management

**Data Models:**
- Pydantic BaseModels for all data structures with automatic validation
- Comprehensive models for clustering results, batch processing, and performance monitoring
- JSON serialization/deserialization with proper datetime handling
- Type safety and runtime validation for all pipeline data

**Exception Hierarchy:**
- Unified exception classes for consistent error handling
- Specific exceptions for different pipeline components
- Enhanced error context and debugging information

**Key Features:**
- Automatic validation of all configuration parameters
- Type-safe data models with comprehensive field validation
- Consistent error handling across all modules
- MLflow integration for experiment tracking
- Hospital profile integration for multi-tenant support

**Usage Example:**
```python
from core import PipelineConfig, ClusteringResult, ValidationError

# Create configuration with validation
config = PipelineConfig(
    noise_threshold=0.25,
    batch_size=500,
    hospital_profile_id="hospital_123"
)

# Use validated data models
result = ClusteringResult(
    labels=[0, 1, 0, -1],
    cluster_centers=[[1.0, 2.0], [3.0, 4.0]],
    noise_ratio=0.25,
    metrics={"silhouette_score": 0.75}
)
```

**Architecture Integration:**
This core module is imported and used by all other modules in the project:
- `utils/` - Uses core exceptions and configuration
- `models/` - Uses core data models and configuration
- `mlflow_integration/` - Uses core configuration and exceptions
- `monitoring/` - Uses core data models for alerts and events
- `pipeline/` - Uses all core components for orchestration
"""

from .config import PipelineConfig
from .data_models import (
    ClusteringResult,
    BatchResult,
    PerformanceReport,
    RetrainingEvent,
    PerformanceAlert,
    DriftDetectionResult,
)
from .exceptions import (
    PipelineError,
    ConfigurationError,
    ValidationError,
    ClusteringError,
    MLflowIntegrationError,
)

__all__ = [
    "PipelineConfig",
    "ClusteringResult",
    "BatchResult",
    "PerformanceReport",
    "RetrainingEvent",
    "PerformanceAlert",
    "DriftDetectionResult",
    "PipelineError",
    "ConfigurationError",
    "ValidationError",
    "ClusteringError",
    "MLflowIntegrationError",
]