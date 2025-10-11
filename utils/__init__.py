"""
Shared utilities for the MLflow Time Series Clustering Pipeline.

This module provides essential utilities that are used across all components of the pipeline:

**Authentication & Hospital Management:**
- `HospitalProfile`: Pydantic-based hospital profile management
- Multi-tenant support with hospital-specific configurations
- Session management and authentication context
- Qdrant collection naming and document ID generation

**Logging Infrastructure:**
- `PipelineLogger`: Enhanced logging with MLflow integration
- Consistent log formatting across all components
- Performance metrics logging and timing operations
- Context-aware logging with structured data



**Key Features:**
- Hospital-specific data isolation and security
- Centralized logging with experiment tracking

**Usage Example:**

```python
from utils import PipelineLogger, validate_patient_dataframe, HospitalProfile

# Set up logging with MLflow integration
logger = PipelineLogger("MyComponent", enable_mlflow=True)
logger.info("Processing started", {"batch_size": 100})

# Hospital profile management
profile = HospitalProfile(
    hospital_name="General Hospital",
    hospital_id="gh_001"
)
collection_name = profile.qdrant_collection_name()
```

"""
