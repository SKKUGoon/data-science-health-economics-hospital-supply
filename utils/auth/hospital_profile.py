import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, field_validator
import re


class HospitalProfile(BaseModel):
    """Hospital profile data structure using Pydantic"""
    hospital_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique hospital identifier")
    hospital_name: str = Field(..., description="Hospital name")
    is_admin: bool = Field(default=False, description="Administrator privileges flag")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    contact_info: Optional[Dict[str, Any]] = Field(default_factory=dict)
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True

    @field_validator('hospital_name')
    def validate_hospital_name(cls, v):
        """Validate hospital name is not empty and has reasonable length."""
        if not v or not v.strip():
            raise ValueError('Hospital name cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Hospital name must be at least 2 characters long')
        if len(v.strip()) > 200:
            raise ValueError('Hospital name cannot exceed 200 characters')
        return v.strip()

    @field_validator('hospital_id')
    def validate_hospital_id(cls, v):
        """Validate hospital ID format."""
        if not v or not v.strip():
            raise ValueError('Hospital ID cannot be empty')
        # Allow UUID format or custom format with alphanumeric and hyphens
        if not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('Hospital ID can only contain alphanumeric characters, hyphens, and underscores')
        return v.strip()

    @field_validator('contact_info', 'settings', 'metadata')
    def validate_dict_fields(cls, v):
        """Ensure dict fields are not None."""
        return v if v is not None else {}

    # Mlflow operations
    def mlflow_prep_experiment_name(self) -> str:
        """Generate Mlflow experiment name for preparation experiment. - embedding"""
        return f"{self.hospital_id}_prep"

    def mlflow_main_experiment_name(self) -> str:
        """Generate Mlflow experiment name for main experiment. - modeling"""
        return f"{self.hospital_id}_main"

    # Qdrant vector database operations
    def qdrant_collection_name(self) -> str:
        """Generate Qdrant collection name for this hospital."""
        return f"{self.hospital_id}_patient_embedding"

    def qdrant_collection_name_with_suffix(self, suffix: str) -> str:
        """Generate Qdrant collection name with custom suffix."""
        return f"{self.hospital_id}_{suffix}"

    def qdrant_document_id(self, patient_id: str, date: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate Qdrant document ID in format: [hospital_id]__[patient_id]__[date]__[uuid]

        Args:
            patient_id: Patient identifier
            date: Date string (YYYYMMDD format). If None, uses current date

        Returns:
            Tuple of (document_id, uuid) where document_id is the formatted string
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        document_uuid = str(uuid.uuid4())
        document_id = f"{self.hospital_id}__{patient_id}__{date}"

        return document_id, document_uuid

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value with optional default."""
        return self.settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value."""
        if self.settings is None:
            self.settings = {}
        self.settings[key] = value
        self.updated_at = datetime.now()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value with optional default."""
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value
        self.updated_at = datetime.now()

    def update_contact_info(self, **kwargs) -> None:
        """Update contact information."""
        if self.contact_info is None:
            self.contact_info = {}
        self.contact_info.update(kwargs)
        self.updated_at = datetime.now()

    def is_valid_for_operations(self) -> bool:
        """Check if hospital profile has minimum required information for operations."""
        return (
            bool(self.hospital_id and self.hospital_id.strip()) and
            bool(self.hospital_name and self.hospital_name.strip())
        )

    def get_display_name(self) -> str:
        """Get display name for UI purposes."""
        return self.hospital_name or self.hospital_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper datetime serialization."""
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HospitalProfile':
        """Create HospitalProfile from dictionary."""
        return cls(**data)
