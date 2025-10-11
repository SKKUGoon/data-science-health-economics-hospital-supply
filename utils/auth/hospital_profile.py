import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, validator
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

    @validator('hospital_name')
    def validate_hospital_name(cls, v):
        """Validate hospital name is not empty and has reasonable length."""
        if not v or not v.strip():
            raise ValueError('Hospital name cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Hospital name must be at least 2 characters long')
        if len(v.strip()) > 200:
            raise ValueError('Hospital name cannot exceed 200 characters')
        return v.strip()

    @validator('hospital_id')
    def validate_hospital_id(cls, v):
        """Validate hospital ID format."""
        if not v or not v.strip():
            raise ValueError('Hospital ID cannot be empty')
        # Allow UUID format or custom format with alphanumeric and hyphens
        if not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('Hospital ID can only contain alphanumeric characters, hyphens, and underscores')
        return v.strip()

    @validator('contact_info', 'settings', 'metadata')
    def validate_dict_fields(cls, v):
        """Ensure dict fields are not None."""
        return v if v is not None else {}

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


# class HospitalManager:
#     """
#     Simple hospital profile manager for session management.

#     This is a basic implementation that can be extended with proper
#     authentication and persistence mechanisms.
#     """

#     def __init__(self):
#         self._current_hospital: Optional[HospitalProfile] = None
#         self._is_logged_in = False

#     def login(self, hospital_profile: HospitalProfile) -> bool:
#         """
#         Login with a hospital profile.

#         Args:
#             hospital_profile: Hospital profile to login with

#         Returns:
#             True if login successful, False otherwise
#         """
#         if not hospital_profile.is_valid_for_operations():
#             return False

#         self._current_hospital = hospital_profile
#         self._is_logged_in = True
#         return True

#     def logout(self) -> None:
#         """Logout current hospital."""
#         self._current_hospital = None
#         self._is_logged_in = False

#     def is_logged_in(self) -> bool:
#         """Check if a hospital is currently logged in."""
#         return self._is_logged_in and self._current_hospital is not None

#     def get_current_hospital(self) -> Optional[HospitalProfile]:
#         """Get current hospital profile."""
#         return self._current_hospital if self._is_logged_in else None

#     def update_current_hospital(self, **kwargs) -> bool:
#         """
#         Update current hospital profile.

#         Args:
#             **kwargs: Fields to update

#         Returns:
#             True if update successful, False otherwise
#         """
#         if not self.is_logged_in():
#             return False

#         try:
#             # Create updated profile
#             current_data = self._current_hospital.dict()
#             current_data.update(kwargs)
#             current_data['updated_at'] = datetime.now()

#             updated_profile = HospitalProfile(**current_data)
#             self._current_hospital = updated_profile
#             return True
#         except Exception:
#             return False


# # Global hospital manager instance
# _hospital_manager = HospitalManager()


# def get_hospital_manager() -> HospitalManager:
#     """Get the global hospital manager instance."""
#     return _hospital_manager


# def create_hospital_profile(
#     hospital_name: str,
#     hospital_id: Optional[str] = None,
#     is_admin: bool = False,
#     **kwargs
# ) -> HospitalProfile:
#     """
#     Create a new hospital profile with validation.

#     Args:
#         hospital_name: Name of the hospital
#         hospital_id: Optional hospital ID (will generate UUID if not provided)
#         is_admin: Whether this profile has admin privileges
#         **kwargs: Additional fields for the profile

#     Returns:
#         HospitalProfile instance
#     """
#     if hospital_id is None:
#         hospital_id = str(uuid.uuid4())

#     return HospitalProfile(
#         hospital_id=hospital_id,
#         hospital_name=hospital_name,
#         is_admin=is_admin,
#         **kwargs
#     )
