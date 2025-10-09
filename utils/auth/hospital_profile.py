import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field


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

    def qdrant_collection_name(self):
        return f"{self.hospital_id}_patient_embedding"

    def qdrant_document_id(self, patient_id: str, date: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate Qdrant document ID in format: [hospital_id]__[patient_id]__[date]__[uuid]

        Args:
            hospital_id: Hospital identifier
            patient_id: Patient identifier
            date: Date string (YYYYMMDD format). If None, uses current date

        Returns:
            Formatted document ID string
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        document_uuid = str(uuid.uuid4())

        return f"{self.hospital_id}__{patient_id}__{date}", document_uuid
