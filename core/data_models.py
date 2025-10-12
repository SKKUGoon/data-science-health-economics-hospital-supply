from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field, model_validator
import pandas as pd
import numpy as np


class EmbeddingMetrics(BaseModel):
    """
    Metrics for embedding operations.

    This model tracks performance and quality metrics
    for embedding generation and storage operations.
    """
    total_documents: int = Field(..., ge=0, description="Total number of documents processed")
    total_embeddings: int = Field(..., ge=0, description="Total number of embeddings generated")
    avg_embedding_time: float = Field(..., ge=0.0, description="Average time to generate embeddings in seconds")
    total_processing_time: float = Field(..., ge=0.0, description="Total processing time in seconds")
    documents_per_second: float = Field(..., ge=0.0, description="Processing rate in documents per second")
    batch_count: int = Field(..., ge=0, description="Number of batches processed")
    avg_batch_time: float = Field(..., ge=0.0, description="Average batch processing time in seconds")
    error_count: int = Field(default=0, ge=0, description="Number of errors encountered")

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


class EmbeddingResult(BaseModel):
    """
    Result of embedding operation.

    This model represents the outcome of document embedding
    with comprehensive metrics and validation.
    """
    success: bool = Field(..., description="Whether embedding operation was successful")
    collection_name: str = Field(..., min_length=1, description="Name of the vector collection")
    metrics: EmbeddingMetrics = Field(..., description="Performance metrics for the embedding operation")
    dataset_stats: Dict[str, Any] = Field(default_factory=dict, description="Statistics about the processed dataset")
    timestamp: datetime = Field(default_factory=datetime.now, description="When embedding operation was completed")
    error_message: Optional[str] = Field(default=None, description="Error message if operation failed")
    hospital_profile_id: Optional[str] = Field(default=None, description="Hospital profile used for embedding")

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


class EmbeddingRetrieval(BaseModel):
    """
    Result of embedding retrieval operation.
    """
    metadata: pd.DataFrame = Field(..., description="Metadata of the retrieved documents")
    embeddings: np.ndarray = Field(..., description="Embeddings of the retrieved documents")
    embsize: int = Field(..., ge=0, description="Size of the embeddings")

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "arbitrary_types_allowed": True
    }

    @model_validator(mode='after')
    def validate_same_data_length(self, v):
        if self.metadata.shape[0] != self.embeddings.shape[0]:
            raise ValueError("Metadata and embeddings must have the same number of rows")
        return self

    def mergeable_embeddings(self):
        """
        Returns embeddings with 'id' and 'date' primary key attached
        """
        emb = pd.DataFrame(
            self.embeddings,
            index=self.metadata.index,
            columns=[f'emb{i+1}' for i in range(self.embsize)]
        )
        return self.metadata[['id', 'date']].join(emb)
