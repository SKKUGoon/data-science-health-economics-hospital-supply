from typing import List, Dict, Optional, Any
import os
import time
from datetime import datetime

from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
import pandas as pd
import numpy as np
import mlflow
import openai
from tqdm import tqdm

from core.data_models import EmbeddingResult, EmbeddingMetrics, EmbeddingRetrieval
from utils.auth.hospital_profile import HospitalProfile
from utils.logging.pipeline_logger import PipelineLogger



class PatientChartEmbedding:
    """
    Patient chart embedding service with core component integration.

    This class handles the generation and storage of embeddings for patient
    chart data using OpenAI's embedding API and Qdrant vector database,
    with integrated logging, configuration, and validation.
    """

    def __init__(
        self,
        profile: HospitalProfile,
        vector_size: int = 1536,
        batch_size: int = 50,
    ):
        """
        Initialize PatientChartEmbedding with core components.

        Args:
            config: Pipeline configuration
            profile: Hospital profile (optional, can be loaded from config)
            vector_size: Dimension of embedding vectors
        """
        self.profile = profile
        self.logger = PipelineLogger("PatientChartEmbedding")

        self.collection_name = self.profile.qdrant_collection_name()
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.client: Optional[QdrantClient] = None
        self.embedding_model = "text-embedding-ada-002"

        # Initialize OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None or openai.api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")

        self.logger.info("PatientChartEmbedding initialized", {
            "hospital_profile": self.profile.hospital_name,
            "collection_name": self.collection_name,
            "vector_size": self.vector_size,
            "embedding_model": self.embedding_model
        })

    def initialize_qdrant(self, host: str = "localhost", port: int = 6333):
        """Initialize Qdrant client and create collection with enhanced logging."""
        with self.logger.time_operation("qdrant_initialization"):
            with mlflow.start_run(nested=True, run_name="qdrant_setup"):
                self.client = QdrantClient(host=host, port=port)

                # Log parameters using both MLflow and pipeline logger
                params = {
                    "qdrant_host": host,
                    "qdrant_port": port,
                    "collection_name": self.collection_name,
                    "vector_size": self.vector_size
                }
                self.logger.log_parameters(params)

                collection_exists = self.client.collection_exists(self.collection_name)

                if not collection_exists:
                    self.logger.info(f"Creating new collection: {self.collection_name}")
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    self.logger.log_parameters({
                        "collection_created": True,
                        "collection_exists": True
                    })
                else:
                    self.logger.info(f"Using existing collection: {self.collection_name}")
                    self.logger.log_parameters({
                        "collection_created": False,
                        "collection_exists": True
                    })

    @staticmethod
    def get_embedding_single(text: str) -> List[float]:
        """Get embedding from openAI"""
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate dataframe structure with basic checks."""
        required_columns = [
            'id', 'date', 'sex', 'age', 'department',
            'primary_diagnosis', 'secondary_diagnosis', 'prescription'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info("DataFrame validation passed", {
            "total_rows": len(df),
            "total_columns": len(df.columns)
        })
        return True

    def _log_dataset_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Log comprehensive dataset statistics using pipeline logger and return stats."""
        # Basic dataset info
        basic_metrics = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "unique_patients": df['id'].nunique(),
            "unique_dates": df['date'].nunique()
        }

        # Date range
        date_min = df['date'].min()
        date_max = df['date'].max()
        date_metrics = {
            "date_range_days": (date_max - date_min).days
        }
        date_params = {
            "date_range_start": date_min.strftime("%Y-%m-%d"),
            "date_range_end": date_max.strftime("%Y-%m-%d")
        }

        # Patient demographics
        demographic_metrics = {
            "avg_patient_age": df['age'].mean(),
            "min_patient_age": df['age'].min(),
            "max_patient_age": df['age'].max(),
            "male_patients_pct": (df['sex'] == 1).mean() * 100
        }

        # Medical data stats
        medical_metrics = {
            "unique_departments": df['department'].nunique(),
            "unique_primary_diagnoses": df['primary_diagnosis'].nunique(),
            "unique_secondary_diagnoses": df['secondary_diagnosis'].nunique(),
            "unique_prescriptions": df['prescription'].nunique()
        }

        # Data quality metrics
        quality_metrics = {
            "missing_primary_diagnosis_pct": df['primary_diagnosis'].isna().mean() * 100,
            "missing_secondary_diagnosis_pct": df['secondary_diagnosis'].isna().mean() * 100,
            "missing_prescription_pct": df['prescription'].isna().mean() * 100
        }

        # Combine all metrics
        all_metrics = {**basic_metrics, **date_metrics, **demographic_metrics,
                      **medical_metrics, **quality_metrics}

        # Log using pipeline logger (which also logs to MLflow)
        self.logger.log_metrics(all_metrics)
        self.logger.log_parameters(date_params)

        # Top categories
        top_departments = df['department'].value_counts().head(5).to_dict()
        top_primary_dx = df['primary_diagnosis'].value_counts().head(5).to_dict()

        category_params = {
            "top_departments": str(top_departments),
            "top_primary_diagnoses": str(top_primary_dx)
        }
        self.logger.log_parameters(category_params)

        # Return comprehensive stats for EmbeddingResult
        return {
            **all_metrics,
            **date_params,
            **category_params
        }

    def create_patient_document_by_date(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        resume_from: Optional[int] = None
    ) -> EmbeddingResult:
        """Create patient document embeddings with enhanced error handling and metrics."""
        if self.client is None:
            raise ValueError("Qdrant client is not initialized")

        # Use batch_size from config if not provided
        if batch_size is None:
            batch_size = self.batch_size

        rfr = resume_from if resume_from is not None else 0

        if resume_from is not None:
            self.logger.info(f"Resuming from group number {resume_from}")

        try:
            with self.logger.time_operation("patient_embedding_pipeline"):
                with mlflow.start_run(nested=True, run_name="patient_embedding_pipeline"):
                    start_time = time.time()

                    # Validate dataframe structure
                    self._validate_dataframe(df)

                    # Log dataset statistics and get stats for result
                    dataset_stats = self._log_dataset_stats(df)

                    # Log pipeline parameters
                    pipeline_params = {
                        "batch_size": batch_size,
                        "embedding_model": self.embedding_model,
                        "vector_size": self.vector_size,
                        "hospital_profile": self.profile.hospital_name,
                        "collection_name": self.collection_name
                    }
                    self.logger.log_parameters(pipeline_params)

                    colname_id = 'id'
                    colname_date = 'date'

                    batch_insert: List[PointStruct] = []
                    grouped_data = df.groupby(by=[colname_id, colname_date])
                    total_groups = len(grouped_data)

                    self.logger.log_metrics({"total_patient_documents": total_groups})

                    processed_count = 0
                    batch_count = 0
                    embedding_times = []
                    error_count = 0

                    for idx, gdf in tqdm(grouped_data, desc="Processing patient documents"):
                        if processed_count < rfr:
                            # For abrupt stop of embedding, and resuming from that line
                            processed_count += 1
                            continue

                        try:
                            id_, dt_ = idx

                            doc_info, doc_id = self.profile.qdrant_document_id(id_, dt_.strftime("%Y-%m-%d"))

                            # Time embedding creation
                            embed_start = time.time()
                            document = self._create_patient_document(gdf)
                            embedding = self.get_embedding_single(document)
                            embed_time = time.time() - embed_start
                            embedding_times.append(embed_time)

                            metadata = self._create_patient_meta(gdf)
                            metadata.update({"id": int(id_), "date": dt_.strftime("%Y-%m-%d"), "doc_info": doc_info})

                            # Create a point structure for vector database
                            point = PointStruct(
                                id=doc_id,
                                vector=embedding,
                                payload=metadata,
                            )
                            batch_insert.append(point)
                            processed_count += 1

                            if len(batch_insert) >= batch_size:
                                batch_start = time.time()
                                self.client.upsert(
                                    collection_name=self.collection_name,
                                    points=batch_insert
                                )
                                batch_time = time.time() - batch_start
                                batch_count += 1

                                # Log batch metrics using pipeline logger
                                batch_metrics = {
                                    "batch_upsert_time": batch_time,
                                    "processed_documents": processed_count
                                }
                                self.logger.log_metrics(batch_metrics, step=batch_count)

                                batch_insert = []

                        except Exception as e:
                            error_count += 1
                            self.logger.error(f"Error processing document {idx}: {e}", exc_info=True)
                            continue

                    # Handle remaining batch
                    if batch_insert:
                        try:
                            batch_start = time.time()
                            self.client.upsert(
                                collection_name=self.collection_name,
                                points=batch_insert
                            )
                            batch_time = time.time() - batch_start
                            batch_count += 1
                            self.logger.log_metrics({"batch_upsert_time": batch_time}, step=batch_count)
                        except Exception as e:
                            error_count += 1
                            self.logger.error(f"Error in final batch upsert: {e}", exc_info=True)

                    # Calculate final metrics
                    total_time = time.time() - start_time
                    avg_embedding_time = sum(embedding_times) / len(embedding_times) if embedding_times else 0
                    avg_batch_time = total_time / batch_count if batch_count > 0 else 0
                    documents_per_second = processed_count / total_time if total_time > 0 else 0

                    final_metrics = {
                        "total_pipeline_time": total_time,
                        "avg_embedding_time": avg_embedding_time,
                        "documents_per_second": documents_per_second,
                        "total_batches": batch_count,
                        "final_processed_count": processed_count,
                        "error_count": error_count
                    }
                    self.logger.log_metrics(final_metrics)

                    # Log pipeline completion
                    completion_params = {
                        "pipeline_completed_at": datetime.now().isoformat(),
                        "pipeline_status": "completed" if error_count == 0 else "completed_with_errors"
                    }
                    self.logger.log_parameters(completion_params)

                    self.logger.info(f"Pipeline completed: {processed_count} documents processed in {total_time:.2f}s")

                    # Create and return EmbeddingResult
                    metrics = EmbeddingMetrics(
                        total_documents=total_groups,
                        total_embeddings=processed_count,
                        avg_embedding_time=avg_embedding_time,
                        total_processing_time=total_time,
                        documents_per_second=documents_per_second,
                        batch_count=batch_count,
                        avg_batch_time=avg_batch_time,
                        error_count=error_count
                    )

                    return EmbeddingResult(
                        success=error_count == 0,
                        collection_name=self.collection_name,
                        metrics=metrics,
                        dataset_stats=dataset_stats,
                        hospital_profile_id=self.profile.hospital_id
                    )

        except Exception as e:
            self.logger.error(f"Embedding pipeline failed: {e}", exc_info=True)

            # Return failed result
            failed_metrics = EmbeddingMetrics(
                total_documents=0,
                total_embeddings=0,
                avg_embedding_time=0.0,
                total_processing_time=0.0,
                documents_per_second=0.0,
                batch_count=0,
                avg_batch_time=0.0,
                error_count=1
            )

            return EmbeddingResult(
                success=False,
                collection_name=self.collection_name,
                metrics=failed_metrics,
                dataset_stats={},
                error_message=str(e),
                hospital_profile_id=self.profile.hospital_profile_id
            )

    def _create_patient_document(self, df: pd.DataFrame) -> str:  # Document
        colname_sex = 'sex'
        colname_age = 'age'
        colname_dep = 'department'
        colname_primary = 'primary_diagnosis'
        colname_secondary = 'secondary_diagnosis'
        colname_prescription = 'prescription'

        feat_is_male = 'male' if int(max(df[colname_sex].unique())) == 1 else 'female'
        ages = df[colname_age].unique()
        if len(ages) > 1:
            feat_ages = f"{min(ages)} ~ {max(ages)}"
        else:
            feat_ages = f"{max(ages)}"
        feat_department = " | ".join(sorted(df[colname_dep].dropna().unique()))
        feat_primary = " | ".join(sorted(df[colname_primary].dropna().unique()))
        feat_secondary = " | ".join(sorted(df[colname_secondary].dropna().unique()))
        feat_prescription = " | ".join(sorted(df[colname_prescription].dropna().unique()))

        return f"""
[Patient information]
Sex: {feat_is_male}
Age: {feat_ages}

[Diagnosis]
Department: {feat_department}
Primary: {feat_primary}
Secondary: {feat_secondary}

[Prescriptions]
{feat_prescription}
"""

    def _create_patient_meta(self, df: pd.DataFrame) -> Dict[str, Any]:
        colname_sex = 'sex'
        colname_age = 'age'
        colname_primary = 'primary_diagnosis'
        colname_secondary = 'secondary_diagnosis'

        feat_is_male = 'male' if int(max(df[colname_sex].unique())) == 1 else 'female'
        ages = df[colname_age].unique()
        feat_ages = int(max(ages))  # Convert numpy.int64 to native Python int
        feat_primary = " | ".join(sorted(df[colname_primary].dropna().unique()))
        feat_secondary = " | ".join(sorted(df[colname_secondary].dropna().unique()))

        return {
            "sex": feat_is_male,
            "age": feat_ages,
            "diagnosis_a": feat_primary,
            "diagnosis_b": feat_secondary,
        }

    def retrieve_embedding(
        self,
        dt_start_str: Optional[str] = None,
        dt_end_str: Optional[str] = None,
        retreive_limit: int = 99_999,
        date_fmt: str = '%Y-%m-%d'
    ) -> EmbeddingRetrieval:
        """
        Retrieve embeddings from Qdrant for a given date range.

        Args:
            dt_start_str (str): Start date in string format (e.g., '2023-01-01')
            dt_end_str (str): End date in string format (e.g., '2023-01-31')

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Meta and Vectors

        Raises:
            ValueError: If the date range is invalid or if the Qdrant client is not initialized
        """
        if not self.client:
            raise ValueError("Qdrant client is not initialized")

        dts_s: str = dt_start_str if dt_start_str else '2015-01-01'
        dts_y: int = datetime.strptime(dts_s, date_fmt).year

        dte_s: str = dt_end_str if dt_end_str else '2025-12-31'
        dte_y: int = datetime.strptime(dte_s, date_fmt).year

        # Validate and parse the date range
        idx, meta, vector = [], [], []
        with self.logger.time_operation("qdrant_retrieval"):
            with mlflow.start_run(nested=True, run_name="retreive_patient_embedding"):
                start_time = time.time()

                for y in tqdm(range(dts_y, dte_y + 1), desc="Retrieving patient vectors"):
                    scroll_filter = Filter(
                        must=[
                            FieldCondition(
                                key="year",
                                match=MatchValue(value=y)
                            )
                        ]
                    )

                    vectors = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=retreive_limit,  # Adjust limit based on the size of the data
                        with_vectors=True,  # Including embedding vectors
                        scroll_filter=scroll_filter,
                    )[0]

                    for i in vectors:
                        idx.append(i.id)
                        meta.append([
                            i.payload['id'],
                            i.payload['sex'],
                            i.payload['age'],
                            i.payload['date'],
                            i.payload['diagnosis_a'],
                            i.payload['diagnosis_b'],
                            i.payload['doc_info'],
                        ])
                        vector.append(i.vector)

        meta_df: pd.DataFrame = pd.DataFrame(meta, index=idx, columns=["id", "sex", "age", "date", "diagnosis_a", "diagnosis_b", "doc_info"])
        meta_df['date'] = pd.to_datetime(meta_df['date'], format=date_fmt)

        result = EmbeddingRetrieval(
            metadata=meta_df,
            embeddings=np.array(vector),
            embsize=self.vector_size
        )

        return result

