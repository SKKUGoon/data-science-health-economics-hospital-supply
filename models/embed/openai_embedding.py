from typing import List, Dict, Optional, Any
import os
import time
from datetime import datetime

from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
import pandas as pd
import mlflow
import openai
from tqdm import tqdm

from utils.auth.hospital_profile import HospitalProfile


class PatientChartEmbedding:
    embedding_model = "text-embedding-ada-002"

    def __init__(self, profile: HospitalProfile, vector_size: int = 1536):
        print("PatientChartEmbedding starting...")
        self.profile = profile
        self.collection_name = profile.qdrant_collection_name()
        self.vector_size = vector_size
        self.client: Optional[QdrantClient] = None

        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None or openai.api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")

        print("PatientChartEmbedding ready")

    def initialize_qdrant(self, host: str = "localhost", port: int = 6333):
        """Initialize Qdrant client and create collection"""
        with mlflow.start_run(nested=True, run_name="qdrant_setup"):
            self.client = QdrantClient(host=host, port=port)

            mlflow.log_param("qdrant_host", host)
            mlflow.log_param("qdrant_port", port)
            mlflow.log_param("collection_name", self.collection_name)
            mlflow.log_param("vector_size", self.vector_size)

            if not self.client.collection_exists(self.collection_name):
                print(f"PatientCharEmbedding has created collection {self.collection_name}")
                print(f"PatientCharEmbedding is using collection {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                mlflow.log_param("collection_created", True)
                mlflow.log_param("collection_exists", True)

            else:
                print(f"PatientCharEmbedding is using collection {self.collection_name}")
                mlflow.log_param("collection_created", False)
                mlflow.log_param("collection_exists", True)

    @staticmethod
    def get_embedding_single(text: str) -> List[float]:
        """Get embedding from openAI"""
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    @staticmethod
    def _check_keys(cols: List[str]) -> bool:
        # ID keys
        assert 'id' in cols
        assert 'date' in cols

        # Document keys
        assert 'sex' in cols
        assert 'age' in cols
        assert 'department' in cols
        assert 'primary_diagnosis' in cols
        assert 'secondary_diagnosis' in cols
        assert 'prescription' in cols
        return True

    def _log_dataset_stats(self, df: pd.DataFrame) -> None:
        """Log comprehensive dataset statistics to MLflow"""
        # Basic dataset info
        mlflow.log_metric("total_rows", len(df))
        mlflow.log_metric("total_columns", len(df.columns))
        mlflow.log_metric("unique_patients", df['id'].nunique())
        mlflow.log_metric("unique_dates", df['date'].nunique())

        # Date range
        date_min = df['date'].min()
        date_max = df['date'].max()
        mlflow.log_param("date_range_start", date_min.strftime("%Y-%m-%d"))
        mlflow.log_param("date_range_end", date_max.strftime("%Y-%m-%d"))
        mlflow.log_metric("date_range_days", (date_max - date_min).days)

        # Patient demographics
        mlflow.log_metric("avg_patient_age", df['age'].mean())
        mlflow.log_metric("min_patient_age", df['age'].min())
        mlflow.log_metric("max_patient_age", df['age'].max())
        mlflow.log_metric("male_patients_pct", (df['sex'] == 1).mean() * 100)

        # Medical data stats
        mlflow.log_metric("unique_departments", df['department'].nunique())
        mlflow.log_metric("unique_primary_diagnoses", df['primary_diagnosis'].nunique())
        mlflow.log_metric("unique_secondary_diagnoses", df['secondary_diagnosis'].nunique())
        mlflow.log_metric("unique_prescriptions", df['prescription'].nunique())

        # Data quality metrics
        mlflow.log_metric("missing_primary_diagnosis_pct", df['primary_diagnosis'].isna().mean() * 100)
        mlflow.log_metric("missing_secondary_diagnosis_pct", df['secondary_diagnosis'].isna().mean() * 100)
        mlflow.log_metric("missing_prescription_pct", df['prescription'].isna().mean() * 100)

        # Top categories (log as parameters)
        top_departments = df['department'].value_counts().head(5).to_dict()
        top_primary_dx = df['primary_diagnosis'].value_counts().head(5).to_dict()

        mlflow.log_param("top_departments", str(top_departments))
        mlflow.log_param("top_primary_diagnoses", str(top_primary_dx))

    def create_patient_document_by_date(self, df: pd.DataFrame, batch_size: int = 50, resume_from: Optional[int] = None) -> None:
        if self.client is None:
            raise ValueError("Qdrant client is not initiated")

        if resume_from is not None:
            print(f"Resuming from group number {resume_from}...")

        rfr = resume_from if resume_from is not None else 0

        with mlflow.start_run(nested=True, run_name="patient_embedding_pipeline"):
            start_time = time.time()

            # Check the dataframe name
            self._check_keys(df.columns)

            # Log dataset statistics
            self._log_dataset_stats(df)

            # Log pipeline parameters
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("embedding_model", self.embedding_model)
            mlflow.log_param("vector_size", self.vector_size)
            mlflow.log_param("hospital_profile", self.profile.hospital_name)
            mlflow.log_param("collection_name", self.collection_name)

            colname_id = 'id'
            colname_date = 'date'

            batch_insert: List[PointStruct] = []
            grouped_data = df.groupby(by=[colname_id, colname_date])
            total_groups = len(grouped_data)

            mlflow.log_metric("total_patient_documents", total_groups)

            processed_count = 0
            batch_count = 0
            embedding_times = []

            for idx, gdf in tqdm(grouped_data, desc="Processing patient documents"):
                if processed_count < rfr:
                    # For abrupt stop of embedding, and resuming from that line
                    # e.g. If the datapoint only has 57000 points, you should set `resume_from` 57000
                    processed_count += 1
                    continue

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

                    # Log batch metrics
                    mlflow.log_metric("batch_upsert_time", batch_time, step=batch_count)
                    mlflow.log_metric("processed_documents", processed_count, step=batch_count)

                    batch_insert = []

            # Handle remaining batch
            if batch_insert:
                batch_start = time.time()
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_insert
                )
                batch_time = time.time() - batch_start
                batch_count += 1
                mlflow.log_metric("batch_upsert_time", batch_time, step=batch_count)

            # Log final pipeline metrics
            total_time = time.time() - start_time
            avg_embedding_time = sum(embedding_times) / len(embedding_times) if embedding_times else 0

            mlflow.log_metric("total_pipeline_time", total_time)
            mlflow.log_metric("avg_embedding_time", avg_embedding_time)
            mlflow.log_metric("documents_per_second", processed_count / total_time if total_time > 0 else 0)
            mlflow.log_metric("total_batches", batch_count)
            mlflow.log_metric("final_processed_count", processed_count)

            # Log pipeline completion
            mlflow.log_param("pipeline_completed_at", datetime.now().isoformat())
            mlflow.log_param("pipeline_status", "completed")

            print(f"Pipeline completed: {processed_count} documents processed in {total_time:.2f}s")

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