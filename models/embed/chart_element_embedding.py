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

from core.data_models import EmbeddingResult
from utils.auth.hospital_profile import HospitalProfile
from utils.logging.pipeline_logger import PipelineLogger


class ChartElementEmbedding:
    """
    Patient chart embedding service. But embed using the document's element, 
    not the whole document.

    This class handles the generation and storage of embeddings for pateitn
    chart data using OpenAI's embedding API and Qdrant vector database,
    with integrated logging, configuration, and validation.
    """
    def __init__(
        self,
        profile: HospitalProfile,
        model: str = "text-embedding-3-large",
        vector_size: int = 3072,
        batch_size: int = 50,
    ):
        """
        Initialize ChartElementEmbedding with core components.

        Args:
            profile: Hospital profile
            vector_size: Dimension of embedding vectors
            batch_size: Batch size for embedding
        """
        self.profile = profile
        self.logger = PipelineLogger("ChartElementEmbedding")

        self.collection_name_map: Dict[str, str] = {
            "department": self.profile.qdrant_collection_name_with_suffix("department"),
            "diagnosis": self.profile.qdrant_collection_name_with_suffix("diagnosis"),
            "prescription": self.profile.qdrant_collection_name_with_suffix("prescription"),
        }
        self.vector_size = vector_size
        self.batch_size = batch_size
        self.client: Optional[QdrantClient] = None
        self.embedding_model = model
        
        # Cache for embeddings to avoid duplicate API calls
        self._embedding_cache: Dict[str, List[float]] = {}

        # Initialize OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None or openai.api_key == "":
            raise ValueError("OPENAI_API_KEY is not set")

        self.logger.info("ChartElementEmbedding initialized", {
            "hospital_profile": self.profile.hospital_name,
            "embedding_model": self.embedding_model,
            "vector_size": self.vector_size,
            "batch_size": self.batch_size,
        })

    def initialize_qdrant(self, host: str = "localhost", port: int = 6333):
        """Initialize Qdrant client and create collections with enhanced logging."""
        with self.logger.time_operation("qdrant_initialization"):
            with mlflow.start_run(nested=True, run_name="qdrant_setup"):
                self.client = QdrantClient(host=host, port=port)

                # Log parameters using both MLflow and pipeline logger
                params = {
                    "qdrant_host": host,
                    "qdrant_port": port,
                    "collection_names": list(self.collection_name_map.values()),
                    "vector_size": self.vector_size,
                    "batch_size": self.batch_size,
                }
                self.logger.log_parameters(params)

                for collection_name in self.collection_name_map.values():
                    if not self.client.collection_exists(collection_name):
                        self.client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=self.vector_size,
                                distance=Distance.COSINE
                            )
                        )
                        self.logger.log_parameters({
                            "collection_created": True,
                            "collection_exists": False
                        })
                    else:
                        self.logger.info(f"Using existing collection: {collection_name}")
                        self.logger.log_parameters({
                            "collection_created": False,
                            "collection_exists": True
                        })

    def get_embedding_single(self, text: str) -> List[float]:
        """Get embedding from openAI with caching to avoid duplicate API calls"""
        # Check cache first
        if text in self._embedding_cache:
            self.logger.debug(f"Using cached embedding for text: {text[:50]}...")
            return self._embedding_cache[text]
        
        # Generate new embedding
        response = openai.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        embedding = response.data[0].embedding
        
        # Cache the result
        self._embedding_cache[text] = embedding
        self.logger.debug(f"Cached new embedding for text: {text[:50]}...")
        
        return embedding

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache"""
        return {
            "cache_size": len(self._embedding_cache),
            "cache_keys": list(self._embedding_cache.keys())[:10]  # First 10 keys for debugging
        }

    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()
        self.logger.info("Embedding cache cleared")

    def prewarm_cache(self, df: pd.DataFrame):
        """Pre-warm the cache by processing all groups and caching their document texts"""
        self.logger.info("Pre-warming embedding cache by processing all groups...")
        
        colname_id, colname_date = 'id', 'date'
        grouped_data = df.groupby(by=[colname_id, colname_date])
        
        texts_to_embed = set()
        
        # Process each group to generate the actual documents that will be created
        for idx, gdf in tqdm(grouped_data, desc="Pre-warming cache with group documents"):
            try:
                # Generate the same documents that will be created during actual processing
                department_document = self._create_department_document(gdf)
                diagnosis_document = self._create_diagnosis_document(gdf)
                prescription_document = self._create_prescription_document(gdf)
                
                # Add to set of texts to embed (set automatically handles duplicates)
                texts_to_embed.add(department_document)
                texts_to_embed.add(diagnosis_document)
                texts_to_embed.add(prescription_document)
                
            except Exception as e:
                self.logger.warning(f"Error pre-warming cache for group {idx}: {e}")
                continue
        
        # Generate embeddings for all unique group documents
        for text in tqdm(texts_to_embed, desc="Generating embeddings for unique group documents"):
            if text not in self._embedding_cache:
                self.get_embedding_single(text)
        
        self.logger.info(f"Cache pre-warmed with {len(texts_to_embed)} unique group documents")

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
    
    def create_element_document(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        resume_from: Optional[int] = None,
        prewarm_cache: bool = True
    ):
        """Create element document embeddings with enhanced error handling and metrics."""
        if self.client is None:
            raise ValueError("Qdrant client is not initialized")

        # Use batch_size from config if not provided
        if batch_size is None:
            batch_size = self.batch_size

        rfr = resume_from if resume_from is not None else 0

        if resume_from is not None:
            self.logger.info(f"Resuming from group number {resume_from}")
        
        try:
            with self.logger.time_operation("element_embedding_pipeline"):
                start_time = time.time()

                # Validate dataframe structure
                self._validate_dataframe(df)
                
                # Pre-warm cache if requested
                if prewarm_cache:
                    self.prewarm_cache(df)

                colname_id, colname_date = 'id', 'date'
                department_batch_insert: List[PointStruct] = []
                diagnosis_batch_insert: List[PointStruct] = []
                prescription_batch_insert: List[PointStruct] = []
                grouped_data = df.groupby(by=[colname_id, colname_date])
                total_groups = len(grouped_data)

                process_count,batch_count,error_count = 0, 0, 0

                for idx, gdf in tqdm(grouped_data, desc="Processing element documents"):
                    if process_count < rfr:
                        # For abrupt stop of embedding, and resuming from that line
                        process_count += 1
                        continue

                    try:
                        id_, dt_ = idx

                        doc_info, doc_id = self.profile.qdrant_document_id(id_, dt_.strftime("%Y-%m-%d"))

                        department_document = self._create_department_document(gdf)
                        department_metadata = self._create_department_metadata(gdf)
                        department_metadata.update({"id": int(id_), "date": dt_.strftime("%Y-%m-%d"), "doc_info": doc_info})
                        department_embedding = self.get_embedding_single(department_document)
                        department_point = PointStruct(
                            id=doc_id,
                            vector=department_embedding,
                            payload=department_metadata,
                        )
                        department_batch_insert.append(department_point)

                        diagnosis_document = self._create_diagnosis_document(gdf)
                        diagnosis_metadata = self._create_diagnosis_metadata(gdf)
                        diagnosis_metadata.update({"id": int(id_), "date": dt_.strftime("%Y-%m-%d"), "doc_info": doc_info})
                        diagnosis_embedding = self.get_embedding_single(diagnosis_document)
                        diagnosis_point = PointStruct(
                            id=doc_id,
                            vector=diagnosis_embedding,
                            payload=diagnosis_metadata,
                        )
                        diagnosis_batch_insert.append(diagnosis_point)

                        prescription_document = self._create_prescription_document(gdf)
                        prescription_metadata = self._create_prescription_metadata(gdf)
                        prescription_metadata.update({"id": int(id_), "date": dt_.strftime("%Y-%m-%d"), "doc_info": doc_info})
                        prescription_embedding = self.get_embedding_single(prescription_document)
                        prescription_point = PointStruct(
                            id=doc_id,
                            vector=prescription_embedding,
                            payload=prescription_metadata,
                        )
                        prescription_batch_insert.append(prescription_point)

                        process_count += 1

                        if len(department_batch_insert) >= batch_size:
                            batch_start = time.time()
                            self.client.upsert(
                                collection_name=self.collection_name_map["department"],
                                points=department_batch_insert
                            )
                            batch_count += 1
                            department_batch_insert = []

                        if len(diagnosis_batch_insert) >= batch_size:
                            batch_start = time.time()
                            self.client.upsert(
                                collection_name=self.collection_name_map["diagnosis"],
                                points=diagnosis_batch_insert
                            )
                            batch_count += 1
                            diagnosis_batch_insert = []
                        
                        if len(prescription_batch_insert) >= batch_size:
                            batch_start = time.time()
                            self.client.upsert(
                                collection_name=self.collection_name_map["prescription"],
                                points=prescription_batch_insert
                            )
                            batch_count += 1
                            prescription_batch_insert = []
                    
                    except Exception as e:
                        error_count += 1
                        self.logger.error(f"Error processing document {idx}: {e}", exc_info=True)
                        continue
                

                if department_batch_insert:
                    self.client.upsert(
                        collection_name=self.collection_name_map["department"],
                        points=department_batch_insert
                    )
                    batch_count += 1
                    department_batch_insert = []
                    
                if diagnosis_batch_insert:
                    self.client.upsert(
                        collection_name=self.collection_name_map["diagnosis"],
                        points=diagnosis_batch_insert
                    )
                    batch_count += 1
                    diagnosis_batch_insert = []
                    
                if prescription_batch_insert:
                    self.client.upsert(
                        collection_name=self.collection_name_map["prescription"],
                        points=prescription_batch_insert
                    )
                    batch_count += 1
                    prescription_batch_insert = []

                # Log cache statistics
                cache_stats = self.get_cache_stats()
                self.logger.info("Embedding pipeline completed", {
                    "total_processed": process_count,
                    "total_batches": batch_count,
                    "total_errors": error_count,
                    "cache_size": cache_stats["cache_size"],
                    "unique_embeddings_generated": cache_stats["cache_size"]
                })

                return True
        
        except Exception as e:
            self.logger.error(f"Embedding pipeline failed: {e}", exc_info=True)
            return False

    def _create_department_document(self, df: pd.DataFrame) -> str:
        colname_department = 'department'
        return " | ".join(sorted(df[colname_department].dropna().unique()))

    def _create_department_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        colname_department = 'department'
        return {
            "department": " | ".join(sorted(df[colname_department].dropna().unique()))
        }

    def _create_diagnosis_document(self, df: pd.DataFrame) -> str:
        colname_primary_diagnosis = 'primary_diagnosis'
        colname_secondary_diagnosis = 'secondary_diagnosis'

        pdiag = df[colname_primary_diagnosis].dropna().unique().tolist()
        sdiag = df[colname_secondary_diagnosis].dropna().unique().tolist()
        
        return " | ".join(sorted(pdiag + sdiag))

    def _create_diagnosis_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        colname_primary_diagnosis = 'primary_diagnosis'
        colname_secondary_diagnosis = 'secondary_diagnosis'

        pdiag = df[colname_primary_diagnosis].dropna().unique().tolist()
        sdiag = df[colname_secondary_diagnosis].dropna().unique().tolist()

        return {
            "primary_diagnosis": " | ".join(sorted(pdiag)),
            "secondary_diagnosis": " | ".join(sorted(sdiag))
        }

    def _create_prescription_document(self, df: pd.DataFrame) -> str:
        colname_prescription = 'prescription'
        return " | ".join(sorted(df[colname_prescription].dropna().unique()))

    def _create_prescription_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        colname_prescription = 'prescription'
        return {
            "prescription": " | ".join(sorted(df[colname_prescription].dropna().unique()))
        }

    def retrieve_embedding(
        self,
        dt_start_str: Optional[str] = None,
        dt_end_str: Optional[str] = None,
        retreive_limit: int = 99_999,
        date_fmt: str = '%Y-%m-%d'
    ):
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

        idx_dep, idx_diag, idx_pres = [], [], []
        meta_dep, meta_diag, meta_pres = [], [], []
        vector_dep, vector_diag, vector_pres = [], [], []
        with self.logger.time_operation("qdrant_retrieval"):
            for y in tqdm(range(dts_y, dte_y + 1), desc="Retrieving element vectors"):
                scroll_filter = Filter(
                    must=[
                        FieldCondition(
                            key="year",
                            match=MatchValue(value=y)
                        )
                    ]
                )
                department_vectors = self.client.scroll(
                    collection_name=self.collection_name_map["department"],
                    limit=retreive_limit,
                    with_vectors=True,
                    scroll_filter=scroll_filter,
                )[0]
                diagnosis_vectors = self.client.scroll(
                    collection_name=self.collection_name_map["diagnosis"],
                    limit=retreive_limit,
                    with_vectors=True,
                    scroll_filter=scroll_filter,
                )[0]
                prescription_vectors = self.client.scroll(
                    collection_name=self.collection_name_map["prescription"],
                    limit=retreive_limit,
                    with_vectors=True,
                    scroll_filter=scroll_filter,
                )[0]

                for i in department_vectors:
                    idx_dep.append(i.id)
                    meta_dep.append([i.payload['id'], i.payload['date'], i.payload['department']])
                    vector_dep.append(i.vector)
                
                for i in diagnosis_vectors:
                    idx_diag.append(i.id)
                    meta_diag.append([
                        i.payload['id'], 
                        i.payload['date'], 
                        i.payload['primary_diagnosis'],
                        i.payload['secondary_diagnosis'],
                    ])
                    vector_diag.append(i.vector)
                
                for i in prescription_vectors:
                    idx_pres.append(i.id)
                    meta_pres.append([i.payload['id'], i.payload['date'], i.payload['prescription']])
                    vector_pres.append(i.vector)

        df_meta_dep = pd.DataFrame(
            meta_dep, 
            index=idx_dep, 
            columns=['id', 'date', 'department']
        )
        df_meta_diag = pd.DataFrame(
            meta_diag, 
            index=idx_diag, 
            columns=['id', 'date', 'primary_diagnosis', 'secondary_diagnosis']
        )
        df_meta_pres = pd.DataFrame(
            meta_pres,
            index=idx_pres,
            columns=['id', 'date', 'prescription']
        )

        df_meta_dep['date'] = pd.to_datetime(df_meta_dep['date'], format=date_fmt)
        df_meta_diag['date'] = pd.to_datetime(df_meta_diag['date'], format=date_fmt)
        df_meta_pres['date'] = pd.to_datetime(df_meta_pres['date'], format=date_fmt)

        vector_dep = np.array(vector_dep)
        vector_diag = np.array(vector_diag)
        vector_pres = np.array(vector_pres)

        print(f"vector_dep shape: {vector_dep.shape}")
        print(f"vector_diag shape: {vector_diag.shape}")
        print(f"vector_pres shape: {vector_pres.shape}")

        return {
            "idx_department": idx_dep,
            "idx_diagnosis": idx_diag,
            "idx_prescription": idx_pres,
            "vector_department": vector_dep,
            "vector_diagnosis": vector_diag,
            "vector_prescription": vector_pres,
            "meta_department": df_meta_dep,
            "meta_diagnosis": df_meta_diag,
            "meta_prescription": df_meta_pres,
        }


        