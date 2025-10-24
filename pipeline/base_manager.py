from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel
import pandas as pd
import mlflow
import joblib

from core.data_models import EmbeddingRetrieval
from utils.auth.hospital_profile import HospitalProfile


class TransformerBase(ABC):
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class MLModelBase(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass


class EvaluatorBase(ABC):
    @abstractmethod
    def evaluate(self):
        pass

class MLPipelineResult(BaseModel):
    profile: HospitalProfile


class MLPipeline(BaseModel):
    profile: HospitalProfile

    # Dataset
    original: pd.DataFrame
    embedding: pd.DataFrame

    # Bases
    transformer: Optional[TransformerBase] = None
    model: Optional[MLModelBase] = None
    evaluator: Optional[EvaluatorBase] = None

    model_config = {
        'arbitrary_types_allowed': True
    }

    def run(self, run_name: str):
        mlflow.set_experiment(self.profile.mlflow_main_experiment_name())

        with mlflow.start_run(run_name=run_name):
            # All the data
            m = pd.merge(
                self.original[['id', 'date', 'visit_date', 'prescription', 'quantity']],
                self.embedding,
                on=['id', 'date'],
            )
            assert m.shape[0] == self.original.shape[0]

            # Some transformation
            if self.transformer is not None:
                m = self.transformer.fit_transform(m)
            else:
                print("[Warning] data transformer not set")

            if self.model is not None:
                models = self.model.fit(m)
            else:
                print("[Warning] model not set")

        return m, models


