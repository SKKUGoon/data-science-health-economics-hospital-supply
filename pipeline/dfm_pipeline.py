from typing import Literal, Dict, Tuple, Any
from urllib.parse import urlparse
from pathlib import Path
import joblib
import os
import re

from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from umap import UMAP
from tqdm import tqdm
import mlflow
import pandas as pd
import numpy as np

from pipeline.base_manager import TransformerBase, MLModelBase
from utils.logging.pipeline_logger import PipelineLogger
from models.data_container.rolling_window_feeder import RollingWindowFeeder


class DFMTransformer(TransformerBase):
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = (
            df.groupby(["visit_date", "prescription"], as_index=False)
            .agg({
                "quantity": "sum",                      # total quantity
                "id": pd.Series.nunique,                # unique patient count
                **{c: "mean" for c in df.columns if c.startswith("emb")}  # inline
            })
        )
        agg = agg.rename(
            columns={
                "visit_date": "date",
                "prescription": "substance_id",
                "quantity": "substance_quantity",
                "id": "unique_patients",
            }
        )

        return agg


class DFMConfig(BaseModel):
    # Rolling window parameters
    window_size_days: int = Field(default=365 * 2, description="Size of the rolling window in days")
    forward_window_days: int = Field(default=28, description="Size of the forward window in days")

    # Dynamic Factor Model parameters
    freq: Literal["D", "W", "2W", "M"] = Field(default="D", description="Resampling frequency")
    k_factors: int = Field(default=5, description="Number of latent factors (latentcluster of patient)")
    factor_order: int = Field(default=1, description="Order of autoregression for factors")
    error_cov_type: str = Field(default="diagonal", description="Error covariance type")
    forecast_steps: int = Field(default=0, description="Number of future steps to forecast after fit")

    # UMAP parameters
    reduce_to: int = Field(default=16, description="Number of components to reduce to")


class DFMResult(BaseModel):
    model_result: Any
    prediction: Any

    model_config = {
        "arbitrary_types_allowed": True
    }


class DynamicFactorModel(MLModelBase):
    def __init__(self, config: DFMConfig):
        self.config = config
        self.logger = PipelineLogger("DynamicFactorModel")
        self.logger.info("DynamicFactorModel initialized", {
            **self.config.model_dump()
        })

    def fit_one(self, df: pd.DataFrame) -> Tuple[StandardScaler, UMAP, sm.tsa.DynamicFactor, Any]:
        # Scale the data
        scaler = StandardScaler()
        sdf = scaler.fit_transform(df)

        # Reduce the dimensionality of the data
        reducer = UMAP(n_components=self.config.reduce_to, random_state=42)
        rdf = reducer.fit_transform(sdf)

        model = sm.tsa.DynamicFactor(
            endog=rdf,
            k_factors=self.config.k_factors,
            factor_order=self.config.factor_order,
            error_cov_type=self.config.error_cov_type,
            enforce_stationarity=False
        )
        res = model.fit()

        return scaler, reducer, model, res

    def fit(
        self, 
        df: pd.DataFrame, 
        group_by_col: str = "substance_id", 
        patient_count_col: str = "unique_patients", 
        patient_threshold: int = 1000
    ) -> Dict[str, sm.tsa.DynamicFactor]:
        """Fit Dynamic Factor Model to data."""
        feeder = RollingWindowFeeder(
            window_size_days=self.config.window_size_days, 
            forward_window_days=self.config.forward_window_days
        )
        df = df.copy()

        # Start modeling by unique substance IDs
        substances = df[group_by_col].unique().tolist()

        fitted: Dict[str, DFMResult] = {}

        for train_df, test_df, train_dts, test_dts in feeder.feed(df, "date"):
            # Refit model hyperparameters for each train-test split periods
            with mlflow.start_run(run_name=f"DFM_{train_dts[0]}_{train_dts[1]}", nested=True):
                mlflow.log_param("train_start", train_dts[0])
                mlflow.log_param("train_end", train_dts[1])
                mlflow.log_param("test_start", test_dts[0])
                mlflow.log_param("test_end", test_dts[1])
                mlflow.log_param("freq", self.config.freq)
                mlflow.log_param("k_factors", self.config.k_factors)
                mlflow.log_param("factor_order", self.config.factor_order)
                mlflow.log_param("error_cov_type", self.config.error_cov_type)
                mlflow.log_param("forecast_steps", self.config.forecast_steps)

                for sid in tqdm(substances, desc="Fitting Dynamic Factor Models"):
                    # Train data
                    self.logger.info(
                        f"Fitting Dynamic Factor Model for substance {sid} "
                        f"from {train_dts[0]} to {train_dts[1]}"
                    )

                    sub_train = train_df.loc[train_df[group_by_col] == sid].copy()
                    total_patients = sub_train[patient_count_col].sum()
                    sub_train = sub_train.set_index("date").sort_index().select_dtypes(include=[np.number])
                    full_index = pd.date_range(start=train_dts[0], end=train_dts[1], freq=self.config.freq)
                    sub_train = sub_train.reindex(full_index).fillna(0.0)

                    if total_patients < patient_threshold:
                        self.logger.warning(f"{sid}: fewer than {patient_threshold}({total_patients}) patients — skipped.")
                        continue

                    if sub_train.shape[1] < 2:
                        self.logger.warning(f"{sid}: fewer than 2 numeric series — skipped.")
                        continue

                    if self.config.k_factors >= sub_train.shape[1]:
                        self.logger.warning(f"{sid}: k_factors ≥ series count — skipped.")
                        continue

                    # Test data
                    self.logger.info(
                        f"Testing Dynamic Factor Model for substance {sid} "
                        f"from {test_dts[0]} to {test_dts[1]}"
                    )
                    sub_test = test_df.loc[test_df[group_by_col] == sid].copy()
                    sub_test = sub_test.set_index("date").sort_index().select_dtypes(include=[np.number])
                    full_index = pd.date_range(start=test_dts[0], end=test_dts[1], freq=self.config.freq)
                    sub_test = sub_test.reindex(full_index).fillna(0.0)

                    if sub_train.empty:
                        self.logger.warning(f"{sid}: no data to fit — skipped for {train_dts[0]} to {train_dts[1]}")
                        continue

                    if sub_test.empty:
                        self.logger.warning(f"{sid}: no data to test — skipped for {test_dts[0]} to {test_dts[1]}")
                        continue
                    
                    try:
                        scaler, reducer, model, res = self.fit_one(sub_train)
                        forecast_res = res.get_forecast(
                            steps=self.config.forecast_steps, 
                            exog=reducer.transform(scaler.transform(sub_test))
                        )
                        prediction = forecast_res.predicted_mean
                        fitted[sid] = DFMResult(model_result=res, prediction=prediction)

                        mlflow.log_metric("aic", res.aic)
                        mlflow.log_metric("bic", res.bic)
                        mlflow.log_metric("llf", res.llf)

                        # Save fitted model
                        artifacts_uri = mlflow.get_artifact_uri()
                        artifacts_path = Path(urlparse(artifacts_uri).path).resolve()
                        artifacts_path.mkdir(parents=True, exist_ok=True)
                        
                        safe_sid = re.sub(r'[^\w\-_\. ]', '_', sid)
                        model_path = os.path.join(artifacts_path, f"dfm_{safe_sid}.joblib")
                        
                        joblib.dump(res, model_path)
                        mlflow.log_artifact(str(model_path))
                    except Exception as e:
                        self.logger.error(f"{sid}: {e}", exc_info=True)
                        continue

        return fitted
