import warnings

import pandas as pd
import mlflow
from tqdm import tqdm

from core.config import PipelineConfig

from utils.auth.hospital_profile import HospitalProfile

from models.embed.openai_embedding import PatientChartEmbedding
from models.data_container.plugin.patient_epurun import load
from models.data_container.expanding_window_feeder import ExpandingWindowFeeder

from pipeline.backtesting_experiment import BacktestingExperiment

# Ignore some warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    root_dir = os.path.abspath(".")

    # Generate the hospital profile for epurun
    print("[Run] Create hospital profile (epurun)")
    hosp_epurun = HospitalProfile(hospital_id="epurun", hospital_name="이푸른병원")
    config = PipelineConfig(
        # MLFlow configuration
        experiment_name="epurun-experiments",
        model_registry_prefix=hosp_epurun.hospital_id,

        # Clustering configuration
        noise_threshold=0.5,
    )

    with mlflow.start_run(run_name="epurun") as run:
        print("[Run] Get embedding ...")
        emb = PatientChartEmbedding(config=config, profile=hosp_epurun,)
        emb.initialize_qdrant(
            host=os.getenv("QDRANT_HOST"),
            port=os.getenv("QDRANT_PORT")
        )
        meta, vector = emb.retrieve_embedding()
        backtest = meta.join(vector)  # has the same index (doc_id)

        print("[Run] Start experimentation")
        summaries = BacktestingExperiment(
            config=config,
            feeder=ExpandingWindowFeeder(window_size_days=365*4, forward_window_days=14),

        ).run(
            data=backtest,
            datetime_col='date',
            feature_cols=vector.columns,
            run_name='backtesting-experiment',
        )

        mtrc = []
        assignments = []
        for s in summaries:
            print(s.train_asssignments.head())
            print(s.train_asssignments.tail())
            print("-" * 80)
            print(s.test_assignments.head())
            print(s.test_assignments.tail())
