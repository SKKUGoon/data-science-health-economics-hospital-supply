import warnings
import os
import logging

root_dir = os.path.abspath(".")

# Ignore some warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)  # Ignore some warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden to 1 by setting random_state.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Use no seed for parallelism.*", category=UserWarning)

# Configure logging to only show WARNING and ERROR messages
logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

# Custom packages
from utils.auth.hospital_profile import HospitalProfile
from models.embed.openai_embedding import PatientChartEmbedding
from models.data_container.plugin.patient_epurun import load
from pipeline.base_manager import MLPipeline
from pipeline.dfm_pipeline import DFMTransformer, DFMConfig, DynamicFactorModel

# Generate the hospital profile for epurun
print("[Run] Create hospital profile (epurun)")
hosp = HospitalProfile(hospital_id="epurun", hospital_name="이푸른병원")

print("[Run] Load data ...")
origin = load(root_dir)

print("[Run] Get embedding ...")
emb = PatientChartEmbedding(profile=hosp)
emb.initialize_qdrant(
    host=os.getenv("QDRANT_HOST"),
    port=os.getenv("QDRANT_PORT")
)
embed = emb.retrieve_embedding()

print("[Run] Run the pipeline")
model_config = DFMConfig(
    window_size_days=365 * 3,
    forward_window_days=14,
    freq="W",  # Weekly prediction
    k_factors=5,  # Latent factors
    factor_order=1,  # AR
    error_cov_type="diagonal",
    forecast_steps=1,
    reduce_to=10
)
pipe = MLPipeline(
    profile=hosp,
    original=origin,
    embedding=embed.mergeable_embeddings(),
    transformer=DFMTransformer(),
    model=DynamicFactorModel(config=model_config),
    evaluator=None,
)
f = pipe.run("test-epurun-1")