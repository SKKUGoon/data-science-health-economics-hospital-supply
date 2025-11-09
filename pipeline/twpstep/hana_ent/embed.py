import warnings
import os
import logging

root_dir = os.path.abspath(".")

# Ignore some warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)  # Ignore some warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden to 1 by setting random_state.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Use no seed for parallelism.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Use no seed for.*", category=UserWarning)  # Ignore some warnings

# Configure logging to only show WARNING and ERROR messages
logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

# Custom packages
from utils.auth.hospital_profile import HospitalProfile
from models.embed.chart_element_embedding import ChartElementEmbedding
from models.data_container.plugin.patient_hana_ent import load

# Generate the hospital profile for epurun
print("[Run] Load data ...")
origin = load(root_dir)

print("[Run] Create hospital profile (hana_ent)")
hosp = HospitalProfile(hospital_id="hana_ent_v2", hospital_name="하나이비인후과병원")
emb = ChartElementEmbedding(profile=hosp)
emb.initialize_qdrant(
    host=os.getenv("QDRANT_HOST"),
    port=os.getenv("QDRANT_PORT")
)

print("[Run] Creating Embedding")
emb.create_element_document(origin)