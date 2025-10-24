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

import pandas as pd
import umap

# Custom packages
from utils.auth.hospital_profile import HospitalProfile
from models.embed.chart_element_embedding import ChartElementEmbedding
from models.data_container.plugin.patient_hana_ent import load

# Generate the hospital profile for epurun
# print("[Run] Load data ...")
# origin = load(root_dir)

print("[Run] Create hospital profile (hana_ent)")
hosp = HospitalProfile(hospital_id="hana_ent_v2", hospital_name="하나이비인후과병원")
emb = ChartElementEmbedding(profile=hosp)
emb.initialize_qdrant(
    host=os.getenv("QDRANT_HOST"),
    port=os.getenv("QDRANT_PORT")
)

# print("[Run] Creating Embedding")
# emb.create_element_document(origin)

print("[Run] Get embedding ...")
embed = emb.retrieve_embedding()

print("[Run] Running UMAP ...")
umap_2d = umap.UMAP(n_neighbors=20, n_components=2, metric='cosine', random_state=42)
umap_3d = umap.UMAP(n_neighbors=20, n_components=3, metric='cosine', random_state=42)

f2_dep = pd.DataFrame(umap_2d.fit_transform(embed['vector_department']), index=embed['idx_department'], columns=[f"umap{i+1}" for i in range(2)])
f3_dep = pd.DataFrame(umap_3d.fit_transform(embed['vector_department']), index=embed['idx_department'], columns=[f"umap{i+1}" for i in range(3)])
f2_diag = pd.DataFrame(umap_2d.fit_transform(embed['vector_diagnosis']), index=embed['idx_diagnosis'], columns=[f"umap{i+1}" for i in range(2)])
f3_diag = pd.DataFrame(umap_3d.fit_transform(embed['vector_diagnosis']), index=embed['idx_diagnosis'], columns=[f"umap{i+1}" for i in range(3)])
f2_pres = pd.DataFrame(umap_2d.fit_transform(embed['vector_prescription']), index=embed['idx_prescription'], columns=[f"umap{i+1}" for i in range(2)])
f3_pres = pd.DataFrame(umap_3d.fit_transform(embed['vector_prescription']), index=embed['idx_prescription'], columns=[f"umap{i+1}" for i in range(3)])

# Save the data to parquet files. data/processed/hana_ent
print("[Run] Saving vector data to parquet files ...")
f2_dep.to_parquet("data/processed/hana_ent/f2_dep.parquet")
f3_dep.to_parquet("data/processed/hana_ent/f3_dep.parquet")
f2_diag.to_parquet("data/processed/hana_ent/f2_diag.parquet")
f3_diag.to_parquet("data/processed/hana_ent/f3_diag.parquet")
f2_pres.to_parquet("data/processed/hana_ent/f2_pres.parquet")
f3_pres.to_parquet("data/processed/hana_ent/f3_pres.parquet")

print("[Run] Saving metadata to parquet files ...")
embed['meta_department'].to_parquet("data/processed/hana_ent/meta_dep.parquet")
embed['meta_diagnosis'].to_parquet("data/processed/hana_ent/meta_diag.parquet")
embed['meta_prescription'].to_parquet("data/processed/hana_ent/meta_pres.parquet")
