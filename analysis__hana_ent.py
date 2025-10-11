from core.config import PipelineConfig
from utils.auth.hospital_profile import HospitalProfile
from models.embed.openai_embedding import PatientChartEmbedding
from models.data_container.plugin.patient_epurun import load

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    root_dir = os.path.abspath(".")

    # Generate the hospital profile for hana ent
    print("[Run] Create hospital profile (hana_ent)")
    hosp_hana = HospitalProfile(hospital_id="hana_ent", hospital_name="하나이비인후과병원")
    config = PipelineConfig()

    print("[Run] Get embedding ...")
    emb = PatientChartEmbedding(config=config, profile=hosp_hana)
    emb.initialize_qdrant(
        host=os.getenv("QDRANT_HOST"),
        port=os.getenv("QDRANT_PORT")
    )
    meta, vector = emb.retrieve_embedding()

    print(meta.shape)
    print(vector.shape)
    print(meta.head())
    print(vector.head())