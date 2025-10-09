from utils.auth.hospital_profile import HospitalProfile
from models.embed.openai_embedding import PatientChartEmbedding
from models.container.plugin.patient_epurun import load

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    root_dir = os.path.abspath(".")

    # Generate the hospital profile for epurun
    print("[Run] Create hospital profile (epurun)")
    hosp_epurun = HospitalProfile(hospital_id="epurun", hospital_name="이푸른병원")

    print("[Run] Load patient data")
    df = load(root_dir)

    print("[Run] Create embedding ...")
    emb = PatientChartEmbedding(profile=hosp_epurun)
    emb.initialize_qdrant(
        host=os.getenv("QDRANT_HOST"),
        port=os.getenv("QDRANT_PORT")
    )
    emb.create_patient_document_by_date(df)
    print("[Run] Embedding completed")
