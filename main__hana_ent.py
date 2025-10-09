from utils.auth.hospital_profile import HospitalProfile
from models.embed.openai_embedding import PatientChartEmbedding
from models.container.plugin.patient_hana_ent import load

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    root_dir = os.path.abspath(".")

    # Generate the hospital profile for hana ent
    print("[Run] Create hospital profile (hana_ent)")
    hosp_hana = HospitalProfile(hospital_id="hana_ent", hospital_name="하나이비인후과병원")

    print("[Run] Load patient data")
    df = load(root_dir)  # load all the data

    print("[Run] Create embedding ...")
    emb = PatientChartEmbedding(profile=hosp_hana)
    emb.initialize_qdrant(
        host=os.getenv("QDRANT_HOST"),
        port=os.getenv("QDRANT_PORT")
    )
    emb.create_patient_document_by_date(df, resume_from=57_000)
    print("[Run] Embedding completed")
