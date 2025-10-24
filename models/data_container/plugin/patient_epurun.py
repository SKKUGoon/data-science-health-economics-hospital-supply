from pathlib import Path
import os
import pandas as pd
from . import prescription_detail

def load(root_dir: Path):
    fn = "patient_epurun_20251009_011300.csv"
    fp = os.path.join(root_dir, "data", "external", "epurun", fn)

    atc = prescription_detail(root_dir)  # Merge key: ['제품코드']
    atc['제품코드'] = atc['제품코드'].map(str)

    patient: pd.DataFrame = pd.read_csv(fp, low_memory=False)
    patient = patient.loc[patient['단가'] > 0]
    patient = patient.loc[
        ~patient['청구코드'].str.strip().str.match(r'^[a-zA-Z]', na=False)
    ]
    patient = patient.loc[
        ~patient['청구코드'].str.contains(r'앞치마|에어메트1|에어메트2|치매장갑', na=False)
    ]
    patient = patient.loc[patient['date'] != 20000000]  # 잘못된 데이터. 사용 불가

    # Data cleaning - Change type
    patient['입원일'] = pd.to_datetime(patient['입원일'], format='%m/%d/%Y')
    patient['퇴원일'] = pd.to_datetime(patient['퇴원일'], format='%m/%d/%Y')
    patient['visit_date'] = pd.to_datetime(patient['date'].map(str), format='%Y%m%d')
    patient['date'] = patient['입원일']  # Match it with patient_hana_ent data. Unique patient (per visit) is based on 입원일
    patient = patient.loc[
        ~((patient['입원일'] != patient['입원일']) & (patient['퇴원일'] != patient['퇴원일']))
    ]
    patient = patient.loc[
        ~((patient['입원일'] > patient['date']) | (patient['퇴원일'] < patient['date']))
    ]
    patient['sex'] = patient['male'].apply(lambda x: 1 if x == 1 else 0)
    patient: pd.DataFrame = pd.merge(patient, atc, left_on='청구코드', right_on='제품코드')

    patient = patient.rename(
        {
            "차트번호": "id",
            "kcd1_e": "primary_diagnosis",
            "kcd2_e": "secondary_diagnosis",
            'ATC코드 명칭': 'prescription',
            "q_per_c": "quantity",
        },
        axis=1
    )

    patient = patient[
        [
            'id', 'date',  # key columns
            'visit_date', 'sex', 'age', 'primary_diagnosis', 'secondary_diagnosis', 'prescription',  # Embedding columns
            "quantity",  # Supply estimation - per prescription
        ]
    ]
    patient['department'] = 'Unspecified'

    patient['id'] = patient['id'].astype(int)
    patient['age'] = patient['age'].astype(int)
    patient['primary_diagnosis'] = patient['primary_diagnosis'].str.strip()
    patient['secondary_diagnosis'] = patient['secondary_diagnosis'].str.strip()

    return patient