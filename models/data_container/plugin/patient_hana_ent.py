from io import BytesIO
from typing import Optional
from pathlib import Path
import os

import pandas as pd
import msoffcrypto

from . import prescription_detail


def _load_inpatient_data(root_dir: Path, year: Optional[int]) -> pd.DataFrame:
    """
    Load inpatient data for a specific year.

    Args:
        year (int | None): Year of data to load. If not provided use all the year

    Returns:
        A pandas DataFrame containing the outpatient data.
    ...
    """

    if year is None:
        ys = range(2018, 2024+1)  # NOTE: Update when the data is updated
    else:
        ys = [year]

    dfs = []

    for y in ys:
        print(f"Loading inpatient data for {y}...")
        fn = f"{y}.01~12_입원.xlsx"
        fp = os.path.join(root_dir, "data", "external", "hana_ent", fn)
        pw = os.getenv("DATA_PASSWORD_HANAENT")
        if pw is None or pw == "":
            raise ValueError("DATA_PASSWORD_HANAENT is not set")

        with open(fp, "rb") as f:
            office_file = msoffcrypto.OfficeFile(f)
            office_file.load_key(password=pw)
            decrypted = BytesIO()
            office_file.decrypt(decrypted)

        df = pd.read_excel(decrypted, engine='openpyxl')
        dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def _load_outpatient_data(root_dir: Path, year: Optional[int], quarter: Optional[int]) -> pd.DataFrame:
    """
    Load outpatient data for a specific year.

    Args:
        year (int | None): Year of data to load. If not provided use all the year

    Returns:
        A pandas DataFrame containing the outpatient data.
    ...
    """

    if year is None:
        ys = range(2018, 2024+1)  # NOTE: Update when the data is updated
    else:
        ys = [year]

    dfs = []

    for y in ys:
        fns = [
            ("Q1", f"{y}.01~03_외래.xlsx"),
            ("Q2", f"{y}.04~06_외래.xlsx"),
            ("Q3", f"{y}.07~09_외래.xlsx"),
            ("Q4", f"{y}.10~12_외래.xlsx"),
        ]

        for q, fn in fns:
            if quarter and f"Q{quarter}" != q:
                continue

            print(f"Loading outpatient data for {y} {q}")
            fp = os.path.join(root_dir, "data", "external", "hana_ent", fn)
            pw = os.getenv("DATA_PASSWORD_HANAENT")
            if pw is None or pw == "":
                raise ValueError("DATA_PASSWORD_HANAENT is not set")
            with open(fp, "rb") as f:
                office_file = msoffcrypto.OfficeFile(f)
                office_file.load_key(password=pw)
                decrypted = BytesIO()
                office_file.decrypt(decrypted)

            df = pd.read_excel(decrypted, engine='openpyxl')
            dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def _engineer_features_birthday(x: Optional[str]):
    if x is None:
        return None

    if len(str(x)) < 2:
        return None

    curr_year = pd.Timestamp.now().year % 100

    y = str(x)[:2]
    if int(y) > curr_year:
        return f"19{y}"
    else:
        return f"20{y}"


def _engineer_features_department(x: Optional[str]):
    if x is None or x == "":
        return None

    if x == "EN":
        return "Ear, Nose and Throat"
    elif x == "IM":
        return "Internal Medicine"
    elif x == "RA":
        return "Radiology"
    elif x == "DE":
        return "Dermatology"
    else:
        return "Unknown"


def load(root_dir: Path, year: Optional[int] = None, quarter: Optional[int] = None) -> pd.DataFrame:
    """
    Load patient data for a specific year.

    Args:
        year (int | None): Year of data to load. If not provided use all the year

    Returns:
        A pandas DataFrame containing the patient data.
    ...
    """
    tgt = ['주사약제', '수술재료료', '마취료재료']

    inpatient = _load_inpatient_data(root_dir, year)
    outpatient = _load_outpatient_data(root_dir, year, quarter)

    patient = pd.concat([inpatient, outpatient]).reset_index(drop=True)
    patient = patient.loc[patient['구분'].isin(tgt)]

    # Merge with ATC code
    atc = prescription_detail(root_dir)  # Merge key: ['제품코드']
    patient: pd.DataFrame = pd.merge(patient, atc, left_on='보험코드', right_on='제품코드')

    # Feature Engineering
    patient['날짜'] = pd.to_datetime(patient['날짜'], format='%Y-%m-%d', errors='coerce')
    patient['department'] = patient['과'].apply(_engineer_features_department)
    patient['생년월일'] = patient['생년월일'].apply(_engineer_features_birthday)
    patient['age'] = patient['날짜'].dt.year - patient['생년월일'].astype(int)
    patient['sex'] = patient['성별'].apply(lambda x: 1 if x == 'M' else 0)
    patient['prescription'] = patient['ATC코드 명칭']
    patient['id'] = patient['환자번호']
    patient['date'] = patient['날짜']
    patient = patient[['상병명', '상병구분', 'id', 'date', 'sex', 'age', 'department', 'prescription']]

    gdfs = []
    for idx, gdf in patient.groupby(by=['id', 'date']):
        if gdf.shape[0] > 1:
            diag_main = gdf.loc[gdf['상병구분'] == '주상병']['상병명'].unique().tolist()
            diag_sub = gdf.loc[gdf['상병구분'] == '부상병']['상병명'].unique().tolist()

            diag_main_str = ", ".join(diag_main)
            diag_sub_str = ", ".join(diag_sub)
            gdf['primary_diagnosis'] = diag_main_str
            gdf['secondary_diagnosis'] = diag_sub_str
            gdfs.append(gdf)

    patient = pd.concat(gdfs).reset_index(drop=True)
    patient = patient[
        ['id', 'date', 'sex', 'age', 'primary_diagnosis', 'secondary_diagnosis', 'prescription', 'department']
    ]

    patient['id'] = patient['id'].astype(int)
    patient['age'] = patient['age'].astype(int)
    patient['primary_diagnosis'] = patient['primary_diagnosis'].str.strip()
    patient['secondary_diagnosis'] = patient['secondary_diagnosis'].str.strip()
    patient['department'] = patient['department'].str.strip()

    return patient


def load_with_code(root_dir: Path, year: Optional[int] = None, quarter: Optional[int] = None) -> pd.DataFrame:
    """
    Load patient data for a specific year.

    Args:
        year (int | None): Year of data to load. If not provided use all the year

    Returns:
        A pandas DataFrame containing the patient data.
    ...
    """
    tgt = ['주사약제', '수술재료료', '마취료재료']

    inpatient = _load_inpatient_data(root_dir, year)
    outpatient = _load_outpatient_data(root_dir, year, quarter)

    patient = pd.concat([inpatient, outpatient]).reset_index(drop=True)
    patient = patient.loc[patient['구분'].isin(tgt)]

    # Merge with ATC code
    atc = prescription_detail(root_dir)  # Merge key: ['제품코드']
    patient: pd.DataFrame = pd.merge(patient, atc, left_on='보험코드', right_on='제품코드')

    # Feature Engineering
    patient['날짜'] = pd.to_datetime(patient['날짜'], format='%Y-%m-%d', errors='coerce')
    patient['department'] = patient['과'].apply(_engineer_features_department)
    patient['생년월일'] = patient['생년월일'].apply(_engineer_features_birthday)
    patient['age'] = patient['날짜'].dt.year - patient['생년월일'].astype(int)
    patient['sex'] = patient['성별'].apply(lambda x: 1 if x == 'M' else 0)
    patient['prescription_code'] = patient['ATC코드']
    patient['id'] = patient['환자번호']
    patient['date'] = patient['날짜']
    patient = patient[['상병명', '상병구분', 'id', 'date', 'sex', 'age', 'department', 'prescription_code']]

    gdfs = []
    for idx, gdf in patient.groupby(by=['id', 'date']):
        if gdf.shape[0] > 1:
            diag_main = gdf.loc[gdf['상병구분'] == '주상병']['상병명'].unique().tolist()
            diag_sub = gdf.loc[gdf['상병구분'] == '부상병']['상병명'].unique().tolist()

            diag_main_str = ", ".join(diag_main)
            diag_sub_str = ", ".join(diag_sub)
            gdf['primary_diagnosis'] = diag_main_str
            gdf['secondary_diagnosis'] = diag_sub_str
            gdfs.append(gdf)

    patient = pd.concat(gdfs).reset_index(drop=True)
    patient = patient[
        ['id', 'date', 'sex', 'age', 'primary_diagnosis', 'secondary_diagnosis', 'prescription_code', 'department']
    ]

    patient['id'] = patient['id'].astype(int)
    patient['age'] = patient['age'].astype(int)
    patient['primary_diagnosis'] = patient['primary_diagnosis'].str.strip()
    patient['secondary_diagnosis'] = patient['secondary_diagnosis'].str.strip()
    patient['department'] = patient['department'].str.strip()

    return patient


def load_with_diag_code(root_dir: Path, year: Optional[int] = None, quarter: Optional[int] = None) -> pd.DataFrame:
    """
    Load patient data for a specific year.

    Args:
        year (int | None): Year of data to load. If not provided use all the year

    Returns:
        A pandas DataFrame containing the patient data.
    ...
    """
    tgt = ['주사약제', '수술재료료', '마취료재료']

    inpatient = _load_inpatient_data(root_dir, year)
    outpatient = _load_outpatient_data(root_dir, year, quarter)

    patient = pd.concat([inpatient, outpatient]).reset_index(drop=True)
    patient = patient.loc[patient['구분'].isin(tgt)]

    # Merge with ATC code
    atc = prescription_detail(root_dir)  # Merge key: ['제품코드']
    patient: pd.DataFrame = pd.merge(patient, atc, left_on='보험코드', right_on='제품코드')

    # Feature Engineering
    patient['날짜'] = pd.to_datetime(patient['날짜'], format='%Y-%m-%d', errors='coerce')
    patient['department'] = patient['과'].apply(_engineer_features_department)
    patient['생년월일'] = patient['생년월일'].apply(_engineer_features_birthday)
    patient['age'] = patient['날짜'].dt.year - patient['생년월일'].astype(int)
    patient['sex'] = patient['성별'].apply(lambda x: 1 if x == 'M' else 0)
    patient['prescription_code'] = patient['ATC코드']
    patient['id'] = patient['환자번호']
    patient['date'] = patient['날짜']
    patient = patient[['상병코드', '상병명', '상병구분', 'id', 'date', 'sex', 'age', 'department', 'prescription_code']]

    gdfs = []
    for idx, gdf in patient.groupby(by=['id', 'date']):
        if gdf.shape[0] > 1:
            diagnosis = gdf['상병코드']
            diagnosis = diagnosis.unique().tolist()
            diagnosis_str = "|".join(map(lambda x: str(x).strip(), diagnosis))
            gdf['diagnosis'] = diagnosis_str
            gdfs.append(gdf)

    patient = pd.concat(gdfs).reset_index(drop=True)
    patient = patient[
        ['id', 'date', 'sex', 'age', 'diagnosis', 'prescription_code', 'department']
    ]

    patient['id'] = patient['id'].astype(int)
    patient['age'] = patient['age'].astype(int)
    patient['diagnosis'] = patient['diagnosis'].str.strip()
    patient['department'] = patient['department'].str.strip()

    return patient