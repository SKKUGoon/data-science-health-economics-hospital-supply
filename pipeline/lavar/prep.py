import os
root_dir = os.path.abspath(".")

import pandas as pd
import numpy as np

from models.data_container.plugin.patient_hana_ent import load_with_diag_code
from models.data_container.kcd_code import kcd_code_range_key

df = load_with_diag_code(root_dir)
df['male'] = (df['sex'] == 1).astype(int)
df['female'] = (df['sex'] == 0).astype(int) 

# Bin ages
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
labels = ['age_0_10','age_10_20','age_20_30','age_30_40','age_40_50',
          'age_50_60','age_60_70','age_70_80','age_80_plus']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Trim ATC code to first 5 chars
# ATC5 is the 5-character code for the prescription
df['ATC5'] = df['prescription_code'].str[:5]

# Get diagnosis codes and bucket them into KCD ranges
df_diag = (
    df['diagnosis'].str.split("|")
    .explode()
    .apply(kcd_code_range_key)
    .groupby(level=0)
    .value_counts()
    .unstack(fill_value=0)
)

# One hot encode department, age_group
dep_dummies = pd.get_dummies(df['department'], prefix='dep')
age_dummies = pd.get_dummies(df['age_group'])

# Combine
df_features = pd.concat(
    [
        df[['date', 'male', 'female', 'ATC5']], 
        df_diag,
        age_dummies, 
        dep_dummies
    ], 
    axis=1
)

# Result of the prediction - Supply
df_atc = df_features.groupby(['date','ATC5']).size().unstack(fill_value=0)
df_dg = df_features.groupby('date')[[c for c in df_features.columns if c.startswith('KCD')]].sum().fillna(0)
df_demo = df_features.groupby('date')[['male', 'female'] + list(age_dummies.columns) + list(dep_dummies.columns)].sum()


df_daily1 = df_demo.join(df_dg, how='outer').fillna(0)  # Merge the demographics dataframe with the diagnosis code bucket dataframe
df_daily2 = df_demo.join(df_atc, how='outer').fillna(0)  # Merge the demographics dataframe with the ATC code (:5) dataframe
df_daily1 = df_daily1.sort_index()
df_daily2 = df_daily2.sort_index()


# Add the calendar features
df_daily1['weekday'] = df_daily1.index.weekday  # Monday is 0, Sunday is 6
df_daily1['week_of_year'] = df_daily1.index.isocalendar().week.astype(int)

h18 = pd.read_csv(os.path.join(root_dir, "data/external/holidays/2018.csv"))
h19 = pd.read_csv(os.path.join(root_dir, "data/external/holidays/2019.csv"))
h20 = pd.read_csv(os.path.join(root_dir, "data/external/holidays/2020.csv"))
h21 = pd.read_csv(os.path.join(root_dir, "data/external/holidays/2021.csv"))
h22 = pd.read_csv(os.path.join(root_dir, "data/external/holidays/2022.csv"))
h23 = pd.read_csv(os.path.join(root_dir, "data/external/holidays/2023.csv"))
h24 = pd.read_csv(os.path.join(root_dir, "data/external/holidays/2024.csv"))
hs = pd.concat([h18, h19, h20, h21, h22, h23, h24])
hs['Start date'] = pd.to_datetime(hs['Start date'])

def next_weekday(d: pd.Timestamp) -> pd.Timestamp:
    d = d + pd.Timedelta(days=1)
    while d.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        d += pd.Timedelta(days=1)
    return d

day_after_holiday = hs['Start date'].apply(next_weekday)

df_daily1['day_after_holiday'] = df_daily1.index.isin(day_after_holiday).astype(int)
df_daily2['day_after_holiday'] = df_daily2.index.isin(day_after_holiday).astype(int)

df_daily1.to_parquet(os.path.join(root_dir, "data/processed/hana_ent/lavar_daily1.parquet"))
df_daily2.to_parquet(os.path.join(root_dir, "data/processed/hana_ent/lavar_daily2.parquet"))
