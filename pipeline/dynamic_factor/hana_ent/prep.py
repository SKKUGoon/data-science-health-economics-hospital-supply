import os
root_dir = os.path.abspath(".")

import pandas as pd
import numpy as np

from models.data_container.plugin.patient_hana_ent import load

print("[Run] Load data ...")
df = load(root_dir)

print("[Run] Engineering features ...")

# Encode sex as binary
df['male'] = (df['sex'] == 1).astype(int)
df['female'] = (df['sex'] == 0).astype(int)

# Bin ages
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]
labels = ['age_0_10','age_10_20','age_20_30','age_30_40','age_40_50',
          'age_50_60','age_60_70','age_70_80','age_80_plus']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Trim ATC code to first 3 chars
df['ATC3'] = df['prescription_code'].str[:3]

# One hot encode department, age_group
dep_dummies = pd.get_dummies(df['department'], prefix='dep')
age_dummies = pd.get_dummies(df['age_group'])

# Combine
df_features = pd.concat(
    [
        df[['date', 'male', 'female', 'ATC3']], 
        age_dummies, 
        dep_dummies
    ], 
    axis=1
)

df_atc = df_features.groupby(['date','ATC3']).size().unstack(fill_value=0)
df_demo = df_features.groupby('date')[['male','female'] + list(age_dummies.columns) + list(dep_dummies.columns)].sum()
df_daily = df_demo.join(df_atc, how='outer').fillna(0)
df_daily = df_daily.sort_index()

df_daily.to_parquet(os.path.join(root_dir, "data/processed/hana_ent/dfm_daily.parquet"))
