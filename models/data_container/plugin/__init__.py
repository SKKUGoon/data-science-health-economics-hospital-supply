import pandas as pd
from pathlib import Path
import os


def prescription_detail(root_dir: Path):
    print("Loading ATC code map. Last Update 2025/09/12")

    fp = os.path.join(root_dir, "data", "external", "ATC_20250912_110516.csv")
    df = pd.read_csv(fp, encoding='cp949')
    return df