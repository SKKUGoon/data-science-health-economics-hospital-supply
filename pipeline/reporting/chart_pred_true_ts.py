import pandas as pd
import numpy as np

def compute_mse_per_target(df: pd.DataFrame) -> pd.Series:
    """
    Compute MSE for each target (top-level column) where columns are a MultiIndex
    structured like (target, ['pred', 'true', ...]).
    """

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Input DataFrame must have MultiIndex columns.")

    # We assume exactly two columns per target used for calculation: pred & true.
    # We group by the level=0 (the target names).
    mses = {}
    for target, subdf in df.groupby(level=0, axis=1):
        # Safely pick only pred and true
        y_true = subdf.get((target, "true"))
        y_pred = subdf.get((target, "pred"))
        if y_true is None or y_pred is None:
            continue  # skip if one is missing

        mse_val = np.mean((y_true - y_pred)**2)
        mses[target] = mse_val

    return pd.Series(mses, name="mse")