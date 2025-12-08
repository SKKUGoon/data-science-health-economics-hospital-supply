from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.api import VAR  # For factor forecasting
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from typing import List, Tuple, Literal, Optional

class DFMConfig(BaseModel):
    n_factors: int = Field(..., description="Number of latent factors")
    factor_order: int = Field(1, description="Factor VAR order (for VAR, not DFM)")
    var_clipping: bool = Field(False, description="Clip columns with almost no variance")


class RFConfig(BaseModel):
    target_columns: Optional[List[str]] = Field(None, description="Supply usage columns. If None, uses all columns.")
    n_estimators: int = Field(100, description="Number of trees in Random Forest")
    random_state: int = Field(42, description="Random state for reproducibility")


class RollingConfig(BaseModel):
    window_type: Literal["rolling", "expanding"] = Field(..., description="Window mode")
    window_size: int = Field(..., description="Number of time points per rolling window")
    forecast_horizon: int = Field(..., description="Steps ahead")
    min_train_size: int = Field(50, description="Minimum size for first window")


def prepare_dfm_single_window(X: pd.DataFrame, min_var: float = 1e-8, var_clipping: bool = False) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Per-window preprocessing for DynamicFactor:
    - Drop constant / near-constant columns
    - Standardize (z-score)
    """
    # Drop columns with almost no variance
    if var_clipping:
        var = X.var()
        keep_cols = var[var > min_var].index
        print(f"[DEBUG] Keeping {len(keep_cols)}/{len(X.columns)} columns")
        X = X[keep_cols]

    # Standardize
    scaler = StandardScaler()
    X_vals = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_vals, index=X.index, columns=X.columns)

    return X_scaled, scaler


def fit_dfm(X: pd.DataFrame, cfg: DFMConfig):
    """
    Fit DynamicFactor on standardized weekly data.
    Returns factors as a pandas DataFrame with shape (T_window, n_factors).
    """

    model = DynamicFactor(
        X,
        k_factors=cfg.n_factors,
        factor_order=1,
        error_cov_type="scalar"
    )

    result = model.fit(disp=False)

    raw = result.factors.filtered  # may be ndarray or DataFrame
    T = len(X)
    k = cfg.n_factors

    # --- Case 1: ndarray ---
    if isinstance(raw, np.ndarray):

        # Shape may be (T, k) or (k, T)
        if raw.shape == (T, k):
            data = raw
        elif raw.shape == (k, T):
            data = raw.T
        else:
            raise ValueError(f"Unexpected factor shape: {raw.shape}")

        factor_df = pd.DataFrame(
            data,
            index=X.index,
            columns=[f"F{i+1}" for i in range(k)]
        )
        return factor_df, result

    # --- Case 2: DataFrame ---
    if isinstance(raw, pd.DataFrame):
        # Ensure correct orientation
        if raw.shape == (T, k):
            factor_df = raw.copy()
            factor_df.index = X.index
            factor_df.columns = [f"F{i+1}" for i in range(k)]
            return factor_df, result

        if raw.shape == (k, T):
            factor_df = raw.T.copy()
            factor_df.index = X.index
            factor_df.columns = [f"F{i+1}" for i in range(k)]
            return factor_df, result

        raise ValueError(f"Unexpected DataFrame factor shape: {raw.shape}")

    # Should not reach here
    raise ValueError("Unrecognized factor result type.")

def fit_supply_models(factors: pd.DataFrame,
                      supply_df: pd.DataFrame,
                      reg_cfg: RFConfig):
    """
    Fits RandomForestRegressor for each supply column using factors as features.
    """
    models = {}
    X = factors.values
    for col in reg_cfg.target_columns:
        print(f"[DEBUG] Fitting RandomForest model for {col}")
        y = supply_df[col].loc[factors.index]
        model = RandomForestRegressor(
            n_estimators=reg_cfg.n_estimators,
            random_state=reg_cfg.random_state
        )
        model.fit(X, y)
        models[col] = model
    return models


def forecast_factors_var(factors: pd.DataFrame,
                         steps: int,
                         lags: int) -> pd.DataFrame:
    """
    Forecast factors via VAR. This is the 'dynamic' part of method 1.1.
    """
    var_model = VAR(factors)
    var_res = var_model.fit(maxlags=lags)

    fc_values = var_res.forecast(y=factors.values[-lags:], steps=steps)

    freq = factors.index.inferred_freq or "W"
    start = factors.index[-1] + pd.tseries.frequencies.to_offset(freq)
    idx = pd.date_range(start, periods=steps, freq=freq)

    return pd.DataFrame(fc_values, index=idx, columns=factors.columns)


def predict_supplies(factor_fc: pd.DataFrame,
                     models: dict,
                     reg_cfg: RFConfig) -> pd.DataFrame:
    Xf = factor_fc.values
    out = {}
    for col in reg_cfg.target_columns:
        out[col] = models[col].predict(Xf)
    return pd.DataFrame(out, index=factor_fc.index)


def rolling_method_dfm(
    dfm_data: pd.DataFrame,
    supply_data: pd.DataFrame,
    dfm_cfg: DFMConfig,
    reg_cfg: RFConfig,
    roll_cfg: RollingConfig,
    n_windows: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Align indices
    common_idx = dfm_data.index.intersection(supply_data.index)
    dfm_data = dfm_data.loc[common_idx]
    supply_data = supply_data.loc[common_idx]

    if not reg_cfg.target_columns:
        reg_cfg.target_columns = list(supply_data.columns)

    preds = []
    factor_preds = []
    dates = dfm_data.index

    # Select window end-points using a rolling/sliding strategy
    # If window_type is 'rolling', we slide by `forecast_horizon` steps.
    # Start: window_size
    # End: len(dates) - forecast_horizon + 1 (exclusive range bound)
    # Step: forecast_horizon
    
    start_point = roll_cfg.window_size
    if roll_cfg.window_type == "expanding":
         start_point = roll_cfg.min_train_size

    ends = range(start_point, 
                 len(dates) - roll_cfg.forecast_horizon + 1, 
                 roll_cfg.forecast_horizon)

    for end_idx in ends:
        print(f"[DEBUG] Fitting window {end_idx}")
        # Compute window positions
        if roll_cfg.window_type == "rolling":
            start_idx = max(0, end_idx - roll_cfg.window_size)
        else:
            start_idx = 0

        wdates = dates[start_idx:end_idx]
        Xw_raw = dfm_data.loc[wdates]
        Sw = supply_data.loc[wdates]

        # Prepare window
        Xw, _ = prepare_dfm_single_window(Xw_raw, var_clipping=dfm_cfg.var_clipping)

        if Xw.shape[1] < dfm_cfg.n_factors:
            continue
        if Xw.isna().any().any() or Sw.isna().any().any():
            continue

        # Fit models
        print(f"[Debug] Xw shape: {Xw.shape} | Sw shape: {Sw.shape}")
        factors, dfm_result = fit_dfm(Xw, dfm_cfg)
        models = fit_supply_models(factors, Sw, reg_cfg)
        factor_fc = forecast_factors_var(
            factors,
            steps=roll_cfg.forecast_horizon,
            lags=dfm_cfg.factor_order
        )
        supply_fc = predict_supplies(factor_fc, models, reg_cfg)

        preds.append(supply_fc)
        factor_preds.append(factor_fc)

    if len(preds) == 0:
        raise ValueError("No windows produced valid forecasts.")

    preds = pd.concat(preds).sort_index()
    preds = preds[~preds.index.duplicated(keep="last")]

    factor_preds_df = pd.concat(factor_preds).sort_index()
    factor_preds_df = factor_preds_df[~factor_preds_df.index.duplicated(keep="last")]

    return preds, factor_preds_df


def compare_pred_with_actual(
    pred: pd.DataFrame,
    supply_true: pd.DataFrame,
    target_cols: List[str]
) -> pd.DataFrame:
    """
    Align predicted DataFrame with true supply values
    and compute error metrics.
    """

    # Extract true values for the prediction dates
    true_aligned = supply_true.loc[pred.index]

    comparisons = []

    for col in target_cols:
        df_col = pd.DataFrame({
            "pred": pred[col],
            "true": true_aligned[col]
        })
        df_col["error"] = df_col["pred"] - df_col["true"]
        df_col["abs_error"] = df_col["error"].abs()
        df_col["sq_error"] = df_col["error"] ** 2

        df_col.columns = pd.MultiIndex.from_product([[col], df_col.columns])
        comparisons.append(df_col)

    return pd.concat(comparisons, axis=1)

