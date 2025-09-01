import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
from typing import Union, Tuple, Optional, Literal

def filter_cells(
    madata: ad.AnnData,
    min_n_metabolites: Optional[int] = None,
    max_n_metabolites: Optional[int] = None,
    min_total_intensity: Optional[float] = None,
    max_total_intensity: Optional[float] = None,
    mean_only_detected: bool = False
) -> ad.AnnData:
    """
    Filter cells based on QC metrics and update feature-level QC metrics.

    Parameters
    ----------
    madata : ad.AnnData
        Input AnnData object. Should have 'n_metabolites' and 'total_intensity' in .obs
    min_n_metabolites, max_n_metabolites : int, optional
        Filtering thresholds on n_metabolites
    min_total_intensity, max_total_intensity : float, optional
        Filtering thresholds on total_intensity

    Returns
    -------
    ad.AnnData
        Filtered AnnData object with updated QC metrics.
    """
    mask = np.ones(madata.n_obs, dtype=bool)
    if min_n_metabolites is not None:
        mask &= madata.obs['n_metabolites'] >= min_n_metabolites
    if max_n_metabolites is not None:
        mask &= madata.obs['n_metabolites'] <= max_n_metabolites
    if min_total_intensity is not None:
        mask &= madata.obs['total_intensity'] >= min_total_intensity
    if max_total_intensity is not None:
        mask &= madata.obs['total_intensity'] <= max_total_intensity
    new_madata = madata[mask].copy()

    # ---- 更新Feature-level QC (var) ----
    X = new_madata.X
    n_cells = np.sum((X > 0) & ~np.isnan(X), axis=0)
    detect_rate = n_cells / np.sum(~np.isnan(X), axis=0) * 100
    if mean_only_detected:
        # 避免除以 0
        mean_intensity = np.array([
            np.nanmean(col[col > 0]) if np.any(col > 0) else 0.0
            for col in X.T
        ])
    else:
        mean_intensity = np.nanmean(X, axis=0)

    new_madata.var["hits"] = np.array(n_cells).ravel()
    new_madata.var["hit_rate"] = np.array(detect_rate).ravel()
    new_madata.var["mean_intensity"] = np.array(mean_intensity).ravel()

    return new_madata

def filter_metabolites(
    madata: ad.AnnData,
    min_hits: Optional[int] = None,
    max_hits: Optional[int] = None,
    min_hit_rate: Optional[float] = None,
    max_hit_rate: Optional[float] = None,
    min_mean_intensity: Optional[float] = None,
    max_mean_intensity: Optional[float] = None,
    min_mean_intensity_detected: Optional[float] = None,
    max_mean_intensity_detected: Optional[float] = None,
    extra_filters: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None
) -> ad.AnnData:
    """
    Filter metabolites (features) based on QC metrics and update cell-level QC metrics.

    Parameters
    ----------
    madata : ad.AnnData
        Input AnnData object. Should have 'hits', 'hit_rate', 'mean_intensity' in .var
    min_hits, max_hits : int, optional
    min_hit_rate, max_hit_rate : float, optional
    min_mean_intensity, max_mean_intensity : float, optional
    extra_filters : dict[str, tuple[Optional[float], Optional[float]]], optional
        Arbitrary column filters. Example:
        {
            "my_metric": (0.1, 0.9),   # keep rows with 0.1 <= my_metric <= 0.9
            "another_metric": (None, 100)  # only upper bound
        }
    Returns
    -------
    ad.AnnData
        Filtered AnnData object with updated QC metrics.
    """
    mask = np.ones(madata.n_vars, dtype=bool)
    if min_hits is not None:
        mask &= madata.var['hits'] >= min_hits
    if max_hits is not None:
        mask &= madata.var['hits'] <= max_hits
    if min_hit_rate is not None:
        mask &= madata.var['hit_rate'] >= min_hit_rate
    if max_hit_rate is not None:
        mask &= madata.var['hit_rate'] <= max_hit_rate
    if min_mean_intensity is not None:
        mask &= madata.var['mean_intensity'] >= min_mean_intensity
    if max_mean_intensity is not None:
        mask &= madata.var['mean_intensity'] <= max_mean_intensity
    if min_mean_intensity_detected is not None:
        mask &= madata.var['mean_intensity_detected'] >= min_mean_intensity_detected
    if max_mean_intensity_detected is not None:
        mask &= madata.var['mean_intensity_detected'] <= max_mean_intensity_detected
    
    # ---- New: flexible extra filters ----
    if extra_filters is not None:
        for col, (min_val, max_val) in extra_filters.items():
            if col not in madata.var.columns:
                raise ValueError(f"Column '{col}' not found in madata.var")
            if min_val is not None:
                mask &= madata.var[col] >= min_val
            if max_val is not None:
                mask &= madata.var[col] <= max_val

    new_madata = madata[:, mask].copy()

    # ---- Cell-level QC (obs) ----
    X = new_madata.X
    n_metabolites = np.sum((X > 0) & ~np.isnan(X), axis=1)
    total_intensity = np.nansum(X, axis=1)

    new_madata.obs["n_metabolites"] = np.array(n_metabolites).ravel()
    new_madata.obs["total_intensity"] = np.array(total_intensity).ravel()

    return new_madata

# Calculate the mean and variance for each metabolic feature and store them in var.
def calculate_qc_metrics(madata: ad.AnnData,mean_only_detected: bool = False) -> ad.AnnData:
    """
    Calculate QC metrics for single-cell metabolomics data.

    Parameters
    ----------
    madata : ad.AnnData
        Input AnnData object.

    Returns
    -------
    ad.AnnData
        A new AnnData object with additional QC metrics stored in
        .obs and .var:
        - obs: "n_metabolites", "total_intensity"
        - var: "n_cells", "detect_rate", "mean_intensity"
    """
    new_madata = madata.copy()
    X = new_madata.X
    # ---- Cell-level QC (obs) ----
    n_metabolites = np.sum((X > 0) & ~np.isnan(X), axis=1)
    total_intensity = np.nansum(X, axis=1)

    new_madata.obs["n_metabolites"] = np.array(n_metabolites).ravel()
    new_madata.obs["total_intensity"] = np.array(total_intensity).ravel()

    # ---- Feature-level QC (var) ----
    n_cells = np.sum((X > 0) & ~np.isnan(X), axis=0)
    detect_rate = n_cells / np.sum(~np.isnan(X), axis=0) * 100
    if mean_only_detected:
        # 避免除以 0
        mean_intensity = np.array([
            np.nanmean(col[col > 0]) if np.any(col > 0) else 0.0
            for col in X.T
        ])
        new_madata.var["mean_intensity_detected"] = np.array(mean_intensity).ravel()
    else:
        mean_intensity = np.nanmean(X, axis=0)
        new_madata.var["mean_intensity"] = np.array(mean_intensity).ravel()

    new_madata.var["hits"] = np.array(n_cells).ravel()
    new_madata.var["hit_rate"] = np.array(detect_rate).ravel()
    

    return new_madata

# Principal component analysis.
def pca():
    pass


# Subsample to a fraction of the number of observations.
def sample_data(madata: ad.AnnData, obs_names: Optional[list[any]] = None, var_names: Optional[list[int]] = None, exclude: bool = False) -> ad.AnnData:
    """
    Subset or exclude cells or features based on given obs_names or var_names.

    Parameters:
        madata: AnnData object
        obs_names: List of cell IDs (strings or integers) to include/exclude
        var_names: List of feature IDs (integers) to include/exclude
        exclude: If True, remove the specified obs or vars; if False, keep only the specified ones

    Returns:
        Subsetted AnnData object
    """
    obs_mask = np.ones(madata.n_obs, dtype=bool)
    var_mask = np.ones(madata.n_vars, dtype=bool)

    if obs_names is not None:
        # Convert integers to formatted cell strings if needed
        if all(isinstance(x, int) for x in obs_names):
            obs_names = [f'Cell{str(x).zfill(5)}' for x in obs_names]
        obs_index = madata.obs_names.isin(obs_names)
        obs_mask = ~obs_index if exclude else obs_index

    if var_names is not None:
        var_index = madata.var_names.astype(int).isin(var_names)
        var_mask = ~var_index if exclude else var_index

    return madata[obs_mask, var_mask].copy()

# Downsample counts from count matrix.
def downsample_counts():
    pass


'''
Multiplet refers to two or more cells that have not benn completely separated during the isolation process and 
are captured and analyzed as a single unit. This results in the analysis containing mixed information from two or more cells,
thereby affecting the accuracy of the experimental data.
'''
#  Detect and remove multiplets from the data.
def remove_multiplets():
    pass

def fill_nan_values(
    madata: ad.AnnData,
    fill_method: Literal["zero", "min_fraction"] = "zero",
    fraction: float = 0.5,
) -> ad.AnnData:
    """
    Fill NaN values in AnnData.X matrix.

    Parameters
    ----------
    madata : ad.AnnData
        Input AnnData object. The .X matrix may contain NaN values.
    fill_method : {"zero", "min_fraction"}, default="zero"
        Strategy for filling NaN values.
        - "zero": replace all NaN with 0.
        - "min_fraction": replace NaN with (min_nonNaN_value * fraction) for each feature (column).
    fraction : float, default=0.5
        Used only when fill_method="min_fraction".
        Must be between 0 and 1.

    Returns
    -------
    ad.AnnData
        A new AnnData object with NaNs filled.
    """
    new_madata = madata.copy()
    X = new_madata.X.copy()
    if fill_method == "zero":
        X = np.nan_to_num(X, nan=0.0)
    elif fill_method == "min_fraction":
        if not (0 < fraction <= 1):
            raise ValueError("When fill_method='min_fraction', fraction must be between 0 and 1.")
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if valid.size > 0:
                min_val = np.min(valid)
                fill_val = min_val * fraction
                col[np.isnan(col)] = fill_val
                X[:, j] = col
            else:
                # if all NaN in this feature, fill with 0
                X[:, j] = np.zeros_like(col)
    else:
        raise ValueError(f"Unsupported fill_method: {fill_method}")
    new_madata.X = X
    return new_madata

def normalize_data(
    madata: ad.AnnData,
    normalization: Literal["total_intensity", "TIC_corrected"] = "total_intensity",
    target_sum: float = 1e4,
    log_transform: bool = False,
    zscore: bool = False,
) -> ad.AnnData:
    """
    Normalize single-cell metabolomics data.

    Parameters
    ----------
    madata : ad.AnnData
        Input AnnData object. Expects .obs['total_intensity'] if cell_normalization is True.
    cell_normalization : bool, default=True
        If True, normalize each cell by its total intensity.
    log_transform : bool, default=True
        If True, apply log1p transformation.
    feature_zscore : bool, default=False
        If True, standardize each metabolite to mean=0, std=1 (Z-score).
    nan_safe : bool, default=True
        If True, ignore NaN during calculations.

    Returns
    -------
    ad.AnnData
        New AnnData object with normalized .X
    """
    new_madata = madata.copy()
    X = new_madata.X.astype(float)

    if normalization == "total_intensity":
        if "total_intensity" not in new_madata.obs:
            raise KeyError("madata.obs 中未找到 'total_intensity'。")
        cell_sums = new_madata.obs["total_intensity"].values
        cell_sums = np.clip(cell_sums, np.finfo(float).eps, None)
        X = X * (target_sum / cell_sums[:, None])

    elif normalization == "TIC_corrected":
        if "TIC_corrected" not in new_madata.obs:
            raise KeyError("madata.obs 中未找到 'TIC_corrected'。")
        tic = new_madata.obs["TIC_corrected"].values.astype(float)
        tic = np.clip(tic, np.finfo(float).eps, None)
        X = X * (target_sum / tic[:, None])

    else:
        raise ValueError("cell_normalization 必须为 'total_intensity' 或 'TIC_corrected'")

    # ---- log1p ----
    if log_transform:
        X = np.log1p(X)
    
    # ---- feature Z-score ----
    if zscore:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std
    
    new_madata.X = X
    return new_madata