import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence


if TYPE_CHECKING:
    import anndata as ad


def drop_anndata_attr(madata: ad.AnnData,
                      attr: str,
                      keys: list[str] | str | None = None,
                      copy: bool = False) -> ad.AnnData:
    """
    Drop attributes or specific keys/columns from an AnnData object.

    Parameters
    ----------
    madata : AnnData
        The input AnnData object.
    attr : str
        The attribute to process. Must be one of: "obs", "var", "obsm", "varm", "obsp".
    keys : list[str] | str | None
        - None or "__all__": drop the entire attribute.
        - list[str] or str: drop only the specified keys/columns.
    copy : bool, default=True
        Whether to return a new AnnData object (True) or modify in place (False).

    Returns
    -------
    AnnData
        The processed AnnData object with the specified attributes/keys removed.

    Raises
    ------
    ValueError
        If `attr` is not one of the allowed attributes.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc3k()

    # Drop the entire `obsm`
    >>> new_adata = drop_anndata_attr(adata, attr="obsm", keys="__all__")

    # Drop specific columns from `obs`
    >>> new_adata = drop_anndata_attr(adata, attr="obs", keys=["n_genes", "percent_mito"])

    # Drop a single column from `var`
    >>> new_adata = drop_anndata_attr(adata, attr="var", keys="highly_variable")

    # In-place modification (not recommended unless necessary)
    >>> drop_anndata_attr(adata, attr="obsp", keys="__all__", copy=False)
    """
    adata = madata.copy() if copy else madata

    if attr not in {"obs", "var", "obsm", "varm", "obsp", "uns"}:
        raise ValueError(f"attr must be one of 'obs', 'var', 'obsm', 'varm', 'obsp', 'uns', but got: {attr}")

    container = getattr(adata, attr)

    if keys is None or keys == "__all__":
        # Drop the entire attribute
        if attr in {"obsm", "varm", "obsp","uns"}:
            container.clear()
        elif attr in {"obs", "var"}:
            # Clear DataFrame but keep index
            container.drop(columns=container.columns, inplace=True)
    else:
        # Drop specified keys/columns
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if k in container:
                del container[k]

    return adata if copy else None


def rename_var_index(
    madata: ad.AnnData,
    col: str = "mz_center",
    decimals: int = 3,
    prefix: str = "mz_",
    copy: bool = True,
    sanitize: bool = True,
    uniq_suffix: str = "-"  # suffix used when making duplicate names unique
):
    """
    Rename var_names according to the following rules:
      1) If a 'metabolite' column exists and the value is non-empty/non-whitespace -> use it;
      2) Otherwise, fall back to labels based on the `col` column (default 'mz_center'),
         rounded to the specified number of decimals, with a prefix.

    Ensures uniqueness of var_names (via AnnData.var_names_make_unique).
    Optionally sanitizes metabolite text (strip whitespace, replace spaces, etc.).
    """
    if copy:
        adata = madata.copy()
    else:
        adata = madata

    # Check if fallback column exists
    if col not in adata.var.columns:
        raise KeyError(f"'{col}' not found in madata.var")

    # Generate baseline m/z labels (for fallback)
    mz_labels = (
        adata.var[col]
        .round(decimals)
        .astype("float64")  # ensure numeric for formatting
        .map(lambda x: f"{prefix}{x:.{decimals}f}")
    )

    # Initialize with fallback labels
    new_names = mz_labels.copy()

    # If metabolite column exists, override with non-empty values
    if "metabolite" in adata.var.columns:
        meta = adata.var["metabolite"].astype(str)

        # Basic cleaning: strip leading/trailing whitespace; treat pure whitespace as empty
        meta = meta.str.strip()
        mask_nonempty = meta.notna() & (meta != "")

        if sanitize:
            # Optional: normalize whitespace and replace with underscores
            meta = (
                meta.str.replace(r"\s+", " ", regex=True)
                    .str.replace(" ", "_")
            )

        # Only replace with metabolite values where non-empty
        new_names.loc[mask_nonempty] = meta.loc[mask_nonempty]

    # Assign as new var_names
    adata.var.index = pd.Index(new_names.astype(str))
    adata.var.index.name = None
    # Ensure uniqueness to avoid downstream errors
    adata.var_names_make_unique(join=uniq_suffix)

    return adata if copy else None


def to_cell_feature_matrix(
    madata: ad.AnnData,
    *,
    save_path: Optional[str] = None,
    var_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Export a cell feature matrix from an AnnData object, with optional saving and
    selective var-columns.

    The function constructs a matrix where:
    - Rows correspond to features (madata.var_names).
    - Columns correspond to cells (madata.obs_names).
    - The feature annotation (madata.var, optionally column-filtered) is inserted
      in the front part of the matrix.
    - The values of the matrix come from madata.X (transposed to features x cells).

    Parameters
    ----------
    madata : AnnData
        Input AnnData object containing single-cell metabolomics data.
    save_path : str, optional
        If provided, save the resulting DataFrame to file. The format is inferred
        from the extension: .csv / .tsv / .xlsx / .parquet.
    var_cols : Sequence[str], optional
        If provided, only keep (and order by) these columns from madata.var.
        Raises ValueError if any requested column is missing.

    Returns
    -------
    pd.DataFrame
        A DataFrame whose first part contains feature annotations (from madata.var,
        possibly column-filtered) and second part contains the feature intensity
        matrix (rows = features, columns = cells). The function does not modify
        the input AnnData object.
    """
    # --- Build feature (intensity) matrix ---
    # AnnData.X is (n_obs x n_vars); we need (n_vars x n_obs), so use X.T.
    # If X is sparse, convert to dense via toarray(); otherwise use as-is.
    madata = madata[madata.obs['scan_start_time'].argsort()]
    cellnumber = ['Cell{:05d}'.format(i + 1) for i in range(len(madata.obs))]
    madata.obs_names = cellnumber

    feature_matrix = pd.DataFrame(
        madata.X.T.toarray() if hasattr(madata.X, "toarray") else madata.X.T,
        index=madata.var_names,    # rows = features
        columns=madata.obs_names   # cols = cells
    )

    # --- Prepare var annotations (front part) ---
    # If var_cols is given, validate and select those columns in order.
    if var_cols is None:
        var_part = madata.var.copy()
    else:
        missing = [c for c in var_cols if c not in madata.var.columns]
        if missing:
            raise ValueError(
                f"Columns not found in madata.var: {missing}. "
                f"Available: {list(madata.var.columns)}"
            )
        var_part = madata.var.loc[:, list(var_cols)]

    # Ensure row alignment by feature index (var_names)
    var_part = var_part.reindex(madata.var_names)

    # --- Concatenate var annotations (front) and feature intensities (back) ---
    cell_feature = pd.concat([var_part, feature_matrix], axis=1)

    # --- Optional saving ---
    if save_path:
        path = Path(save_path)
        ext = path.suffix.lower()
        # Choose writer by extension; always include index (feature names)
        if ext == ".csv":
            cell_feature.to_csv(path, index=True)
        elif ext == ".tsv":
            cell_feature.to_csv(path, sep="\t", index=True)
        elif ext == ".xlsx":
            with pd.ExcelWriter(path) as writer:
                cell_feature.to_excel(writer, sheet_name="cell_feature", index=True)
        elif ext == ".parquet":
            cell_feature.to_parquet(path, index=True)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Use one of: .csv, .tsv, .xlsx, .parquet"
            )
        return None

    return cell_feature


def to_obs_feature(
    madata: ad.AnnData,
    *,
    obs_cols: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a cell-by-feature table: left part contains selected columns from
    madata.obs, right part contains feature intensities (from X), with feature
    names corresponding to madata.var_names.

    Parameters
    ----------
    madata : AnnData
        AnnData object containing single-cell metabolomics data (X shape: n_obs x n_vars).
    obs_cols : Sequence[str], optional
        Which columns from `madata.obs` to include. If None, include all columns.
    feature_names : Sequence[str], optional
        Which features (by `madata.var_names`) to include as intensity columns, in the
        given order. If None, include all features.
    save_path : str, optional
        If provided, save the table to file. Format inferred by extension:
        .csv / .tsv / .xlsx / .parquet.

    Returns
    -------
    pd.DataFrame
        A DataFrame with index = cells (`madata.obs_names`).
        Left block: selected `madata.obs` columns.
        Right block: intensity columns for selected features; column names match `var_names`.

    Notes
    -----
    - Does not modify the input AnnData object.
    - If X is sparse, it will be converted to dense for export.
    """
    # --- Build obs (metadata) part ---
    # Use selected obs columns or all columns if not specified
    if obs_cols is None:
        obs_part = madata.obs.copy()
    else:
        missing_obs = [c for c in obs_cols if c not in madata.obs.columns]
        if missing_obs:
            raise ValueError(
                f"Columns not found in madata.obs: {missing_obs}. "
                f"Available: {list(madata.obs.columns)}"
            )
        obs_part = madata.obs.loc[:, list(obs_cols)].copy()

    # Ensure row index is cells
    obs_part.index = pd.Index(madata.obs_names, name=obs_part.index.name)

    # --- Build intensity (feature) part with rows=cells, cols=features ---
    X = madata.X
    # If X is sparse, convert to dense; otherwise ensure ndarray
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    # Initialize with all features by default
    all_features = pd.Index(madata.var_names)
    if feature_names is None:
        chosen_features = all_features
        X_used = X_dense  # shape: (n_obs x n_vars)
    else:
        # Validate feature_names exist in var_names; preserve user-specified order
        missing_feat = [f for f in feature_names if f not in all_features]
        if missing_feat:
            raise ValueError(
                f"Features not found in madata.var_names: {missing_feat}. "
                f"Available: {list(all_features)}"
            )
        # Map feature order to column indices
        col_indices = all_features.get_indexer(feature_names)
        X_used = X_dense[:, col_indices]
        chosen_features = pd.Index(feature_names)

    feature_part = pd.DataFrame(
        X_used,
        index=madata.obs_names,      # rows = cells
        columns=chosen_features      # cols = features (named by var_names)
    )

    # --- Concatenate obs metadata (left) and feature intensities (right) ---
    # Both have the same row index = obs_names
    table = pd.concat([obs_part, feature_part], axis=1)

    # --- Optional save ---
    if save_path:
        path = Path(save_path)
        ext = path.suffix.lower()
        if ext == ".csv":
            table.to_csv(path, index=True)
        elif ext == ".tsv":
            table.to_csv(path, sep="\t", index=True)
        elif ext == ".xlsx":
            with pd.ExcelWriter(path) as writer:
                table.to_excel(writer, sheet_name="cell_obs_feature", index=True)
        elif ext == ".parquet":
            table.to_parquet(path, index=True)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Use one of: "
                f".csv, .tsv, .xlsx, .parquet"
            )
        return None

    return table

