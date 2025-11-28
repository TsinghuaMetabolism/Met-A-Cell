"""Preprocessing functions"""
from __future__ import annotations
from ._pca import pca

from ._qc import (
    #filter_cells,
    #filter_metabolites,
    sample_data,
    fill_nan_values,
    normalize_data,
    calculate_qc_metrics,
)

from ._simple import (
    filter_cells,
    filter_metabolites,
    log1p,
)

from ._anndata import (
    drop_anndata_attr,
    rename_var_index,
    to_cell_feature_matrix,
)

from ._pseudo_analysis import (
    plot_branch_celltype_composition,
    plot_celltype_branch_composition,
    plot_metabolite_trend_single_branch,
    plot_metabolite_trend_dual_branch,
    build_dot_agg,  
    build_metabolite_order_by_direction,
)

__all__ = [
    "pca",
    "filter_cells",
    "filter_metabolites",
    "log1p",
    "fill_nan_values",
    "normalize_data",
    "sample_data",
    "calculate_qc_metrics",
    "drop_anndata_attr",
    "rename_var_index",
    "to_cell_feature_matrix",
    "plot_branch_celltype_composition",
    "plot_celltype_branch_composition",
    "plot_metabolite_trend_single_branch",
    "plot_metabolite_trend_dual_branch",
    "build_dot_agg",
    "build_metabolite_order_by_direction",
]