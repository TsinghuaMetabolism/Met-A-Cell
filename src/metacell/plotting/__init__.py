"""Plotting functions and classes."""
from __future__ import annotations
from .._compat import deprecated
from . import palettes



from ._plt_umap import umap_scm,umap_cell_type,umap_analysis
from ._plt_resource import plt_cell_type_colors
from ._plt_tsne import tsne_scm,tsne_cell_type,tsne_analysis
from ._plt_peaks import plt_scMet,plt_tic_time,plt_scm_events,plt_merged_scm,plt_cell_type_marker_annotation,plt_feature_eic
#核心绘图功能包括了以下几个部分
# 1. Scatter plots for embeddings (eg. UMAP, t-SNE).
# 2. 按照不同聚类
from ._plt_qc import plt_distribution

from ._anndata import (
    ranking,
    violin,
    clustermap,
    heatmap,
    scatter,
)