"""Shared docstrings for preprocessing function parameters."""
from __future__ import annotations

doc_adata_basic = """\
adata
    Annotated data matrix for single-cell metabolomics.\
"""

doc_expr_reps = """\
layer
    If provided, use `adata.layers[layer]` for expression values instead
    of `adata.X`.
use_raw
    If True, use `adata.raw.X` for expression values instead of `adata.X`.\
"""

doc_mask_var_hvg = """\
mask_var
    To run only on a certain set of genes given by a boolean array
    or a string referring to an array in :attr:`~anndata.AnnData.var`.
    By default, uses `.var['highly_variable']` if available, else everything.
use_highly_variable
    Whether to use highly variable genes only, stored in
    `.var['highly_variable']`.
    By default uses them if they have been determined beforehand.

    .. deprecated:: 1.10.0
       Use `mask_var` instead
"""

doc_obs_qc_args = """\
qc_vars
    Keys for boolean columns of `.var` which identify variables you could
    want to control for (e.g. "ERCC" or "mito").
percent_top
    List of ranks (where genes are ranked by expression) at which the cumulative
    proportion of expression will be reported as a percentage. This can be used to
    assess library complexity. Ranks are considered 1-indexed, and if empty or None
    don't calculate.

    E.g. `percent_top=[50]` finds cumulative proportion to the 50th most expressed gene.
"""