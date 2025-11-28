from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _utils

def _get_palette(adata, values_key: str, palette=None):
    color_key = f"{values_key}_colors"
    if adata.obs[values_key].dtype == bool:
        values = pd.Categorical(adata.obs[values_key].astype(str))
    else:
        values = pd.Categorical(adata.obs[values_key])
    if palette:
        _utils._set_colors_for_categorical_obs(adata, values_key, palette)
    elif color_key not in adata.uns or len(adata.uns[color_key]) < len(
        values.categories
    ):
        #  set a default palette in case that no colors or too few colors are found
        _utils._set_default_colors_for_categorical_obs(adata, values_key)
    else:
        _utils._validate_palette(adata, values_key)
    return dict(
        zip(
            values.categories,
            adata.uns[color_key][: len(values.categories)],
            strict=True,
        )
    )
