"""BasePlot for dotplot, matrixplot and stacked_violin."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple
from warnings import warn

import numpy as np
from matplotlib import colormaps, gridspec
from matplotlib import pyplot as plt

from .. import logging as logg
from .._compat import old_positionals
from .._utils import _empty
from ._anndata import (
    VarGroups,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    
    _VarNames = str | Sequence[str]

def _var_groups(
    var_names: _VarNames | Mapping[str, _VarNames],
) -> tuple[Sequence[str], VarGroups | None]:
    """Normalize var_names.

    If it’s a mapping, also return var_group_labels and var_group_positions.
    """
    if not isinstance(var_names, Mapping):
        var_names = [var_names] if isinstance(var_names, str) else var_names
        return var_names, None
    if len(var_names) == 0:
        return [], None

    var_group_labels: list[str] = []
    var_names_seq: list[str] = []
    var_group_positions: list[tuple[int, int]] = []
    for label, vars in var_names.items():
        vars_list = [vars] if isinstance(vars, str) else vars
        start = len(var_names_seq)
        # use list() in case var_list is a numpy array or pandas series
        var_names_seq.extend(list(vars_list))
        var_group_labels.append(label)
        var_group_positions.append((start, start + len(vars_list) - 1))
    if not var_names_seq:
        msg = "No valid var_names were passed."
        raise ValueError(msg)
    return var_names_seq, VarGroups(var_group_labels, var_group_positions)
