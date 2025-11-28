"""Simple preprocessing functions."""

from __future__ import annotations

from re import M
import warnings
from functools import singledispatch

import numba
import numpy as np
from anndata import AnnData
from fast_array_utils import stats 
from fast_array_utils.conv import to_dense
from pandas.api.types import CategoricalDtype
from scipy.sparse import data
from sklearn.utils import check_array, sparsefuncs

from .. import logging as logg
from .._compat import CSBase, CSRBase, DaskArray, deprecated, njit, old_positionals
from .._settings import settings as sett
from typing import TYPE_CHECKING, Optional, Union, Sequence, Tuple, TypeVar, overload

from .._utils import (
    _check_array_function_arguments,
    is_backed_type,
    raise_not_implemented_error_if_backed_type,
    sanitize_anndata,
    view_to_actual,
)
from ..get import _check_mask, _get_obs_rep, _set_obs_rep

from ._distributed import materialize_as_ndarray

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence
    from numbers import Number
    from typing import Literal

Number = Union[int, float, np.integer, np.floating]
FloatPair = Tuple[float, float]
TimeWindows = Union[Sequence[Sequence[Number]], np.ndarray, Sequence[Number]]

A = TypeVar("A", bound=np.ndarray | CSBase | DaskArray)


def filter_cells(
    data: AnnData | CSBase | np.ndarray | DaskArray,
    *,
    min_metabolites: int | None = None,
    max_metabolites: int | None = None,
    min_total_intensity: float | None = None,
    max_total_intensity: float | None = None,
    time_windows: Optional[TimeWindows] = None,
    time_key: str = "scan_start_time",
    excluded_time_windows: bool = False,
    inplace: bool = True,
) -> AnnData | tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """ Filter cells basd on total intensity and numbers of metabolites detected and time windows.
    
    For instance, only keep cells with at least `min_metabolites` metaboltes detected 
    or `min_total_intensity` total intensithy. This is measurement outliers,
    or time windows extracted from the data.
    i.e. “unreliable” observations.

    Parameters
    ----------
    madata
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to metabolites.
    min_metabolites
        Minimum number of metabolites detected per cell.
    max_metabolites
        Maximum number of metabolites detected per cell.
    min_total_intensity
        Minimum total intensity per cell.
    max_total_intensity
        Maximum total intensity per cell.
    time_windows
        Time windows to filter cells.
    excluded_time_windows
        Include or exclude time windows.
    inplace
        If True, filter cells directly on the original `madata` (inplace) and return None.  
        If False, return a new filtered `AnnData` object, leaving the original unchanged.

    Returns
    -------
    Returns a new filtered AnnData if `copy=False`;  
    otherwise modifies the original inplace and returns None.

    Examples
    --------
    >>> import metacell as mc
    # filter n_metabolites >= 200 and min_total_intensity >= 1e5
    >>> mc.pp.filter_cells(madata, min_metabolites=200, min_total_intensity=1e5)
    # Select the two time periods 8-13 and 14-20.
    >>> mc.pp.filter_cells(madta, time_windows=[(8,13),(10,11)])
    """

    if isinstance(data, AnnData):
        raise_not_implemented_error_if_backed_type(data.X, "filter_cells")
        madata = data if inplace else data.copy()
        cell_subset, n_metabolites, total_intensity = materialize_as_ndarray(
            filter_cells(
                madata.X,
                min_metabolites=min_metabolites,
                max_metabolites=max_metabolites,
                min_total_intensity=min_total_intensity,
                max_total_intensity=max_total_intensity,
            ),
        )
        madata.obs["n_metabolites"] = n_metabolites
        madata.obs["total_intensity"] = total_intensity
        if time_windows is not None:
            if time_key not in madata.obs.columns:
                raise KeyError(f"Column {time_key} not found in madata.obs")
            scan_times = madata.obs[time_key].to_numpy(dtype=float)
            time_windows = _normalize_time_windows(time_windows)
            mask = np.zeros_like(scan_times, dtype=bool)
            for start, end in time_windows:
                mask |= (scan_times >= start) & (scan_times <= end)
            if excluded_time_windows:
                mask = ~mask
            cell_subset &= mask


        madata._inplace_subset_obs(cell_subset)
        return madata
    X = data
    n_metabolites = stats.sum(X>0, axis=1)
    total_intensity = stats.sum(X, axis=1)

    msg = ''
    mask = np.ones(X.shape[0], dtype=bool)
    if min_metabolites is not None:
        mask &= n_metabolites >= min_metabolites
        msg += f"less than {min_metabolites} detected metabolites;\t"
    if max_metabolites is not None:
        mask &= n_metabolites <= max_metabolites
        msg += f"more than {max_metabolites} detected metabolites;\t"
    if min_total_intensity is not None:
        mask &= total_intensity >= min_total_intensity
        msg += f"total intensity lower than {min_total_intensity};\t"
    if max_total_intensity is not None:
        mask &= total_intensity <= max_total_intensity
        msg += f"total intensity higher than {max_total_intensity};\t"

    s = stats.sum(~mask)
    if s > 0:
        prefix = f"filtered {s} cells that have\t"
        msg = prefix + msg
    
        logg.info(msg)
    return mask, n_metabolites, total_intensity
    
def _normalize_time_windows(windows: TimeWindows) -> np.ndarray:
    """
    Normalize various 'time_windows' inputs to a (n, 2) float array.

    Accepted forms:
      - Single window: [L, R], (L, R), np.array([L, R])
      - Multiple windows: [(L1, R1), (L2, R2)], [[L1, R1], [L2, R2]],
                          np.array([[L1, R1], [L2, R2]])
    Endpoints can be None indicating unbounded:
      - left None -> -inf; right None -> +inf
    Validate: L <= R

    Returns
    -------
    np.ndarray of shape (n, 2), dtype float
    """
    if windows is None:
        raise ValueError("`time_windows` must not be None.")

    # ---- numpy array input ----
    if isinstance(windows, np.ndarray):
        arr = np.asarray(windows, dtype=object)  # keep None if present
        if arr.ndim == 1:
            if arr.size != 2:
                raise ValueError("For a single window, provide length-2 array [start, end].")
            arr = arr.reshape(1, 2)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pass
        else:
            raise ValueError("`time_windows` ndarray must have shape (2,) or (n, 2).")
        # Coerce None to +/-inf and cast to float
        out = np.empty_like(arr, dtype=float)
        for i, (a, b) in enumerate(arr):
            la = -np.inf if a is None else float(a)
            rb =  np.inf if b is None else float(b)
            if la > rb:
                raise ValueError(f"Invalid window with start > end: {(a, b)!r}")
            out[i, 0], out[i, 1] = la, rb
        return out

    # ---- list/tuple (or other sequence) input ----
    if isinstance(windows, (list, tuple)):
        if len(windows) == 0:
            raise ValueError("`time_windows` must be a non-empty sequence.")

        # Case A: single window like [L, R] / (L, R)
        is_single_candidate = (
            len(windows) == 2 and
            not isinstance(windows[0], (list, tuple, np.ndarray)) and
            not isinstance(windows[1], (list, tuple, np.ndarray))
        )
        if is_single_candidate:
            L, R = windows  # type: ignore[misc]
            L = -np.inf if L is None else float(L)
            R =  np.inf if R is None else float(R)
            if L > R:
                raise ValueError(f"Invalid window with start > end: {windows!r}")
            return np.asarray([[L, R]], dtype=float)

        # Case B: multiple windows like [(L1,R1), (L2,R2)] / [[...], [...]]
        out_list: list[tuple[float, float]] = []
        for w in windows:  # type: ignore[assignment]
            if not isinstance(w, (list, tuple, np.ndarray)) or len(w) != 2:
                raise ValueError("Each time window must be a length-2 iterable (start, end).")
            a, b = w
            la = -np.inf if a is None else float(a)
            rb =  np.inf if b is None else float(b)
            if la > rb:
                raise ValueError(f"Invalid window with start > end: {w!r}")
            out_list.append((la, rb))
        return np.asarray(out_list, dtype=float)

    raise ValueError(
        "`time_windows` must be a (start, end) pair, or a sequence of such pairs, "
        "or a numpy array with shape (2,) or (n,2)."
    )

def filter_metabolites(
    data: AnnData | CSBase | np.ndarray | DaskArray,
    *,
    min_cells: int | None = None,
    max_cells: int | None = None,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
    inplace: bool = True,
) -> AnnData | tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Filter metabolites based on the intensity and the number of cells.
    
    Keeps metabolites that have at least `min_intensity` intensity or detected in
    at least `min_cells` cells or have at most `max_intensity` intensity or are expressed
    in at most `max_cells` cells.

    Parameters
    ----------
    data
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows corresponds to cells and 
        columns to metabolites.
    min_cells
        Minimum number of cell.
    max_cells
        Maximum number of metabolites detected per cell.
    min_intensity
        Minimum intensity required for a metabolite to pass filtering.
    max_intensity
        Maximum intensity required for a metabolite to pass filtering.
    inplace
        If True, filter cells directly on the original `madata` (inplace) and return None.  
        If False, return a new filtered `AnnData` object, leaving the original unchanged.
    Return
    ------
    Returns a new filtered AnnData if `copy=False`;  
    otherwise modifies the original inplace and returns None.

    """
    if isinstance(data, AnnData):
        raise_not_implemented_error_if_backed_type(data.X, "filter_cells")
        madata = data if inplace else data.copy()
        metabolite_subset, n_cells, mean_intensity = materialize_as_ndarray(
            filter_metabolites(
                madata.X,
                min_cells=min_cells,
                max_cells=max_cells,
                min_intensity=min_intensity,
                max_intensity=max_intensity,
            ),
        )

        madata.var['n_cells'] = n_cells
        madata.var['hit_rate'] = n_cells/len(madata.obs) * 100
        madata.var['mean_intensity'] = mean_intensity
    
        madata._inplace_subset_var(metabolite_subset)
        return madata
    
    X = data
    
    n_cells = stats.sum(X>0, axis=0)
    mean_intensity = np.divide(
    stats.sum(np.where(X > 0, X, 0), axis=0),
    stats.sum(X > 0, axis=0),
    where=stats.sum(X > 0, axis=0) > 0,
    out=np.zeros(X.shape[1], dtype=float),
    )

    msg = ''
    mask = np.ones(X.shape[1], dtype=bool)
    if min_cells is not None:
        mask &= n_cells >= min_cells
        msg += f"less than {min_cells} detected metabolites;\t"
    if max_cells is not None:
        mask &= n_cells <= max_cells
        msg += f"more than {max_cells} detected metabolites;\t"
    if min_intensity is not None:
        mask &= mean_intensity >= min_intensity
        msg += f"total intensity lower than {min_intensity};\t"
    if max_intensity is not None:
        mask &= mean_intensity <= max_intensity
        msg += f"total intensity higher than {max_intensity};\t"
    
    s = stats.sum(~mask)
    if s > 0:
        prefix = f"filtered {s} metabolites that have\t"
        msg = prefix + msg
    
        logg.info(msg)
    return mask, n_cells, mean_intensity


@singledispatch
def log1p(
    data: AnnData | np.ndarray | CSBase,
    *,
    base: Number | None = None,
    copy: bool = False,
    chunked: bool | None = None,
    chunk_size: int | None = None,
    layer: str | None = None,
    obsm: str | None = None,
) -> AnnData | np.ndarray | CSBase | None:
    r"""Logarithmize the data
    Computes :math:`X = \log(X + 1)`,
    where :math:`log` denotes the natural logarithm unless a different base is given.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to metabolites.
    base
        Base of the logarithm. Natural logarithm is used by default.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned.
    chunked
        Process the data matrix in chunks, which will save memory.
        Applies only to :class:`~anndata.AnnData`.
    chunk_size
        `n_obs` of the chunks to process the data in.
    layer
        Entry of layers to transform
    obsm
        Entry of obsm to transform

    Returns
    -------
    Return or update `data`, depending on `copy`.

    """
    _check_array_function_arguments(
        chunked=chunked, chunk_size=chunk_size, layer=layer, obsm=obsm
    )
    return log1p_array(data, copy=copy, base=base)

@log1p.register(CSBase)
def log1p_sparse(X: CSBase, *, base: Number | None = None, copy: bool = False):
    X = check_array(
        X, accept_sparse=("csr", "csc"), dtype=(np.float64, np.float32), copy=copy
    )
    X.data = log1p(X.data, copy=False, base=base)
    return X

@log1p.register(np.ndarray)
def log1p_array(X: np.ndarray, *, base: Number | None = None, copy: bool = False):
    # Can force arrays to be np.ndarrays, but would be useful to not
    # X = check_array(X, dtype=(np.float64, np.float32), ensure_2d=False, copy=copy)
    if copy:
        X = X.astype(float) if not np.issubdtype(X.dtype, np.floating) else X.copy()
    elif not (np.issubdtype(X.dtype, np.floating) or np.issubdtype(X.dtype, complex)):
        X = X.astype(float)
    np.log1p(X, out=X)
    if base is not None:
        np.divide(X, np.log(base), out=X)
    return X


@log1p.register(AnnData)
def log1p_anndata(
    adata: AnnData,
    *,
    base: Number | None = None,
    copy: bool = False,
    chunked: bool = False,
    chunk_size: int | None = None,
    layer: str | None = None,
    obsm: str | None = None,
) -> AnnData | None:
    if "log1p" in adata.uns:
        logg.warning("adata.X seems to be already log-transformed.")

    adata = adata.copy() if copy else adata
    view_to_actual(adata)

    if chunked:
        if (layer is not None) or (obsm is not None):
            msg = (
                "Currently cannot perform chunked operations on arrays not stored in X."
            )
            raise NotImplementedError(msg)
        if adata.isbacked and adata.file._filemode != "r+":
            msg = "log1p is not implemented for backed AnnData with backed mode not r+"
            raise NotImplementedError(msg)
        for chunk, start, end in adata.chunked_X(chunk_size):
            adata.X[start:end] = log1p(chunk, base=base, copy=False)
    else:
        X = _get_obs_rep(adata, layer=layer, obsm=obsm)
        if is_backed_type(X):
            msg = f"log1p is not implemented for matrices of type {type(X)}"
            if layer is not None:
                msg = f"{msg} from layers"
                raise NotImplementedError(msg)
            msg = f"{msg} without `chunked=True`"
            raise NotImplementedError(msg)
        X = log1p(X, copy=False, base=base)
        _set_obs_rep(adata, X, layer=layer, obsm=obsm)

    adata.uns["log1p"] = {"base": base}
    if copy:
        return adata


def sqrt(
    data: AnnData | CSBase | np.ndarray,
    *,
    copy: bool = False,
    chunked: bool = False,
    chunk_size: int | None = None,
) -> AnnData | CSBase | np.ndarray | None:
    r"""Take square root of the data matrix.

    Computes :math:`X = \sqrt(X)`.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    copy
        If an :class:`~anndata.AnnData` object is passed,
        determines whether a copy is returned.
    chunked
        Process the data matrix in chunks, which will save memory.
        Applies only to :class:`~anndata.AnnData`.
    chunk_size
        `n_obs` of the chunks to process the data in.

    Returns
    -------
    Returns or updates `data`, depending on `copy`.

    """
    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        if chunked:
            for chunk, start, end in adata.chunked_X(chunk_size):
                adata.X[start:end] = sqrt(chunk)
        else:
            adata.X = sqrt(data.X)
        return adata if copy else None
    X = data  # proceed with data matrix
    return X.sqrt() if isinstance(X, CSBase) else np.sqrt(X)


DT = TypeVar("DT")

@njit
def _create_regressor_categorical(
    X: np.ndarray, number_categories: int, cat_array: np.ndarray
) -> np.ndarray:
    # create regressor matrix for categorical variables
    # would be best to use X dtype but this matches old behavior
    regressors = np.zeros(X.shape, dtype=np.float32)
    # iterate over categories
    for category in range(number_categories):
        # iterate over genes and calculate mean expression
        # for each gene per category
        mask = category == cat_array
        for ix in numba.prange(X.T.shape[0]):
            regressors[mask, ix] = X.T[ix, mask].mean()
    return regressors

@njit
def get_resid(
    data: np.ndarray,
    regressor: np.ndarray,
    coeff: np.ndarray,
) -> np.ndarray:
    for i in numba.prange(data.shape[0]):
        data[i] -= regressor[i] @ coeff
    return data

def numpy_regress_out(
    data: np.ndarray,
    regressor: np.ndarray,
) -> np.ndarray:
    """Numba kernel for regress out unwanted sorces of variantion.

    Finding coefficient using Linear regression (Linear Least Squares).
    """
    inv_gram_matrix = np.linalg.inv(regressor.T @ regressor)
    coeff = inv_gram_matrix @ (regressor.T @ data)
    data = get_resid(data, regressor, coeff)
    return data

