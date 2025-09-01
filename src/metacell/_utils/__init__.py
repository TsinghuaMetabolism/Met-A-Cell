"""Utility functions and classes."""
from __future__ import annotations


import re
import h5py
from enum import Enum
from textwrap import indent
from types import UnionType
from functools import partial, reduce, wraps
from operator import mul, or_, truediv
from .._compat import CSBase, DaskArray, _CSArray, pkg_version
from typing import (
    TYPE_CHECKING,
    overload,
    Literal,
    Union,
    get_origin,
    get_args,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, KeysView, Mapping
    from pathlib import Path
    from typing import Any, TypeVar
    from anndata import AnnData
    from igraph import Graph
    from numpy.typing import ArrayLike, NDArray
    from .._compat import CSRBase
    from ..neighbors import NeighborsParams, RPForestDict
    _ForT = TypeVar("_ForT", bound=Callable | type)

class Empty(Enum):
    token = 0

    def __repr__(self) -> str:
        return "_empty"

from anndata import __version__ as anndata_version
from packaging.version import Version
if Version(anndata_version) >= Version("0.10.0"):
    from anndata._core.sparse_dataset import \
        BaseCompressedSparseDataset as SparseDataset
else:
    from anndata._core.sparse_dataset import SparseDataset

_empty = Empty.token

LegacyUnionType = type(Union[int, str])  # noqa: UP007

class NeighborsView:
    """Convenience class for accessing neighbors graph representations.

    Allows to access neighbors distances, connectivities and settings
    dictionary in a uniform manner.

    Parameters
    ----------
    adata
        AnnData object.
    key
        This defines where to look for neighbors dictionary,
        connectivities, distances.

        neigh = NeighborsView(adata, key)
        neigh['distances']
        neigh['connectivities']
        neigh['params']
        'connectivities' in neigh
        'params' in neigh

        is the same as

        adata.obsp[adata.uns[key]['distances_key']]
        adata.obsp[adata.uns[key]['connectivities_key']]
        adata.uns[key]['params']
        adata.uns[key]['connectivities_key'] in adata.obsp
        'params' in adata.uns[key]

    """

    def __init__(self, adata: AnnData, key=None):
        self._connectivities = None
        self._distances = None

        if key is None or key == "neighbors":
            if "neighbors" not in adata.uns:
                msg = 'No "neighbors" in .uns'
                raise KeyError(msg)
            self._neighbors_dict = adata.uns["neighbors"]
            self._conns_key = "connectivities"
            self._dists_key = "distances"
        else:
            if key not in adata.uns:
                msg = f"No {key!r} in .uns"
                raise KeyError(msg)
            self._neighbors_dict = adata.uns[key]
            self._conns_key = self._neighbors_dict["connectivities_key"]
            self._dists_key = self._neighbors_dict["distances_key"]

        if self._conns_key in adata.obsp:
            self._connectivities = adata.obsp[self._conns_key]
        if self._dists_key in adata.obsp:
            self._distances = adata.obsp[self._dists_key]

        # fallback to uns
        self._connectivities, self._distances = _fallback_to_uns(
            self._neighbors_dict,
            self._connectivities,
            self._distances,
            self._conns_key,
            self._dists_key,
        )

    @overload
    def __getitem__(self, key: Literal["distances", "connectivities"]) -> CSRBase: ...
    @overload
    def __getitem__(self, key: Literal["params"]) -> NeighborsParams: ...
    @overload
    def __getitem__(self, key: Literal["rp_forest"]) -> RPForestDict: ...
    @overload
    def __getitem__(self, key: Literal["connectivities_key"]) -> str: ...

    def __getitem__(self, key: str):
        if key == "distances":
            if "distances" not in self:
                msg = f"No {self._dists_key!r} in .obsp"
                raise KeyError(msg)
            return self._distances
        elif key == "connectivities":
            if "connectivities" not in self:
                msg = f"No {self._conns_key!r} in .obsp"
                raise KeyError(msg)
            return self._connectivities
        elif key == "connectivities_key":
            return self._conns_key
        else:
            return self._neighbors_dict[key]

    def __contains__(self, key: str) -> bool:
        if key == "distances":
            return self._distances is not None
        elif key == "connectivities":
            return self._connectivities is not None
        else:
            return key in self._neighbors_dict


_leading_whitespace_re = re.compile("(^[ ]*)(?:[^ \n])", re.MULTILINE)
def _doc_params(**replacements: str):
    def dec(obj: _ForT) -> _ForT:
        assert obj.__doc__
        assert "\t" not in obj.__doc__

        # The first line of the docstring is unindented,
        # so find indent size starting after it.
        start_line_2 = obj.__doc__.find("\n") + 1
        assert start_line_2 > 0, f"{obj.__name__} has single-line docstring."
        n_spaces = min(
            len(m.group(1))
            for m in _leading_whitespace_re.finditer(obj.__doc__[start_line_2:])
        )

        # The placeholder is already indented, so only indent subsequent lines
        indented_replacements = {
            k: indent(v, " " * n_spaces)[n_spaces:] for k, v in replacements.items()
        }
        obj.__doc__ = obj.__doc__.format_map(indented_replacements)
        return obj

    return dec

def _check_use_raw(
    adata: AnnData,
    use_raw: None | bool,  # noqa: FBT001
    *,
    layer: str | None = None,
) -> bool:
    """Normalize checking `use_raw`.

    My intentention here is to also provide a single place to throw a deprecation warning from in future.
    """
    if use_raw is not None:
        return use_raw
    if layer is not None:
        return False
    return adata.raw is not None

# `get_args` returns `tuple[Any]` so I don’t think it’s possible to get the correct type here
def get_literal_vals(typ: UnionType | Any) -> KeysView[Any]:
    """Get all literal values from a Literal or Union of … of Literal type."""
    if isinstance(typ, UnionType | LegacyUnionType):
        return reduce(
            or_, (dict.fromkeys(get_literal_vals(t)) for t in get_args(typ))
        ).keys()
    if get_origin(typ) is Literal:
        return dict.fromkeys(get_args(typ)).keys()
    msg = f"{typ} is not a valid Literal"
    raise TypeError(msg)

def _fallback_to_uns(dct, conns, dists, conns_key, dists_key):
    if conns is None and conns_key in dct:
        conns = dct[conns_key]
    if dists is None and dists_key in dct:
        dists = dct[dists_key]

    return conns, dists

def sanitize_anndata(adata: AnnData) -> None:
    """Transform string annotations to categoricals."""
    adata._sanitize()

def _choose_graph(
    adata: AnnData, obsp: str | None, neighbors_key: str | None
) -> CSBase:
    """Choose connectivities from neighbors or another obsp entry."""
    if obsp is not None and neighbors_key is not None:
        msg = "You can't specify both obsp, neighbors_key. Please select only one."
        raise ValueError(msg)

    if obsp is not None:
        return adata.obsp[obsp]
    else:
        neighbors = NeighborsView(adata, neighbors_key)
        if "connectivities" not in neighbors:
            msg = (
                "You need to run `pp.neighbors` first to compute a neighborhood graph."
            )
            raise ValueError(msg)
        return neighbors["connectivities"]


def _resolve_axis(
    axis: Literal["obs", 0, "var", 1],
) -> tuple[Literal[0], Literal["obs"]] | tuple[Literal[1], Literal["var"]]:
    if axis in {0, "obs"}:
        return (0, "obs")
    if axis in {1, "var"}:
        return (1, "var")
    msg = f"`axis` must be either 0, 1, 'obs', or 'var', was {axis!r}"
    raise ValueError(msg)

def is_backed_type(X: object) -> bool:
    return isinstance(X, SparseDataset | h5py.File | h5py.Dataset)


def raise_not_implemented_error_if_backed_type(X: object, method_name: str) -> None:
    if is_backed_type(X):
        msg = f"{method_name} is not implemented for matrices of type {type(X)}"
        raise NotImplementedError(msg)