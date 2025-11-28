from __future__ import annotations

from typing import TYPE_CHECKING

from ._dendrogram import dendrogram

if TYPE_CHECKING:
    from typing import Any

def __getattr__(name: str) -> Any:
    if name == "pca":
        from ..preprocessing import pca

        return pca
    raise AttributeError(name)