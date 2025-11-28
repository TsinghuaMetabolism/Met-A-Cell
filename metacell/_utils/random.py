from __future__ import annotations

import random
from collections.abc import Sequence
from contextlib import contextmanager
from functools import WRAPPER_ASSIGNMENTS, wraps
from typing import TYPE_CHECKING

import numpy as np
from sklearn.utils import check_random_state

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

__all__ = [
    "RNGLike",
    "SeedLike",
    "_LegacyRandom",
]

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator
_LegacyRandom = int | np.random.RandomState | None
