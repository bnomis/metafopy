"""Normalisation of model inputs into a single :class:`MetaData` bundle.

This mirrors the argument handling at the top of metafor's ``rma.uni`` and
``rma.mv``: arguments may be arrays or column names in ``data``; ``subset`` is
applied first, then the ``na.action`` rule; ``mods`` may be a one-sided formula
or a matrix (in which case an intercept is added unless ``intercept=False``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from metafopy._formula import design_matrix

__all__ = ["MetaData", "NAAction", "make_unique"]

NAAction: TypeAlias = Literal["omit", "fail", "pass"]
Column: TypeAlias = ArrayLike | str

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def make_unique(labels: Sequence[str]) -> tuple[str, ...]:
    """Make labels unique the way R's ``make.unique`` does (``a``, ``a.1``, ``a.2``)."""
    seen: set[str] = set()
    counts: dict[str, int] = {}
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
            continue
        n = counts.get(label, 0)
        while True:
            n += 1
            candidate = f"{label}.{n}"
            if candidate not in seen:
                break
        counts[label] = n
        seen.add(candidate)
        out.append(candidate)
    return tuple(out)


def _column(value: Column | None, data: pd.DataFrame | None, name: str) -> FloatArray | None:
    if value is None:
        return None
    if isinstance(value, str):
        if data is None:
            raise ValueError(f"{name}={value!r} names a column, but no data was given")
        if value not in data.columns:
            raise KeyError(f"column {value!r} (for {name}) not found in data")
        value = data[value].to_numpy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def _subset_mask(subset: ArrayLike | str | None, k: int, data: pd.DataFrame | None) -> BoolArray:
    if subset is None:
        return np.ones(k, dtype=np.bool_)
    if isinstance(subset, str):
        if data is None:
            raise ValueError("a string subset needs data to evaluate against")
        values = np.asarray(data.eval(subset))
    else:
        values = np.asarray(subset)
    if values.dtype == np.bool_:
        if values.shape != (k,):
            raise ValueError(f"logical subset must have length {k}, got {values.shape}")
        return values.copy()
    mask = np.zeros(k, dtype=np.bool_)
    mask[values.astype(np.intp)] = True
    return mask


@dataclass(frozen=True, slots=True)
class MetaData:
    """Model inputs after subsetting and missing-value handling.

    Attributes hold only the rows used in fitting (``k`` of them). ``keep``
    maps them back to the ``k_all`` rows that were supplied, so that results
    can be reported on the original rows (metafor's ``na.action="na.pass"``).
    """

    yi: FloatArray
    vi: FloatArray
    X: FloatArray
    X_names: tuple[str, ...]
    slab: tuple[str, ...]
    keep: BoolArray
    weights: FloatArray | None = None
    ni: FloatArray | None = None
    intercept: bool = True
    data: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def k(self) -> int:
        """Number of estimates used in fitting."""
        return int(self.yi.shape[0])

    @property
    def k_all(self) -> int:
        """Number of rows supplied before subsetting and NA removal."""
        return int(self.keep.shape[0])

    @property
    def p(self) -> int:
        """Number of fixed-effects coefficients."""
        return int(self.X.shape[1])

    @classmethod
    def from_inputs(
        cls,
        yi: Column | None = None,
        vi: Column | None = None,
        *,
        sei: Column | None = None,
        weights: Column | None = None,
        ni: Column | None = None,
        mods: str | ArrayLike | pd.DataFrame | None = None,
        intercept: bool = True,
        data: pd.DataFrame | None = None,
        slab: Column | Sequence[str] | None = None,
        subset: ArrayLike | str | None = None,
        na_action: NAAction = "omit",
    ) -> MetaData:
        """Build a :class:`MetaData` from ``rma``-style arguments.

        When ``yi`` is omitted and ``data`` has ``yi``/``vi`` columns (as
        returned by :func:`metafopy.escalc`), those columns are used.
        """
        if yi is None and data is not None and "yi" in data.columns:
            yi = "yi"
            if vi is None and sei is None and "vi" in data.columns:
                vi = "vi"
        y = _column(yi, data, "yi")
        if y is None:
            raise ValueError("yi must be given")
        k_all = y.shape[0]

        v = _column(vi, data, "vi")
        se = _column(sei, data, "sei")
        if v is None and se is not None:
            v = se**2
        if v is None:
            raise ValueError("one of vi or sei must be given")

        w = _column(weights, data, "weights")
        n = _column(ni, data, "ni")
        for name, arr in (("vi", v), ("weights", w), ("ni", n)):
            if arr is not None and arr.shape[0] != k_all:
                raise ValueError(f"length of {name} ({arr.shape[0]}) differs from yi ({k_all})")

        X, X_names, x_missing, has_intercept = cls._design(mods, intercept, data, k_all)

        if slab is None:
            labels: tuple[str, ...] = tuple(str(i + 1) for i in range(k_all))
        elif isinstance(slab, str):
            if data is None:
                raise ValueError(f"slab={slab!r} names a column, but no data was given")
            labels = tuple(str(s) for s in data[slab].tolist())
        else:
            labels = tuple(str(s) for s in np.asarray(slab, dtype=object).tolist())
        if len(labels) != k_all:
            raise ValueError(f"length of slab ({len(labels)}) differs from yi ({k_all})")
        labels = make_unique(labels)

        keep = _subset_mask(subset, k_all, data)
        missing = np.isnan(y) | np.isnan(v) | x_missing
        if w is not None:
            missing |= np.isnan(w)
        missing &= keep
        if missing.any():
            if na_action == "fail":
                raise ValueError("missing values in data")
            keep = keep & ~missing

        if np.any(v[keep] < 0):
            raise ValueError("there are outcomes with negative sampling variances")
        if w is not None and np.any(w[keep] < 0):
            raise ValueError("there are negative weights")

        idx = np.flatnonzero(keep)
        return cls(
            yi=y[idx],
            vi=v[idx],
            X=X[idx],
            X_names=X_names,
            slab=tuple(labels[i] for i in idx),
            keep=keep,
            weights=None if w is None else w[idx],
            ni=None if n is None else n[idx],
            intercept=has_intercept,
            data=data,
        )

    @staticmethod
    def _design(
        mods: str | ArrayLike | pd.DataFrame | None,
        intercept: bool,
        data: pd.DataFrame | None,
        k: int,
    ) -> tuple[FloatArray, tuple[str, ...], BoolArray, bool]:
        if mods is None:
            return np.ones((k, 1)), ("intrcpt",), np.zeros(k, dtype=np.bool_), True
        if isinstance(mods, str):
            if data is None:
                raise ValueError("a formula for mods needs data")
            X, names, missing = design_matrix(mods, data)
            return X, names, missing, "intrcpt" in names
        if isinstance(mods, pd.DataFrame):
            names = tuple(str(c) for c in mods.columns)
            X = mods.to_numpy(dtype=float)
        else:
            X = np.asarray(mods, dtype=np.float64)
            if X.ndim == 1:
                X = X[:, None]
            names = tuple(f"X{j + 1}" for j in range(X.shape[1]))
        if X.shape[0] != k:
            raise ValueError(f"mods has {X.shape[0]} rows, expected {k}")
        if intercept:
            X = np.column_stack([np.ones(k), X])
            names = ("intrcpt", *names)
        return X.astype(np.float64), names, np.isnan(X).any(axis=1), intercept
