"""Formula handling for ``mods=``, ``scale=`` and ``random=`` arguments.

Fixed-effects formulas (``mods``, ``scale``) are expanded with formulaic, and
the resulting column names are rewritten to match R's ``model.matrix`` naming
so that coefficient labels line up with metafor (``intrcpt``, ``factorlevel``).

Random-effects formulas for ``rma.mv`` (``~ 1 | study``, ``~ 1 | a/b``,
``~ inner | outer``) are parsed here into :class:`RandomTerm` objects; the
variance structures they imply are built by the model code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from formulaic import model_matrix
from numpy.typing import NDArray

__all__ = ["RandomTerm", "design_matrix", "parse_random", "r_column_name"]

_LEVEL = re.compile(r"\[(?:T\.)?(.*?)\]")
_CWRAP = re.compile(r"^C\((.*?)(?:,.*)?\)$")


def r_column_name(name: str) -> str:
    """Translate a formulaic column name into R ``model.matrix`` style.

    ``Intercept`` becomes metafor's ``intrcpt``; treatment-coded factor columns
    such as ``alloc[T.random]`` or ``C(alloc)[T.random]`` become ``allocrandom``.
    Interaction terms are translated part by part.
    """
    if name == "Intercept":
        return "intrcpt"
    parts: list[str] = []
    for part in name.split(":"):
        level = _LEVEL.search(part)
        if level is not None:
            var = part[: level.start()]
            m = _CWRAP.match(var)
            if m is not None:
                var = m.group(1)
            part = f"{var}{level.group(1)}"
        parts.append(part)
    return ":".join(parts)


def design_matrix(
    formula: str, data: pd.DataFrame
) -> tuple[NDArray[np.float64], tuple[str, ...], NDArray[np.bool_]]:
    """Build a model matrix from a one-sided formula.

    Returns the matrix (one row per row of ``data``), R-style column names, and
    a boolean mask of rows that contained missing values in any variable used.
    Rows with missing values are kept (filled with NaN) so the caller can apply
    metafor's ``na.action`` rules jointly with ``yi`` and ``vi``.
    """
    rhs = formula.strip()
    if "~" in rhs:
        lhs, _, rhs = rhs.partition("~")
        if lhs.strip():
            raise ValueError(f"expected a one-sided formula, got {formula!r}")
    mm = model_matrix(rhs, data, na_action="ignore")
    X = np.asarray(mm.to_numpy(dtype=float), dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    names = tuple(r_column_name(str(c)) for c in mm.columns)
    # formulaic encodes a missing factor level as all-zero dummies rather than
    # NaN, so detect missingness from the variables the formula actually uses
    spec = mm.model_spec
    required = spec.required_variables if spec is not None else set(data.columns)
    used = [v for v in required if v in data.columns]
    missing = np.isnan(X).any(axis=1) | data[used].isna().any(axis=1).to_numpy()
    return X, names, missing


@dataclass(frozen=True, slots=True)
class RandomTerm:
    """One ``random=`` term of an ``rma.mv`` model.

    ``~ 1 | study`` has ``inner=None`` and ``outer=("study",)``;
    ``~ 1 | district/school`` has ``outer=("district", "school")`` and
    ``nested=True``; ``~ outcome | study`` has ``inner="outcome"``.
    """

    inner: str | None
    outer: tuple[str, ...]
    nested: bool

    def __str__(self) -> str:
        lhs = "1" if self.inner is None else self.inner
        sep = "/" if self.nested else ""
        return f"~ {lhs} | {sep.join(self.outer)}"


def parse_random(formula: str) -> RandomTerm:
    """Parse a single ``rma.mv`` random-effects formula."""
    text = formula.strip()
    if not text.startswith("~"):
        raise ValueError(f"random effects formula must start with '~': {formula!r}")
    body = text[1:]
    if body.count("|") != 1:
        raise ValueError(f"random effects formula needs exactly one '|': {formula!r}")
    lhs, rhs = (s.strip() for s in body.split("|"))
    if not lhs or not rhs:
        raise ValueError(f"malformed random effects formula: {formula!r}")
    outer = tuple(s.strip() for s in rhs.split("/"))
    if any(not s for s in outer):
        raise ValueError(f"malformed nesting in random effects formula: {formula!r}")
    inner = None if lhs == "1" else lhs
    if inner is not None and len(outer) > 1:
        raise ValueError(f"'~ inner | outer' terms cannot also be nested: {formula!r}")
    return RandomTerm(inner=inner, outer=outer, nested=len(outer) > 1)
