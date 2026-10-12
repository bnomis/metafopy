"""Example datasets from the R ``metadat`` package.

The CSV files under ``data/`` are exported from metadat by
``tests/r/export_datasets.R`` and ship with metafopy so documentation and
tests can reproduce metafor's published examples. metadat is licensed
GPL (>= 2); see https://cran.r-project.org/package=metadat for the sources of
each dataset.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

import pandas as pd

__all__ = ["list_datasets", "load"]

_PACKAGE = "metafopy.datasets.data"


def list_datasets() -> list[str]:
    """Names of the bundled datasets, e.g. ``"dat.bcg"``."""
    files = resources.files(_PACKAGE).iterdir()
    return sorted(f.name.removesuffix(".csv") for f in files if f.name.endswith(".csv"))


@cache
def _read(name: str) -> pd.DataFrame:
    path = resources.files(_PACKAGE).joinpath(f"{name}.csv")
    if not path.is_file():
        raise KeyError(f"unknown dataset {name!r}; see metafopy.datasets.list_datasets()")
    with path.open("rb") as fh:
        return pd.read_csv(fh)


def load(name: str) -> pd.DataFrame:
    """Load a bundled metadat dataset by its R name (``"dat.bcg"``) or ``"bcg"``."""
    if not name.startswith("dat."):
        name = f"dat.{name}"
    return _read(name).copy()
