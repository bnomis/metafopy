"""metafopy: meta-analysis in Python, re-implementing the R metafor package.

See https://www.metafor-project.org/ for the R package this mirrors.
"""

from __future__ import annotations

from importlib.metadata import version

from metafopy import datasets
from metafopy._data import MetaData
from metafopy._formula import RandomTerm, parse_random

__version__ = version("metafopy")

__all__ = ["MetaData", "RandomTerm", "__version__", "datasets", "parse_random"]
