"""metafopy: meta-analysis in Python, re-implementing the R metafor package.

See https://www.metafor-project.org/ for the R package this mirrors.
"""

from __future__ import annotations

from importlib.metadata import version

from metafopy import datasets
from metafopy._data import MetaData
from metafopy._formula import RandomTerm, parse_random
from metafopy.methods.anova import LRTResult, WaldResult, anova
from metafopy.methods.confint import ConfintResult, confint
from metafopy.methods.predict import PredictResult, predict
from metafopy.models.rma_uni import Control, FitStats, RMAResult, rma, rma_uni

__version__ = version('metafopy')

__all__ = [
    'ConfintResult',
    'Control',
    'FitStats',
    'LRTResult',
    'MetaData',
    'PredictResult',
    'RMAResult',
    'RandomTerm',
    'WaldResult',
    '__version__',
    'anova',
    'confint',
    'datasets',
    'parse_random',
    'predict',
    'rma',
    'rma_uni',
]
