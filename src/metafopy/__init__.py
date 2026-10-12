"""metafopy: meta-analysis in Python, re-implementing the R metafor package.

See https://www.metafor-project.org/ for the R package this mirrors.
"""

from __future__ import annotations

from importlib.metadata import version

from metafopy import datasets
from metafopy._data import MetaData
from metafopy._formula import RandomTerm, parse_random
from metafopy.effects import cmicalc, escalc
from metafopy.methods.anova import LRTResult, WaldResult, anova
from metafopy.methods.blup import BlupResult, blup
from metafopy.methods.confint import ConfintResult, confint
from metafopy.methods.pairmat import pairmat
from metafopy.methods.predict import PredictResult, predict
from metafopy.methods.ranef import RanefResult, ranef
from metafopy.methods.vif import VIFEntry, VIFResult, vif
from metafopy.models.rma_uni import Control, FitStats, RMAResult, rma, rma_uni

__version__ = version('metafopy')

__all__ = [
    'BlupResult',
    'ConfintResult',
    'Control',
    'FitStats',
    'LRTResult',
    'MetaData',
    'PredictResult',
    'RMAResult',
    'RandomTerm',
    'RanefResult',
    'VIFEntry',
    'VIFResult',
    'WaldResult',
    '__version__',
    'anova',
    'blup',
    'cmicalc',
    'confint',
    'datasets',
    'escalc',
    'pairmat',
    'parse_random',
    'predict',
    'ranef',
    'rma',
    'rma_uni',
    'vif',
]
