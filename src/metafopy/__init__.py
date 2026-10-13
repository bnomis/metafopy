"""metafopy: meta-analysis in Python, re-implementing the R metafor package.

See https://www.metafor-project.org/ for the R package this mirrors.
"""

from __future__ import annotations

from importlib.metadata import version

from metafopy import datasets, transf
from metafopy._data import MetaData
from metafopy._formula import RandomTerm, parse_random
from metafopy.bias import FsnResult, HcResult, RanktestResult, RegtestResult, TesResult, TrimfillResult, fsn, hc, ranktest, regtest, tes, trimfill
from metafopy.core.rng import set_seed
from metafopy.effects import cmicalc, escalc, summary_escalc
from metafopy.methods.anova import LRTResult, WaldResult, anova
from metafopy.methods.blup import BlupResult, blup
from metafopy.methods.confint import ConfintResult, confint
from metafopy.methods.gosh import GoshResult, gosh
from metafopy.methods.influence import InfluenceResult, ResidualResult, cooks_distance, dfbetas, influence, rstandard, rstudent
from metafopy.methods.leave1out import CumulResult, Leave1outResult, cumul, leave1out
from metafopy.methods.pairmat import pairmat
from metafopy.methods.permutest import PermutestResult, permutest
from metafopy.methods.predict import PredictResult, predict
from metafopy.methods.profile import ProfileResult, profile
from metafopy.methods.ranef import RanefResult, ranef
from metafopy.methods.vif import VIFEntry, VIFResult, vif
from metafopy.models.rma_uni import Control, FitStats, RMAResult, rma, rma_uni

__version__ = version('metafopy')

__all__ = [
    'BlupResult',
    'ConfintResult',
    'Control',
    'CumulResult',
    'FitStats',
    'FsnResult',
    'GoshResult',
    'HcResult',
    'InfluenceResult',
    'LRTResult',
    'Leave1outResult',
    'MetaData',
    'PermutestResult',
    'PredictResult',
    'ProfileResult',
    'RMAResult',
    'RandomTerm',
    'RanefResult',
    'RanktestResult',
    'RegtestResult',
    'ResidualResult',
    'TesResult',
    'TrimfillResult',
    'VIFEntry',
    'VIFResult',
    'WaldResult',
    '__version__',
    'anova',
    'blup',
    'cmicalc',
    'confint',
    'cooks_distance',
    'cumul',
    'datasets',
    'dfbetas',
    'escalc',
    'fsn',
    'gosh',
    'hc',
    'influence',
    'leave1out',
    'pairmat',
    'parse_random',
    'permutest',
    'predict',
    'profile',
    'ranef',
    'ranktest',
    'regtest',
    'rma',
    'rma_uni',
    'rstandard',
    'rstudent',
    'set_seed',
    'summary_escalc',
    'tes',
    'transf',
    'trimfill',
    'vif',
]
