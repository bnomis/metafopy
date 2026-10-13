"""metafopy: meta-analysis in Python, re-implementing the R metafor package.

See https://www.metafor-project.org/ for the R package this mirrors.
"""

import metafopy.methods.binary
import metafopy.methods.location_scale  # noqa: F401  registers the predict/confint/... methods for location-scale models
from metafopy import datasets, transf
from metafopy.base import __version__
from metafopy.bias import FsnResult, HcResult, RanktestResult, RegtestResult, TesResult, TrimfillResult, fsn, hc, ranktest, regtest, tes, trimfill
from metafopy.core.rng import set_seed
from metafopy.effects import cmicalc, escalc, summary_escalc
from metafopy.formula import RandomTerm, parse_random
from metafopy.metadata import MetaData
from metafopy.methods.anova import LRTResult, WaldResult, anova
from metafopy.methods.blocks import bldiag, blsplit
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
from metafopy.methods.rcalc import RCalcResult, rcalc
from metafopy.methods.robust import robust
from metafopy.methods.selmodel import SelProfileResult
from metafopy.methods.vcalc import vcalc
from metafopy.methods.vif import VIFEntry, VIFResult, vif
from metafopy.models.glmm_fit import GLMMControl
from metafopy.models.glmm_result import GLMMResult
from metafopy.models.mv_result import RMAMVResult
from metafopy.models.rma_glmm import rma_glmm
from metafopy.models.rma_ls_result import LSControl, RMALSResult
from metafopy.models.rma_mh import MHResult, rma_mh
from metafopy.models.rma_mv import MVControl, rma_mv
from metafopy.models.rma_peto import PetoResult, rma_peto
from metafopy.models.rma_uni import FitStats, RMAResult, rma, rma_uni
from metafopy.models.sel_result import SelmodelResult
from metafopy.models.selmodel import selmodel
from metafopy.models.tau2 import Control
from metafopy.plot import addpoly, baujat, forest, funnel, labbe, plot_gosh, plot_infl, plot_profile, plot_selmodel, qqnorm, radial

__all__ = [
    'BlupResult',
    'ConfintResult',
    'Control',
    'CumulResult',
    'FitStats',
    'FsnResult',
    'GLMMControl',
    'GLMMResult',
    'GoshResult',
    'HcResult',
    'InfluenceResult',
    'LRTResult',
    'LSControl',
    'Leave1outResult',
    'MHResult',
    'MVControl',
    'MetaData',
    'PermutestResult',
    'PetoResult',
    'PredictResult',
    'ProfileResult',
    'RCalcResult',
    'RMALSResult',
    'RMAMVResult',
    'RMAResult',
    'RandomTerm',
    'RanefResult',
    'RanktestResult',
    'RegtestResult',
    'ResidualResult',
    'SelProfileResult',
    'SelmodelResult',
    'TesResult',
    'TrimfillResult',
    'VIFEntry',
    'VIFResult',
    'WaldResult',
    '__version__',
    'addpoly',
    'anova',
    'baujat',
    'bldiag',
    'blsplit',
    'blup',
    'cmicalc',
    'confint',
    'cooks_distance',
    'cumul',
    'datasets',
    'dfbetas',
    'escalc',
    'forest',
    'fsn',
    'funnel',
    'gosh',
    'hc',
    'influence',
    'labbe',
    'leave1out',
    'pairmat',
    'parse_random',
    'permutest',
    'plot_gosh',
    'plot_infl',
    'plot_profile',
    'plot_selmodel',
    'predict',
    'profile',
    'qqnorm',
    'radial',
    'ranef',
    'ranktest',
    'rcalc',
    'regtest',
    'rma',
    'rma_glmm',
    'rma_mh',
    'rma_mv',
    'rma_peto',
    'rma_uni',
    'robust',
    'rstandard',
    'rstudent',
    'selmodel',
    'set_seed',
    'summary_escalc',
    'tes',
    'transf',
    'trimfill',
    'vcalc',
    'vif',
]
