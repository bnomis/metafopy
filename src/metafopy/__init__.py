"""metafopy: meta-analysis in Python, re-implementing the R metafor package.

See https://www.metafor-project.org/ for the R package this mirrors.
"""

import metafopy.methods.binary
import metafopy.methods.location_scale  # registers the predict/confint/... methods for location-scale models
import metafopy.methods.mv_diagnostics  # registers the diagnostics for rma.mv models
import metafopy.methods.mv_profile  # registers profile() for rma.mv models
import metafopy.methods.mv_ranef
import metafopy.models.rma_ls  # noqa: F401  loaded so the annotations of rma_uni resolve under inspect.signature
from metafopy import datasets, transf
from metafopy.base import __version__
from metafopy.bias import FsnResult, HcResult, RanktestResult, RegtestResult, TesResult, TrimfillResult, fsn, hc, ranktest, regtest, tes, trimfill
from metafopy.core.rng import set_seed
from metafopy.effects import (
    TableArray,
    aggregate_escalc,
    cmicalc,
    contrmat,
    conv_2x2,
    conv_delta,
    conv_fivenum,
    conv_wald,
    dfround,
    escalc,
    replmiss,
    summary_escalc,
    to_long,
    to_table,
    to_wide,
    vec2mat,
)
from metafopy.formatters import fmtp, fmtp2, fmtt, fmtx
from metafopy.formula import RandomTerm, parse_random
from metafopy.metadata import MetaData
from metafopy.methods.accessors import df_residual, model_matrix, nobs, se, update
from metafopy.methods.anova import LRTResult, WaldResult, anova
from metafopy.methods.blocks import bldiag, blsplit
from metafopy.methods.blup import BlupResult, blup
from metafopy.methods.confint import ConfintResult, confint
from metafopy.methods.deltamethod import DeltaMethodResult, deltamethod
from metafopy.methods.emm_grid import EmmGrid, contrast, emmeans, emmprep, pairs, regrid
from metafopy.methods.emm_summary import EmmSummary
from metafopy.methods.gosh import GoshResult, gosh
from metafopy.methods.influence import InfluenceResult, ResidualResult, cooks_distance, dfbetas, influence, rstandard, rstudent
from metafopy.methods.leave1out import CumulResult, Leave1outResult, cumul, leave1out
from metafopy.methods.matreg import MatregConfintResult, MatregPredictResult, MatregResult, matreg
from metafopy.methods.pairmat import pairmat
from metafopy.methods.permutest import PermutestResult, permutest
from metafopy.methods.predict import PredictResult, predict
from metafopy.methods.profile import ProfileResult, profile
from metafopy.methods.ranef import RanefResult, ranef
from metafopy.methods.rcalc import RCalcResult, rcalc
from metafopy.methods.robust import robust
from metafopy.methods.selmodel import SelProfileResult
from metafopy.methods.simulate import simulate
from metafopy.methods.vcalc import vcalc
from metafopy.methods.vif import VIFEntry, VIFResult, vif
from metafopy.mfopt import getmfopt, setmfopt
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
from metafopy.plot import (
    addpoly,
    baujat,
    forest,
    forest_cumul,
    funnel,
    labbe,
    llplot,
    plot_cumul,
    plot_gosh,
    plot_infl,
    plot_permutest,
    plot_profile,
    plot_rma,
    plot_selmodel,
    plot_vif,
    qqnorm,
    radial,
    regplot,
)

__all__ = [
    'BlupResult',
    'ConfintResult',
    'Control',
    'CumulResult',
    'DeltaMethodResult',
    'EmmGrid',
    'EmmSummary',
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
    'MatregConfintResult',
    'MatregPredictResult',
    'MatregResult',
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
    'TableArray',
    'TesResult',
    'TrimfillResult',
    'VIFEntry',
    'VIFResult',
    'WaldResult',
    '__version__',
    'addpoly',
    'aggregate_escalc',
    'anova',
    'baujat',
    'bldiag',
    'blsplit',
    'blup',
    'cmicalc',
    'confint',
    'contrast',
    'contrmat',
    'conv_2x2',
    'conv_delta',
    'conv_fivenum',
    'conv_wald',
    'cooks_distance',
    'cumul',
    'datasets',
    'deltamethod',
    'df_residual',
    'dfbetas',
    'dfround',
    'emmeans',
    'emmprep',
    'escalc',
    'fmtp',
    'fmtp2',
    'fmtt',
    'fmtx',
    'forest',
    'forest_cumul',
    'fsn',
    'funnel',
    'getmfopt',
    'gosh',
    'hc',
    'influence',
    'labbe',
    'leave1out',
    'llplot',
    'matreg',
    'model_matrix',
    'nobs',
    'pairmat',
    'pairs',
    'parse_random',
    'permutest',
    'plot_cumul',
    'plot_gosh',
    'plot_infl',
    'plot_permutest',
    'plot_profile',
    'plot_rma',
    'plot_selmodel',
    'plot_vif',
    'predict',
    'profile',
    'qqnorm',
    'radial',
    'ranef',
    'ranktest',
    'rcalc',
    'regplot',
    'regrid',
    'regtest',
    'replmiss',
    'rma',
    'rma_glmm',
    'rma_mh',
    'rma_mv',
    'rma_peto',
    'rma_uni',
    'robust',
    'rstandard',
    'rstudent',
    'se',
    'selmodel',
    'set_seed',
    'setmfopt',
    'simulate',
    'summary_escalc',
    'tes',
    'to_long',
    'to_table',
    'to_wide',
    'transf',
    'trimfill',
    'update',
    'vcalc',
    'vec2mat',
    'vif',
]
