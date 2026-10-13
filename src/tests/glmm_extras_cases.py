"""The experimental-measure ``rma_glmm`` calls stored in ``reference/phase6/glmm_extras.json`` (see ``r/generate_glmm_extras.R``)."""

import functools
import typing
import warnings

import glmm_cases
import numpy

import metafopy
import metafopy.transf

type Fit = typing.Callable[[], metafopy.GLMMResult]

# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
PROP: dict[str, typing.Any] = {'xi': 'tpos', 'mi': 'tneg'}


def two_group(data_name: str, measure: str, **kw: typing.Any) -> Fit:
    data = {'bcg': glmm_cases.bcg, 'bcg_na': glmm_cases.bcg_na, 'col': glmm_cases.collins, 'nw': glmm_cases.nielweise}[data_name]()
    args = {'bcg': glmm_cases.BCG, 'bcg_na': glmm_cases.BCG, 'col': glmm_cases.COL, 'nw': glmm_cases.NW}[data_name]
    return glmm_cases.on(data, {**args, 'measure': measure}, **kw)


def one_group(measure: str, **kw: typing.Any) -> Fit:
    return glmm_cases.on(glmm_cases.bcg(), {**PROP, 'measure': measure}, **kw)


# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
CASES: dict[str, Fit] = {
    'rr_fs_ml': two_group('bcg', 'RR', model='UM.FS'),
    'rr_fs_fe': two_group('bcg', 'RR', model='UM.FS', method='FE'),
    'rr_fs_mod': two_group('bcg', 'RR', model='UM.FS', mods='~ ablat'),
    'rr_fs_mod_fe': two_group('bcg', 'RR', model='UM.FS', mods='~ ablat + year', method='FE'),
    'rr_fs_t': two_group('bcg', 'RR', model='UM.FS', test='t', level=90),
    'rr_fs_mod_na': two_group('bcg_na', 'RR', model='UM.FS', mods='~ ablat'),
    'rr_rs_ml': two_group('bcg', 'RR', model='UM.RS'),
    'rr_rs_fe': two_group('bcg', 'RR', model='UM.RS', method='FE'),
    'rr_collins': two_group('col', 'RR', model='UM.FS'),
    'rr_collins_fe': two_group('col', 'RR', model='UM.FS', method='FE'),
    'rr_nw': two_group('nw', 'RR', model='UM.FS'),
    'rr_nw_fe': two_group('nw', 'RR', model='UM.FS', method='FE'),
    'rd_fs_ml': two_group('bcg', 'RD', model='UM.FS'),
    'rd_fs_fe': two_group('bcg', 'RD', model='UM.FS', method='FE'),
    'rd_fs_mod': two_group('bcg', 'RD', model='UM.FS', mods='~ ablat'),
    'rd_fs_mod_fe': two_group('bcg', 'RD', model='UM.FS', mods='~ ablat', method='FE'),
    'rd_rs_ml': two_group('bcg', 'RD', model='UM.RS'),
    'rd_rs_fe': two_group('bcg', 'RD', model='UM.RS', method='FE'),
    'rd_collins': two_group('col', 'RD', model='UM.FS'),
    'rd_collins_fe': two_group('col', 'RD', model='UM.FS', method='FE'),
    'rd_nw': two_group('nw', 'RD', model='UM.FS'),
    'rd_nw_fe': two_group('nw', 'RD', model='UM.FS', method='FE'),
    'pr_ml': one_group('PR'),
    'pr_fe': one_group('PR', method='FE'),
    'pr_mod': one_group('PR', mods='~ ablat'),
    'pr_mod_fe': one_group('PR', mods='~ ablat', method='FE'),
    'pln_ml': one_group('PLN'),
    'pln_fe': one_group('PLN', method='FE'),
    'pln_mod': one_group('PLN', mods='~ ablat'),
    'pln_t': one_group('PLN', test='t', level=90),
    'pln_model_arg': one_group('PLN', model='CM.EL'),
}


# -- predict() ---------------------------------------------------------------------------------------------------

type Predict = typing.Callable[[], metafopy.PredictResult]


@functools.cache
def model(name: str) -> metafopy.GLMMResult:
    """The fitted models that the ``predict`` cases use (cached, as fitting is the slow part)."""
    bcg = glmm_cases.bcg()
    prop: dict[str, typing.Any] = {'xi': 'tpos', 'mi': 'tneg'}
    fits: dict[str, Fit] = {
        'or': glmm_cases.on(bcg, glmm_cases.BCG, model='UM.FS'),
        'or_mod': glmm_cases.on(bcg, glmm_cases.BCG, model='UM.FS', mods='~ ablat'),
        'or_mod2': glmm_cases.on(bcg, glmm_cases.BCG, model='UM.FS', mods='~ ablat + year'),
        'or_fe': glmm_cases.on(bcg, glmm_cases.BCG, model='UM.FS', method='FE'),
        'or_t': glmm_cases.on(bcg, glmm_cases.BCG, model='UM.FS', test='t', mods='~ ablat'),
        'al': glmm_cases.on(bcg, glmm_cases.BCG, model='CM.AL'),
        'el': glmm_cases.on(bcg, glmm_cases.BCG, model='CM.EL'),
        'rs': glmm_cases.on(bcg, glmm_cases.BCG, model='UM.RS'),
        'irr': glmm_cases.on(glmm_cases.hart(), glmm_cases.IRR, model='CM.EL'),
        'plo': glmm_cases.on(bcg, {**prop, 'measure': 'PLO'}),
        'plo_mod': glmm_cases.on(bcg, {**prop, 'measure': 'PLO'}, mods='~ ablat'),
        'irln': glmm_cases.on(glmm_cases.bcg_with_time(), {'xi': 'tpos', 'ti': 'ti', 'measure': 'IRLN'}),
        'na': glmm_cases.on(glmm_cases.bcg_na(), glmm_cases.BCG, model='UM.FS', mods='~ ablat'),
        'rr_mod': glmm_cases.on(bcg, {**glmm_cases.BCG, 'measure': 'RR'}, model='UM.FS', mods='~ ablat'),
        'pln': glmm_cases.on(bcg, {**prop, 'measure': 'PLN'}),
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fits[name]()


def predicting(name: str, *args: typing.Any, **kw: typing.Any) -> Predict:
    return lambda: metafopy.predict(model(name), *args, **kw)


# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
PREDICT: dict[str, Predict] = {
    'or': predicting('or'),
    'or_transf_exp': predicting('or', transf=numpy.exp),
    'or_transf_ilogit': predicting('or', transf=metafopy.transf.transf_ilogit),
    'or_transf_ilogit_int': predicting('or', transf=metafopy.transf.transf_ilogit_int),
    'or_transf_exp_mode': predicting('or', transf=metafopy.transf.transf_exp_mode),
    'or_targs': predicting('or', transf=lambda x, targs: x * targs['m'], targs={'m': 3}),
    'or_level': predicting('or', level=90),
    'or_fe': predicting('or_fe'),
    'or_fe_exp': predicting('or_fe', transf=numpy.exp),
    'or_mod': predicting('or_mod'),
    'or_mod_newmods': predicting('or_mod', newmods=[10, 30, 50]),
    'or_mod_newmods_exp': predicting('or_mod', newmods=[10, 30, 50], transf=numpy.exp),
    'or_mod_newmods_level': predicting('or_mod', newmods=[10, 30], level=80),
    'or_mod_nointercept': predicting('or_mod', newmods=[10, 30], intercept=False),
    'or_mod2_newmods': predicting('or_mod2', newmods=[[10, 1950], [30, 1980]]),
    'or_mod2_row': predicting('or_mod2', newmods=[10, 1950]),
    'or_mod2_p': predicting('or_mod2', newmods=[[1, 10, 1950], [1, 30, 1980]]),
    'or_t': predicting('or_t'),
    'or_t_newmods': predicting('or_t', newmods=[10, 30, 50], level=99),
    'al': predicting('al'),
    'el': predicting('el'),
    'rs': predicting('rs'),
    'irr': predicting('irr', transf=numpy.exp),
    'plo': predicting('plo', transf=metafopy.transf.transf_ilogit),
    'plo_mod': predicting('plo_mod', newmods=[10, 30, 50], transf=metafopy.transf.transf_ilogit),
    'irln': predicting('irln', transf=numpy.exp),
    'na_mod': predicting('na'),
    'rr_mod': predicting('rr_mod', newmods=[10, 30], transf=numpy.exp),
    'pln': predicting('pln', transf=numpy.exp),
}
