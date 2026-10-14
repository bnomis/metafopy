"""The ``rma_glmm`` calls with ``control={'package': ...}`` whose R results are stored in ``reference/phase8/glmm_packages.json``.

The fixture comes from ``r/generate_glmm_packages.R``.
"""

import typing

import glmm_cases

import metafopy

type Control = dict[str, typing.Any]
type Case = typing.Callable[[Control], metafopy.GLMMResult]

ONE_GROUP_ARGS: dict[str, typing.Any] = {'xi': 'tpos', 'mi': 'tneg'}
RATE_ARGS: dict[str, typing.Any] = {'xi': 'tpos', 'ti': 'ti', 'measure': 'IRLN'}


def case(data: typing.Callable[[], typing.Any], args: dict[str, typing.Any], **kw: typing.Any) -> Case:
    """A fit of ``args`` on ``data`` with the keyword arguments ``kw`` and the control settings handed in by the test."""
    return lambda control: metafopy.rma_glmm(**args, data=data(), control=control, **kw)


def bcg(**kw: typing.Any) -> Case:
    return case(glmm_cases.bcg, glmm_cases.BCG, **kw)


def one_group(measure: str, **kw: typing.Any) -> Case:
    return case(glmm_cases.bcg, {**ONE_GROUP_ARGS, 'measure': measure}, **kw)


# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
CASES: dict[str, Case] = {
    'fs_ml': bcg(model='UM.FS'),
    'fs_ml_mod': bcg(model='UM.FS', mods='~ ablat'),
    'fs_ml_coding1': bcg(model='UM.FS', coding=1),
    'fs_ml_t': bcg(model='UM.FS', test='t', level=90),
    'fs_ml_mod_na': case(glmm_cases.bcg_na, glmm_cases.BCG, model='UM.FS', mods='~ ablat'),
    'fs_collins': case(glmm_cases.collins, glmm_cases.COL, model='UM.FS'),
    'fs_nw': case(glmm_cases.nielweise, glmm_cases.NW, model='UM.FS'),
    'rs_ml': bcg(model='UM.RS'),
    'rs_ml_cor': bcg(model='UM.RS', cor=True),
    'rs_fe': bcg(model='UM.RS', method='FE'),
    'rs_ml_mod': bcg(model='UM.RS', mods='~ ablat'),
    'rs_collins': case(glmm_cases.collins, glmm_cases.COL, model='UM.RS'),
    'al_ml': bcg(model='CM.AL'),
    'al_ml_mod': bcg(model='CM.AL', mods='~ ablat'),
    'al_collins': case(glmm_cases.collins, glmm_cases.COL, model='CM.AL'),
    'al_nw': case(glmm_cases.nielweise, glmm_cases.NW, model='CM.AL'),
    'irr_fs': case(glmm_cases.hart, glmm_cases.IRR, model='UM.FS'),
    'irr_rs': case(glmm_cases.hart, glmm_cases.IRR, model='UM.RS'),
    'irr_al': case(glmm_cases.hart, glmm_cases.IRR, model='CM.AL'),
    'irr_el': case(glmm_cases.hart, glmm_cases.IRR, model='CM.EL'),
    'irr_el_mod': case(glmm_cases.hart, glmm_cases.IRR, model='CM.EL', mods='~ year'),
    'plo': one_group('PLO'),
    'plo_mod': one_group('PLO', mods='~ ablat'),
    'irln': case(glmm_cases.bcg_with_time, RATE_ARGS),
    'irln_mod': case(glmm_cases.bcg_with_time, RATE_ARGS, mods='~ ablat'),
}

# the number of quadrature points only matters for GLMMadaptive
GLMMADAPTIVE_ONLY: dict[str, Case] = {
    'fs_nagq3': bcg(model='UM.FS', nAGQ=3),
    'fs_nagq15': bcg(model='UM.FS', nAGQ=15),
    'rs_nagq3': bcg(model='UM.RS', nAGQ=3),
    'al_nagq1': bcg(model='CM.AL', nAGQ=1),
    'plo_nagq1': one_group('PLO', nAGQ=1),
}

# the experimental measures that R fits with at least one package
EXPERIMENTAL: dict[str, Case] = {
    'rr_fs_ml': case(glmm_cases.bcg, {**glmm_cases.BCG, 'measure': 'RR'}, model='UM.FS'),
    'pln_ml': one_group('PLN'),
}


def all_cases(package: str) -> dict[str, Case]:
    extra = GLMMADAPTIVE_ONLY if package == 'GLMMadaptive' else {}
    return {**CASES, **extra, **EXPERIMENTAL}
