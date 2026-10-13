"""The ``rma_glmm`` calls whose R results are stored in ``reference/phase3/glmm.json`` (see ``r/generate_glmm.R``)."""

import typing

import numpy
import pandas

import metafopy
import metafopy.datasets

type Fit = typing.Callable[[], metafopy.GLMMResult]


def bcg() -> pandas.DataFrame:
    return metafopy.datasets.load('bcg')


def collins() -> pandas.DataFrame:
    return metafopy.datasets.load('collins1985a')


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def nielweise() -> pandas.DataFrame:
    return metafopy.datasets.load('nielweise2007')


def hart() -> pandas.DataFrame:
    return metafopy.datasets.load('hart1999')


def bcg_na() -> pandas.DataFrame:
    d = bcg()
    d.loc[2, 'ablat'] = numpy.nan
    return d


def bcg_with_time() -> pandas.DataFrame:
    d = bcg()
    d['ti'] = d['tpos'] + d['tneg']
    return d


BCG: dict[str, typing.Any] = {'ai': 'tpos', 'bi': 'tneg', 'ci': 'cpos', 'di': 'cneg', 'measure': 'OR'}
# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
COL: dict[str, typing.Any] = {'ai': 'b.xci', 'n1i': 'nci', 'ci': 'b.xti', 'n2i': 'nti', 'measure': 'OR'}
# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
NW: dict[str, typing.Any] = {'ai': 'ai', 'n1i': 'n1i', 'ci': 'ci', 'n2i': 'n2i', 'measure': 'OR'}
# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
IRR: dict[str, typing.Any] = {'x1i': 'x1i', 'x2i': 'x2i', 't1i': 't1i', 't2i': 't2i', 'measure': 'IRR'}


def on(data: pandas.DataFrame, args: dict[str, typing.Any], **kw: typing.Any) -> Fit:
    return lambda: metafopy.rma_glmm(**args, data=data, **kw)


# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
CASES: dict[str, Fit] = {
    'fs_ml': on(bcg(), BCG, model='UM.FS'),
    'fs_fe': on(bcg(), BCG, model='UM.FS', method='FE'),
    'fs_ml_mod': on(bcg(), BCG, model='UM.FS', mods='~ ablat'),
    'fs_ml_mod_fe': on(bcg(), BCG, model='UM.FS', mods='~ ablat + year', method='FE'),
    'fs_ml_coding1': on(bcg(), BCG, model='UM.FS', coding=1),
    'fs_ml_t': on(bcg(), BCG, model='UM.FS', test='t', level=90),
    'fs_collins': on(collins(), COL, model='UM.FS'),
    'fs_nw': on(nielweise(), NW, model='UM.FS'),
    'fs_subset': on(bcg(), BCG, model='UM.FS', subset=(bcg()['alloc'] == 'random').to_numpy()),
    'fs_ml_mod_na': on(bcg_na(), BCG, model='UM.FS', mods='~ ablat'),
    'rs_ml': on(bcg(), BCG, model='UM.RS'),
    'rs_ml_cor': on(bcg(), BCG, model='UM.RS', cor=True),
    'rs_fe': on(bcg(), BCG, model='UM.RS', method='FE'),
    'rs_ml_mod': on(bcg(), BCG, model='UM.RS', mods='~ ablat'),
    'rs_collins': on(collins(), COL, model='UM.RS'),
    'al_ml': on(bcg(), BCG, model='CM.AL'),
    'al_fe': on(bcg(), BCG, model='CM.AL', method='FE'),
    'al_ml_mod': on(bcg(), BCG, model='CM.AL', mods='~ ablat'),
    'al_collins': on(collins(), COL, model='CM.AL'),
    'al_nw': on(nielweise(), NW, model='CM.AL'),
    'el_ml': on(bcg(), BCG, model='CM.EL'),
    'el_fe': on(bcg(), BCG, model='CM.EL', method='FE'),
    'el_fe_mod': on(bcg(), BCG, model='CM.EL', mods='~ ablat', method='FE'),
    'el_collins': on(collins(), COL, model='CM.EL'),
    'el_nw': on(nielweise(), NW, model='CM.EL'),
    'irr_fs': on(hart(), IRR, model='UM.FS'),
    'irr_fs_fe': on(hart(), IRR, model='UM.FS', method='FE'),
    'irr_rs': on(hart(), IRR, model='UM.RS'),
    'irr_el': on(hart(), IRR, model='CM.EL'),
    'irr_el_mod': on(hart(), IRR, model='CM.EL', mods='~ year'),
    'irr_al': on(hart(), IRR, model='CM.AL'),
    'plo': on(bcg(), {'xi': 'tpos', 'mi': 'tneg', 'measure': 'PLO'}),
    'plo_fe': on(bcg(), {'xi': 'tpos', 'mi': 'tneg', 'measure': 'PLO'}, method='FE'),
    'plo_mod': on(bcg(), {'xi': 'tpos', 'mi': 'tneg', 'measure': 'PLO'}, mods='~ ablat'),
    'irln': on(bcg_with_time(), {'xi': 'tpos', 'ti': 'ti', 'measure': 'IRLN'}),
    'irln_mod': on(bcg_with_time(), {'xi': 'tpos', 'ti': 'ti', 'measure': 'IRLN'}, mods='~ ablat'),
}
