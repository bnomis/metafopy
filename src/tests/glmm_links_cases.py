"""The ``rma_glmm`` calls with ``link`` and ``family`` whose R results are stored in ``reference/phase9/glmm_links.json`` (see ``r/generate_glmm_links.R``)."""

import typing

import metafopy
import metafopy.core.glm
import metafopy.datasets

type Fit = typing.Callable[[], metafopy.GLMMResult]

LINKS = ('probit', 'cloglog', 'cauchit')


def b2(measure: str = 'OR', **kw: typing.Any) -> Fit:
    return lambda: metafopy.rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'), measure=measure, **kw)


def c2(**kw: typing.Any) -> Fit:
    return lambda: metafopy.rma_glmm(ai='b.xci', n1i='nci', ci='b.xti', n2i='nti', data=metafopy.datasets.load('collins1985a'), measure='OR', **kw)


def n2(**kw: typing.Any) -> Fit:
    return lambda: metafopy.rma_glmm(ai='ai', n1i='n1i', ci='ci', n2i='n2i', data=metafopy.datasets.load('nielweise2007'), measure='OR', **kw)


def h2(**kw: typing.Any) -> Fit:
    return lambda: metafopy.rma_glmm(x1i='x1i', x2i='x2i', t1i='t1i', t2i='t2i', data=metafopy.datasets.load('hart1999'), measure='IRR', **kw)


def p1(measure: str = 'PLO', **kw: typing.Any) -> Fit:
    return lambda: metafopy.rma_glmm(xi='tpos', mi='tneg', data=metafopy.datasets.load('bcg'), measure=measure, **kw)


def r1(**kw: typing.Any) -> Fit:
    data = metafopy.datasets.load('bcg')
    data['ti'] = data['tpos'] + data['tneg']
    return lambda: metafopy.rma_glmm(xi='tpos', ti='ti', data=data, measure='IRLN', **kw)


def build() -> dict[str, Fit]:
    cases: dict[str, Fit] = {}
    for lk in LINKS:
        if lk != 'cauchit':
            cases[f'or_fs_{lk}'] = b2(model='UM.FS', link=lk)
            cases[f'or_fs_{lk}_mod'] = b2(model='UM.FS', link=lk, mods='~ ablat')
        cases[f'or_fs_{lk}_fe'] = b2(model='UM.FS', link=lk, method='FE')
        cases[f'or_fs_{lk}_mod_fe'] = b2(model='UM.FS', link=lk, mods='~ ablat + year', method='FE')
        if lk != 'cauchit':
            cases[f'or_rs_{lk}_fe'] = b2(model='UM.RS', link=lk, method='FE')
    cases['or_rs_probit'] = b2(model='UM.RS', link='probit')
    cases['or_rs_cloglog'] = b2(model='UM.RS', link='cloglog')
    cases['or_fs_probit_t'] = b2(model='UM.FS', link='probit', test='t', level=90)
    cases['or_fs_probit_nw'] = n2(model='UM.FS', link='probit')
    cases['or_fs_probit_nw_fe'] = n2(model='UM.FS', link='probit', method='FE')
    cases['or_fs_cloglog_col'] = c2(model='UM.FS', link='cloglog')
    cases['or_fs_cloglog_col_fe'] = c2(model='UM.FS', link='cloglog', method='FE')
    cases['or_fs_logit'] = b2(model='UM.FS', link='logit')
    cases['or_fs_sqrt_fe'] = b2(model='UM.FS', link='sqrt', method='FE')
    cases['or_fam_probit'] = b2(model='UM.FS', family=metafopy.core.glm.binomial(link='probit'))
    cases['or_fam_binomial'] = b2(model='UM.FS', family='binomial')
    cases['or_fam_cloglog_fe'] = b2(model='UM.FS', family=metafopy.core.glm.binomial(link='cloglog'), method='FE')
    cases['rr_fam_binomial'] = b2('RR', model='UM.FS', family='binomial')
    cases['rr_link_probit'] = b2('RR', model='UM.FS', link='probit')
    cases['rd_link_probit_fe'] = b2('RD', model='UM.FS', link='probit', method='FE')
    cases['plo_fam_cloglog'] = p1('PLO', family=metafopy.core.glm.binomial(link='cloglog'))
    cases['al_probit'] = b2(model='CM.AL', link='probit')
    cases['el_probit_fe'] = b2(model='CM.EL', link='probit', method='FE')
    cases['al_family'] = b2(model='CM.AL', family=metafopy.core.glm.binomial(link='probit'))
    for lk in ('identity', 'sqrt'):
        cases[f'irr_fs_{lk}'] = h2(model='UM.FS', link=lk)
        cases[f'irr_fs_{lk}_fe'] = h2(model='UM.FS', link=lk, method='FE')
    cases['irr_rs_sqrt_fe'] = h2(model='UM.RS', link='sqrt', method='FE')
    cases['irr_fam_poisson_sqrt'] = h2(model='UM.FS', family=metafopy.core.glm.poisson(link='sqrt'), method='FE')
    cases['irr_fam_poisson'] = h2(model='UM.FS', family='poisson')
    for lk in LINKS:
        if lk != 'cauchit':
            cases[f'plo_{lk}'] = p1('PLO', link=lk)
            cases[f'plo_{lk}_mod'] = p1('PLO', link=lk, mods='~ ablat')
        cases[f'plo_{lk}_fe'] = p1('PLO', link=lk, method='FE')
    cases['plo_probit_t'] = p1('PLO', link='probit', test='t', level=90)
    cases['plo_family_probit'] = p1('PLO', family=metafopy.core.glm.binomial(link='probit'))
    cases['pr_probit'] = p1('PR', link='probit')
    cases['pr_logit'] = p1('PR', link='logit')
    cases['pr_logit_fe'] = p1('PR', link='logit', method='FE')
    cases['pln_cloglog'] = p1('PLN', link='cloglog')
    cases['pln_logit'] = p1('PLN', link='logit')
    cases['irln_identity_fe'] = r1(link='identity', method='FE')
    cases['irln_sqrt_fe'] = r1(link='sqrt', method='FE')
    cases['irln_family_sqrt_fe'] = r1(family=metafopy.core.glm.poisson(link='sqrt'), method='FE')
    return cases


CASES = build()
