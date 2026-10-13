"""rma_mh and rma_peto (fits, printing, weights, residuals, leave1out, cumul, confint) checked against R metafor 5.2-1."""

import json
import pathlib
import typing

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.transf

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase3' / 'mh_peto.json').read_text(encoding='utf-8'))
RTOL = 1e-8

type Fit = typing.Callable[[], metafopy.MHResult | metafopy.PetoResult]


def bcg_data() -> pandas.DataFrame:
    d = metafopy.datasets.load('bcg')
    d['label'] = d['author'] + ' ' + d['year'].astype(str)
    return d


def bcg_na_data() -> pandas.DataFrame:
    d = metafopy.datasets.load('bcg')
    d.loc[2, 'tpos'] = numpy.nan
    return d


def collins_data() -> pandas.DataFrame:
    d = metafopy.datasets.load('collins1985a')
    d['label'] = d['trial'].astype(str) + ' ' + d['year'].astype(str)
    return d


BCG: dict[str, typing.Any] = dict(ai='tpos', bi='tneg', ci='cpos', di='cneg')
NW: dict[str, typing.Any] = dict(ai='ai', n1i='n1i', ci='ci', n2i='n2i')
COL: dict[str, typing.Any] = dict(ai='b.xci', n1i='nci', ci='b.xti', n2i='nti')
RATE: dict[str, typing.Any] = dict(x1i='x1i', x2i='x2i', t1i='t1i', t2i='t2i')
X = numpy.array([[5, 95, 8, 92], [3, 47, 6, 44], [0, 20, 0, 20], [8, 92, 3, 97]], dtype=float)


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def nw() -> pandas.DataFrame:
    return metafopy.datasets.load('nielweise2007')


def yusuf() -> pandas.DataFrame:
    return metafopy.datasets.load('yusuf1985')


def yusuf_table6() -> numpy.ndarray[tuple[int], numpy.dtype[numpy.bool_]]:
    return (yusuf()['table'] == '6').to_numpy()


def bcg_subset(select: typing.Callable[[pandas.Series], pandas.Series]) -> pandas.DataFrame:
    d = bcg_data()
    return d[select(d['alloc'])].reset_index(drop=True)


# auditme: accept[non-alphabetical-dict-keys] the order follows metafor's documented option order or the R fixture generator
CASES: dict[str, Fit] = {
    'mh_bcg_or': lambda: metafopy.rma_mh(**BCG, data=bcg_data(), slab='label'),
    'mh_bcg_or_nocorrect': lambda: metafopy.rma_mh(**BCG, data=bcg_data(), correct=False, level=90),
    'mh_bcg_rr': lambda: metafopy.rma_mh(**BCG, data=bcg_data(), measure='RR'),
    'mh_bcg_rd': lambda: metafopy.rma_mh(**BCG, data=bcg_data(), measure='RD'),
    'mh_bcg_subset': lambda: metafopy.rma_mh(**BCG, data=bcg_data(), subset=(bcg_data()['alloc'] == 'random').to_numpy()),
    'mh_nw_or': lambda: metafopy.rma_mh(**NW, data=nw()),
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    'mh_nw_or_addto': lambda: metafopy.rma_mh(**NW, data=nw(), add=(0.5, 0.5), to=('only0', 'all')),
    'mh_nw_or_if0all': lambda: metafopy.rma_mh(**NW, data=nw(), add=(1 / 2, 1 / 4), to=('if0all', 'only0')),
    'mh_nw_or_nodrop': lambda: metafopy.rma_mh(**NW, data=nw(), drop00=False),
    'mh_nw_or_drop2': lambda: metafopy.rma_mh(**NW, data=nw(), drop00=(True, True)),
    'mh_nw_rr': lambda: metafopy.rma_mh(**NW, data=nw(), measure='RR'),
    'mh_nw_rd': lambda: metafopy.rma_mh(**NW, data=nw(), measure='RD'),
    'mh_col_or': lambda: metafopy.rma_mh(**COL, data=collins_data(), slab='label'),
    'mh_irr': lambda: metafopy.rma_mh(**RATE, data=metafopy.datasets.load('hart1999'), measure='IRR'),
    'mh_ird': lambda: metafopy.rma_mh(**RATE, data=metafopy.datasets.load('hart1999'), measure='IRD'),
    'mh_bcg_na': lambda: metafopy.rma_mh(**BCG, data=bcg_na_data()),
    'mh_yusuf': lambda: metafopy.rma_mh(**NW, data=yusuf(), subset=yusuf_table6()),
    'mh_vectors': lambda: metafopy.rma_mh(ai=X[:, 0], bi=X[:, 1], ci=X[:, 2], di=X[:, 3]),
    'mh_k1': lambda: metafopy.rma_mh(ai=5, bi=95, ci=8, di=92),
    'peto_bcg': lambda: metafopy.rma_peto(**BCG, data=bcg_data(), slab='label'),
    'peto_bcg_level': lambda: metafopy.rma_peto(**BCG, data=bcg_data(), level=90, subset=(bcg_data()['alloc'] != 'random').to_numpy()),
    'peto_nw': lambda: metafopy.rma_peto(**NW, data=nw()),
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    'peto_nw_addto': lambda: metafopy.rma_peto(**NW, data=nw(), add=(0.5, 0.5), to=('only0', 'all')),
    'peto_nw_nodrop': lambda: metafopy.rma_peto(**NW, data=nw(), drop00=False),
    'peto_nw_drop2': lambda: metafopy.rma_peto(**NW, data=nw(), drop00=(True, True)),
    'peto_col': lambda: metafopy.rma_peto(**COL, data=collins_data(), slab='label'),
    'peto_yusuf': lambda: metafopy.rma_peto(**NW, data=yusuf(), subset=yusuf_table6()),
    'peto_bcg_na': lambda: metafopy.rma_peto(**BCG, data=bcg_na_data()),
}


def close(actual: typing.Any, expected: typing.Any, what: str) -> None:
    """The fixtures cannot hold infinities (jsonlite writes them as null), so non-finite actual values are compared as NaN."""
    values = numpy.atleast_1d(numpy.asarray(actual, dtype=float))
    refutil.close(numpy.where(numpy.isfinite(values), values, numpy.nan), expected, what, rtol=RTOL)


def labels(value: typing.Any) -> list[str]:
    return [str(v) for v in refutil.as_list(value)]


def order_of(k_all: int) -> list[int]:
    """The 0-based version of the ordering used in generate_mh_peto.R."""
    if k_all > 2:
        return [1, 0, *range(k_all - 1, 1, -1)]
    return list(range(k_all))


def check_table(tab: metafopy.Leave1outResult, ref: dict[str, typing.Any], what: str) -> None:
    for key, attr in (
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        ('estimate', 'estimate'),
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        ('se', 'se'),
        ('zval', 'stat'),
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        ('pval', 'pval'),
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        ('ci.lb', 'ci_lb'),
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        ('ci.ub', 'ci_ub'),
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        ('Q', 'Q'),
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        ('Qp', 'Qp'),
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        ('I2', 'I2'),
        # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
        ('H2', 'H2'),
    ):
        if key in ref:
            close(getattr(tab, attr), ref[key], f'{what}.{key}')
    assert list(tab.slab) == labels(ref['slab']), what
    assert refutil.same_print(str(tab), ref['print']), f'{what}\n{tab}\n{ref["print"]}'


@pytest.mark.parametrize('name', list(CASES))
def test_fit(name: str) -> None:
    ref = REF[name]
    res = CASES[name]()
    close(res.beta, ref['beta'], 'beta')
    close(res.se, ref['se'], 'se')
    close(res.zval, ref['zval'], 'zval')
    close(res.pval, ref['pval'], 'pval')
    close(res.ci_lb, ref['ci_lb'], 'ci_lb')
    close(res.ci_ub, ref['ci_ub'], 'ci_ub')
    close(res.QE, ref['QE'], 'QE')
    close(res.QEp, ref['QEp'], 'QEp')
    close(res.I2, ref['I2'], 'I2')
    close(res.H2, ref['H2'], 'H2')
    assert (res.k, res.k_f, res.k_yi, res.k_all) == (ref['k'], ref['k_f'], ref['k_yi'], ref['k_all'])
    assert res.k_pos == ref['k_pos']
    close(res.yi, ref['yi'], 'yi')
    close(res.vi, ref['vi'], 'vi')
    close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], 'fit ML')
    close(res.fit_stats_reml.as_tuple(), ref['fit_reml'], 'fit REML')
    close(res.weights_(), ref['weights'], 'weights')
    if ref['weights_matrix_diag'] is not None:
        close(numpy.diag(res.weights_(type='matrix')), ref['weights_matrix_diag'], 'weights matrix')
    assert list(res.slab) == labels(ref['slab'])
    for key, attr in (('co', 'CO'), ('cop', 'COp'), ('mh', 'MH'), ('mhp', 'MHp'), ('bd', 'BD'), ('bdp', 'BDp'), ('ta', 'TA'), ('tap', 'TAp')):
        if key in ref:
            close(getattr(res, attr), ref[key], key)
    assert refutil.same_print(str(res), ref['print']), f'\n{res}\n{ref["print"]}'
    assert refutil.same_print(res.summary(showfit=True), ref['print_showfit']), f'\n{res.summary(showfit=True)}\n{ref["print_showfit"]}'


@pytest.mark.parametrize('name', list(CASES))
def test_diagnostics(name: str) -> None:
    ref = REF[name]
    res = CASES[name]()
    ci = metafopy.confint(res).fixed
    assert ci is not None
    close(ci.to_numpy().ravel('F'), ref['confint'], 'confint')
    for which, fn in (('rstandard', metafopy.rstandard), ('rstudent', metafopy.rstudent)):
        out = fn(res)
        close(out.resid, ref[which]['resid'], f'{which}.resid')
        close(out.se, ref[which]['se'], f'{which}.se')
        close(out.z, ref[which]['z'], f'{which}.z')
        assert list(out.slab) == labels(ref[which]['slab'])
    if res.k > 1:
        check_table(metafopy.leave1out(res), ref['leave1out'], 'leave1out')
        check_table(metafopy.leave1out(res, transf=True), ref['leave1out_exp'], 'leave1out_exp')
    check_table(metafopy.cumul(res), ref['cumul'], 'cumul')
    order = order_of(res.k_all)
    check_table(metafopy.cumul(res, order=order), ref['cumul_ord'], 'cumul_ord')
    check_table(metafopy.cumul(res, order=order, decreasing=True), ref['cumul_ord_dec'], 'cumul_ord_dec')


def test_errors() -> None:
    with pytest.raises(metafopy.errors.InputError):
        metafopy.rma_mh(ai=1, bi=2, ci=3, di=4, measure='SMD')
    with pytest.raises(metafopy.errors.InputError, match='never occurred'):
        metafopy.rma_peto(ai=[0, 0], bi=[5, 6], ci=[0, 0], di=[5, 6])
