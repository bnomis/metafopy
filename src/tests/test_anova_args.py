"""p_adjust and anova(adjust=, refit=) against R (src/tests/r/generate_anova_args.R)."""

import typing

import numpy
import pytest
import refutil

import metafopy
import metafopy.core.padjust
import metafopy.errors

REF = refutil.load_phase7('anova_args')
# rma.mv fits come from an optimiser, so the fit statistics agree with R to its tolerance
OPTIM = 1e-6


@pytest.mark.parametrize('method', metafopy.core.padjust.P_ADJUST_METHODS)
def test_p_adjust(method: str) -> None:
    exp = REF['padjust'][method]
    pa = metafopy.core.padjust.p_adjust
    refutil.close(pa([0.01, 0.04, 0.03, 0.005, 0.2, 0.04, 0.8, 0.0001], method), exp['p1'], f'{method} p1')
    refutil.close(pa([0.02, numpy.nan, 0.5, 0.001, 0.02], method), exp['p2'], f'{method} p2')
    refutil.close(pa([0.03, 0.04], method), exp['p3'], f'{method} p3')
    refutil.close(pa([0.03, 0.04], method, n=5), exp['n'], f'{method} n')


def test_match_adjust() -> None:
    assert metafopy.core.padjust.match_adjust(True) == 'bonferroni'
    assert metafopy.core.padjust.match_adjust(False) == 'none'
    assert metafopy.core.padjust.match_adjust('bonf') == 'bonferroni'
    with pytest.raises(metafopy.errors.InputError):
        metafopy.core.padjust.match_adjust('ho')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.core.padjust.match_adjust('nosuch')


@pytest.fixture(scope='module')
def uni() -> tuple[metafopy.RMAResult, metafopy.RMAResult]:
    d = refutil.bcg()
    return metafopy.rma_uni('yi', 'vi', mods='~ ablat + year + alloc', data=d), metafopy.rma_uni('yi', 'vi', data=d)


def qmp(results: typing.Any) -> list[float]:
    return [r.QMp for r in results]


def test_anova_uni_adjust(uni: tuple[metafopy.RMAResult, metafopy.RMAResult]) -> None:
    res, _ = uni
    refutil.close(qmp(metafopy.anova(res, btt=[1, 2, [3, 4]], adjust='holm')), REF['uni_list_holm'], 'list holm')
    refutil.close(qmp(metafopy.anova(res, btt=[1, 2, [3, 4]], adjust=True)), REF['uni_list_bonf'], 'list bonferroni')
    # auditme: accept[literal-duplication] the moderators are named as in the R script
    refutil.close(qmp(metafopy.anova(res, btt=['ablat', 'year'], adjust='BH')), REF['uni_names_BH'], 'names BH')
    refutil.close(metafopy.anova(res, btt=[1, 2], adjust='holm').QMp, REF['uni_single'], 'single')
    refutil.close(metafopy.anova(res, btt=1, rhs=-0.02).QM, REF['uni_rhs'], 'rhs')
    L = numpy.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, -1]], dtype=float)
    refutil.close(metafopy.anova(res, X=L, adjust='hommel').pval, REF['uni_X_hommel'], 'X hommel')
    refutil.close(metafopy.anova(res, X=L, adjust='BY').pval, REF['uni_X_BY'], 'X BY')


def test_anova_uni_refit(uni: tuple[metafopy.RMAResult, metafopy.RMAResult]) -> None:
    res, res0 = uni
    lr = metafopy.anova(res, res0, refit=True)
    exp = REF['uni_refit']
    assert isinstance(lr, metafopy.LRTResult)
    refutil.close(lr.LRT, exp['LRT'], 'LRT', rtol=1e-6)
    refutil.close(lr.pval, exp['pval'], 'pval', rtol=1e-6)
    refutil.close(lr.fit_stats_full, exp['fit_f'], 'fit_f', rtol=1e-6)
    refutil.close(lr.fit_stats_reduced, exp['fit_r'], 'fit_r', rtol=1e-6)
    refutil.close([lr.tau2_full, lr.tau2_reduced, lr.R2], [exp['tau2_f'], exp['tau2_r'], exp['R2']], 'tau2/R2', rtol=1e-6)
    assert (lr.df_full, lr.df_reduced) == (exp['parms_f'], exp['parms_r'])
    with pytest.warns(UserWarning, match='refit=True'):
        metafopy.anova(res, res0)


def test_anova_mv() -> None:
    d = metafopy.datasets.load('konstantopoulos2011')
    mv = metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=d)
    mv0 = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=d)
    lr = metafopy.anova(mv, mv0, refit=True)
    exp = REF['mv_refit']
    refutil.close([lr.LRT, lr.pval], [exp['LRT'], exp['pval']], 'mv LRT', rtol=1e-4)
    refutil.close(lr.fit_stats_full, exp['fit_f'], 'mv fit_f', rtol=OPTIM)
    refutil.close(lr.fit_stats_reduced, exp['fit_r'], 'mv fit_r', rtol=OPTIM)
    X = numpy.array([[1, 0], [1, 10], [0, 1]], dtype=float)
    refutil.close(metafopy.anova(mv, X=X, adjust='holm').pval, REF['mv_X_holm'], 'mv X holm', rtol=1e-5)
    refutil.close(qmp(metafopy.anova(mv, btt=[[0], [1]], adjust='hochberg')), REF['mv_list'], 'mv list', rtol=1e-5)
