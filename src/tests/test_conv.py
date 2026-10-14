"""conv_2x2(), conv_delta() and conv_wald() against R metafor 5.2-1 (fixtures from tests/r/generate_data_helpers.R, mostly the help page examples).

conv_delta() differentiates numerically (numDeriv's Richardson extrapolation, ported step by step), so its sampling variances agree with R
to rounding error of the difference quotients and are compared at rtol 1e-8 like the closed-form values.
"""

import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.transf

C2 = refutil.load_phase7('conv_2x2')
CD = refutil.load_phase7('conv_delta_wald')
# auditme: accept[non-alphabetical-dict-keys] order follows the argument order of the R function
KW: dict[str, typing.Any] = {'ri': 'phi', 'ori': 'oddsratio', 'x2i': 'chisq', 'ni': 'ni', 'n1i': 'n1i', 'n2i': 'n2i'}


@pytest.mark.parametrize(
    ('key', 'source', 'kwargs'),
    [
        ('data1', 'input1', {}),
        ('data1_noappend', 'input1', {'append': False}),
        ('data1_names', 'input1', {'var_names': ['a', 'b', 'c', 'd']}),
        ('data1_all', 'input1', {'replace': 'all'}),
        ('data1_include', 'input1', {'include': [True, True, False, True]}),
        ('data10', 'input10', {}),
    ],
)
def test_conv_2x2_reconstruction(key: str, source: str, kwargs: dict[str, typing.Any]) -> None:
    out = metafopy.conv_2x2(**KW, data=refutil.r_frame(C2[source]), **kwargs)
    refutil.assert_frame(out, C2[key], key)


def test_conv_2x2_marginals() -> None:
    refutil.assert_frame(metafopy.conv_2x2(ri=0.54, ni=400, n1i=200, n2i=248), C2['true_marg'], 'true')
    refutil.assert_frame(metafopy.conv_2x2(ri=0.54, ni=400, n1i=200, n2i=200), C2['guess_marg'], 'guessed')
    with pytest.warns(UserWarning, match='negative cell frequencies'):
        refutil.assert_frame(metafopy.conv_2x2(ri=0.54, ni=400, n1i=200, n2i=50), C2['incompat'], 'incompatible')


def test_conv_2x2_chisq() -> None:
    chisq = C2['chisq_sign']
    refutil.assert_frame(metafopy.conv_2x2(x2i=-chisq, ni=200, n1i=100, n2i=100), C2['neg_chisq'], 'negative')
    refutil.assert_frame(metafopy.conv_2x2(x2i=chisq, ni=200, n1i=100, n2i=100), C2['pos_chisq'], 'positive')
    refutil.assert_frame(metafopy.conv_2x2(x2i=C2['chisq_yates'], ni=72, n1i=42, n2i=40), C2['yates'], 'yates')
    refutil.assert_frame(metafopy.conv_2x2(x2i=C2['chisq_noyates'], ni=72, n1i=42, n2i=40), C2['noyates_correct'], 'correct')
    refutil.assert_frame(metafopy.conv_2x2(x2i=C2['chisq_noyates'], ni=72, n1i=42, n2i=40, correct=False), C2['noyates'], 'uncorrected')


def test_conv_2x2_diagnostic() -> None:
    g = C2['diag']
    refutil.assert_frame(metafopy.conv_2x2(ori=g['dor'], n1i=g['pos'], n2i=g['cases'], ni=g['n']), C2['dor'], 'dor')
    refutil.assert_frame(metafopy.conv_2x2(sens=g['sens'], spec=g['spec'], ppv=g['ppv'], npv=g['npv'], ni=g['n']), C2['sspn'], 'all four')
    refutil.assert_frame(metafopy.conv_2x2(sens=g['sens'], spec=g['spec'], ppv=g['ppv'], ni=g['n']), C2['ssp'], 'no npv')
    refutil.assert_frame(metafopy.conv_2x2(sens=g['sens'], spec=g['spec'], npv=g['npv'], ni=g['n']), C2['ssn'], 'no ppv')
    refutil.assert_frame(metafopy.conv_2x2(sens=g['sens'], ppv=g['ppv'], npv=g['npv'], ni=g['n']), C2['spn'], 'no spec')
    refutil.assert_frame(metafopy.conv_2x2(spec=g['spec'], ppv=g['ppv'], npv=g['npv'], ni=g['n']), C2['xpn'], 'no sens')
    two: dict[str, typing.Any] = {'spec': [g['spec']] * 2, 'ppv': [g['ppv']] * 2, 'ni': [g['n']] * 2}
    refutil.assert_frame(metafopy.conv_2x2(sens=[g['sens'], 1], **two), C2['drop01'], 'drop01')
    refutil.assert_frame(metafopy.conv_2x2(sens=[g['sens'], 1], drop01=False, **two), C2['drop01_false'], 'drop01=False')


@pytest.mark.parametrize(('method', 'key'), [('optim', 'sspn_rounded'), ('simple', 'sspn_rounded_simple'), ('search', 'sspn_rounded_search')])
def test_conv_2x2_rounded_statistics(method: typing.Literal['simple', 'optim', 'search'], key: str) -> None:
    g, r = C2['diag'], C2['diag_rounded']
    if method == 'optim':
        refutil.assert_frame(metafopy.conv_2x2(ori=r['dor'], n1i=g['pos'], n2i=g['cases'], ni=g['n']), C2['dor_rounded'], 'dor')
    out = metafopy.conv_2x2(sens=r['sens'], spec=r['spec'], ppv=r['ppv'], npv=r['npv'], ni=g['n'], method=method)
    refutil.assert_frame(out, C2[key], key)


def test_conv_2x2_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='marginal counts are larger'):
        metafopy.conv_2x2(ri=0.2, ni=10, n1i=20, n2i=5)
    with pytest.raises(metafopy.errors.InputError, match='phi coefficients'):
        metafopy.conv_2x2(ri=1.2, ni=10, n1i=5, n2i=5)
    with pytest.raises(metafopy.errors.InputError, match='same length'):
        metafopy.conv_2x2(ri=[0.2, 0.3], ni=[10, 10, 10], n1i=5, n2i=5)


@pytest.fixture(scope='module')
def proportions() -> pandas.DataFrame:
    return metafopy.escalc('PR', xi=[5, 12], ni=[40, 80])


def test_conv_delta_logit(proportions: pandas.DataFrame) -> None:
    dat = proportions
    plo = metafopy.escalc('PLO', xi=[5, 12], ni=[40, 80])
    refutil.assert_frame(plo, CD['plo'], 'PLO')
    out = metafopy.conv_delta(dat['yi'], dat['vi'], transf=metafopy.transf.transf_logit)
    refutil.assert_frame(out, CD['logit_vec'], 'vectors')
    # the delta method reproduces the analytic variances of PLO
    numpy.testing.assert_allclose(out['vi'], plo['vi'], rtol=1e-7)
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    refutil.assert_frame(metafopy.conv_delta('yi', 'vi', data=dat, transf=metafopy.transf.transf_logit, var_names=['yi.t', 'vi.t']), CD['logit_names'], 'names')
    refutil.assert_frame(metafopy.conv_delta('yi', 'vi', data=dat, transf=metafopy.transf.transf_logit, replace='all'), CD['logit_all'], 'all')
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    out = metafopy.conv_delta('yi', 'vi', data=dat, transf=metafopy.transf.transf_logit, var_names=['yi.t', 'vi.t'], append=False)
    refutil.assert_frame(out, CD['logit_noappend'], 'no append')
    refutil.assert_frame(metafopy.conv_delta(dat['yi'], dat['vi'], transf=metafopy.transf.transf_logit, include=[False, True]), CD['logit_include'], 'include')
    out = metafopy.conv_delta(dat['yi'], dat['vi'], transf=metafopy.transf.transf_logit, ni=[40, 80])
    refutil.close(out.attrs['ni'], CD['logit_ni'], 'ni')


def test_conv_delta_smd() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dat = metafopy.escalc(
            'SMD', m1i='m1i', sd1i='sd1i', n1i='n1i', m2i='m2i', sd2i='sd2i', n2i='n2i', data=metafopy.datasets.load('normand1999'), correct=False
        )
    refutil.assert_frame(dat, CD['smd'], 'SMD')
    refutil.assert_frame(metafopy.conv_delta('yi', 'vi', data=dat, transf=metafopy.transf.transf_dtolnor_norm, replace='all'), CD['dtolnor'], 'dtolnor')
    out = metafopy.conv_delta('yi', 'vi', data=dat, transf=metafopy.transf.transf_dtorpb, n1i='n1i', n2i='n2i', replace='all')
    refutil.assert_frame(out, CD['dtorpb'], 'dtorpb')


def test_smd_preparation_example() -> None:
    dat = refutil.r_frame(CD['smd_step2'])
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    dat = metafopy.conv_wald(out='or', ci_lb='lower', ci_ub='upper', data=dat, transf=numpy.log, var_names=['lnor', 'vlnor'])
    refutil.assert_frame(dat, CD['smd_step3'], 'conv_wald')
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    dat = metafopy.conv_delta('lnor', 'vlnor', data=dat, transf=metafopy.transf.transf_lnortod_norm, var_names=['yi', 'vi'])
    refutil.assert_frame(dat, CD['smd_step4'], 'conv_delta lnor')
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    dat = dat.drop(columns=['lnor', 'vlnor'])
    dat = metafopy.escalc('COR', ri='corr', ni='ntot', data=dat, var_names=['ri', 'vri'])
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    dat = metafopy.conv_delta('ri', 'vri', data=dat, transf=metafopy.transf.transf_rtod, var_names=['yi', 'vi'])
    refutil.assert_frame(dat, CD['smd_step5'], 'conv_delta ri')


def test_correlation_preparation_example() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dat = metafopy.escalc('COR', ri='ri', ni='ni', ti='tval', data=refutil.r_frame(CD['cor_input']))
        dat = metafopy.escalc('RBIS', m1i='m1i', sd1i='sd1i', n1i='n1i', m2i='m2i', sd2i='sd2i', n2i='n2i', data=dat, replace=False)
    dat = metafopy.conv_2x2(ri='phi', ni='ni', n1i='n1i', n2i='n2i', data=dat)
    refutil.assert_frame(dat, CD['cor_step3'], 'conv_2x2')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dat = metafopy.escalc('RTET', ai='ai', bi='bi', ci='ci', di='di', data=dat, replace=False)
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    dat = metafopy.conv_wald(out='or', ci_lb='lower', ci_ub='upper', data=dat, transf=numpy.log, var_names=['lnor', 'vlnor'])
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    dat = metafopy.conv_delta('lnor', 'vlnor', data=dat, transf=metafopy.transf.transf_lnortortet_pearson, var_names=['yi', 'vi'])
    refutil.assert_frame(dat, CD['cor_step6'], 'conv_delta')


def test_conv_delta_failures_give_missing_values() -> None:
    def fail(x: float) -> float:
        raise ValueError('no')

    out = metafopy.conv_delta([0.1, 0.2], [0.01, 0.02], transf=fail)
    assert out['yi'].isna().all()
    assert out['vi'].isna().all()
    with pytest.raises(metafopy.errors.InputError, match="'transf'"):
        metafopy.conv_delta([0.1], [0.01])


def test_conv_wald() -> None:
    dat = pandas.DataFrame({'or': [1.37, 1.89], 'or.lb': [1.03, 1.60], 'or.ub': [1.82, 2.23]})
    kw: dict[str, typing.Any] = {'out': 'or', 'ci_lb': 'or.lb', 'ci_ub': 'or.ub', 'data': dat, 'transf': numpy.log}
    refutil.assert_frame(metafopy.conv_wald(**kw), CD['wald_simple'], 'simple')
    refutil.assert_frame(metafopy.conv_wald(**kw, level=[95, 90]), CD['wald_simple_level'], 'level')
    # auditme: accept[literal-duplication] each test case spells out its own R arguments
    refutil.assert_frame(metafopy.conv_wald(**kw, append=False, var_names=['lor', 'vlor']), CD['wald_noappend'], 'no append')
    refutil.assert_frame(metafopy.conv_wald(out=numpy.log([1.37, 1.89]), ci_ub=numpy.log([1.82, 2.23])), CD['wald_ub_only'], 'upper bound only')
    refutil.assert_frame(metafopy.conv_wald(out=[0.3, -0.2], zval=[2.1, -1.5]), CD['wald_zval'], 'zval')
    with pytest.warns(UserWarning, match='halfway'):
        out = metafopy.conv_wald(out=[1.37, 1.0], ci_lb=[1.03, 1.60], ci_ub=[1.82, 2.23], transf=numpy.log)
    refutil.assert_frame(out, CD['wald_check'], 'check')
    with pytest.raises(metafopy.errors.InputError, match='p-values'):
        metafopy.conv_wald(out=[1.0], pval=[1.2])


def test_conv_wald_bcg() -> None:
    dat = refutil.r_frame(CD['bcg_input'])
    dat = metafopy.escalc('OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat)
    dat = metafopy.conv_wald(out='or', ci_lb='ci.lb', ci_ub='ci.ub', pval='pval', n='ntot', data=dat, transf=numpy.log)
    refutil.assert_frame(dat, CD['bcg'], 'bcg')
    refutil.close(dat.attrs['ni'], CD['bcg_ni'], 'ni')
    assert dat.attrs['measure'] == 'OR'
