"""``digits`` of fits, predictions, tests and robust fits, the significance stars of the printouts, the printout and data frames of ``anova(X=)``,
the result objects of list ``btt`` and of ``confint()`` of several components, and ``progbar``; the printed tests of ``anova(X=)`` are checked
against the printouts of R metafor 5.2-1 stored by tests/r/generate_audit4_robust.R."""

import collections.abc
import typing

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.methods.anova
import metafopy.methods.confint

REF: dict[str, typing.Any] = refutil.load_phase12('audit4_robust')


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def res1(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)


@pytest.fixture(scope='module')
def res2(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat)


@pytest.fixture(scope='module')
def mv_fit(dat: pandas.DataFrame) -> typing.Any:
    kon = metafopy.datasets.load('konstantopoulos2011')
    return metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=kon)


@pytest.fixture
def reset_options() -> collections.abc.Iterator[None]:
    yield
    metafopy.setmfopt(digits=None, signif_stars=None)


def estimate_cells(text: str) -> list[str]:
    """The numbers of the ablat row of a coefficient table."""
    line = next(line for line in text.split('\n') if line.startswith('ablat'))
    return line.split()


# -- digits ---------------------------------------------------------------------------------------------------------------------------------------


def test_digits_stored_in_fit(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, digits=2)
    assert res.digits == 2
    assert estimate_cells(res.summary())[1] == '-0.03'
    assert estimate_cells(str(res))[1] == '-0.03'
    default = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)
    assert default.digits == 4
    assert estimate_cells(default.summary())[1] == '-0.0291'


def test_digits_precedence(dat: pandas.DataFrame, reset_options: None) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, digits=2)
    assert estimate_cells(res.summary(digits=3))[1] == '-0.029'
    metafopy.setmfopt(digits=1)
    assert estimate_cells(res.summary())[1] == '0.0'
    assert estimate_cells(res.summary(digits=3))[1] == '-0.029'


def test_digits_mapping_uses_est(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, digits={'est': 3, 'se': 1})
    assert res.digits == 3


def test_digits_location_scale(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', scale='~ ablat', data=dat, digits=2)
    assert res.digits == 2
    assert 'tau^2' not in res.summary() or '0.' in res.summary()
    assert estimate_cells(res.summary())[1] == '-0.03'


def test_digits_rma_mv(mv_fit: typing.Any) -> None:
    kon = metafopy.datasets.load('konstantopoulos2011')
    res = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon, digits=2)
    assert res.digits == 2
    assert mv_fit.digits == 4
    assert '0.18  0.08  2.18  0.03' in res.summary()
    assert '0.1844' not in res.summary()
    assert metafopy.update(res).digits == 2


def test_digits_predict(res1: metafopy.RMAResult) -> None:
    pred = metafopy.predict(res1, newmods=[10, 20], digits=2)
    assert pred.digits == 2
    assert '-0.04' in str(pred)
    assert metafopy.predict(res1, newmods=[10, 20]).digits == 4
    assert '-0.0396' not in str(pred)
    assert metafopy.predict(res1, newmods=[10, 20]).to_frame().shape == (2, 6)


def test_digits_predict_follows_model(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, digits=3)
    assert metafopy.predict(res).digits == 3
    assert metafopy.predict(res, digits=1).digits == 1


def test_predict_print_layout(res1: metafopy.RMAResult) -> None:
    text = str(metafopy.predict(res1, newmods=[10]))
    assert refutil.r_print(text).strip().split('\n')[0].split() == ['pred', 'se', 'ci.lb', 'ci.ub', 'pi.lb', 'pi.ub']


def test_digits_anova(res1: metafopy.RMAResult) -> None:
    out = metafopy.anova(res1, digits=2)
    assert isinstance(out, metafopy.methods.anova.WaldResult)
    assert out.digits == 2
    assert str(out) == 'Test of Moderators (coefficient 2):\nQM(df = 1) = 16.36, p-val < .01'
    assert metafopy.anova(res1, X=[1, 30.52], digits=1).hyp == ('intrcpt + 30.5*ablat = 0',)
    # as in R, gsub('1*', '', fixed = TRUE) also removes the '1*' of a coefficient such as 30.1*
    assert metafopy.anova(res1, X=[1, 30.12], digits=1).hyp == ('intrcpt + 30.ablat = 0',)
    assert metafopy.anova(res1).digits == 4


def test_digits_anova_lrt(dat: pandas.DataFrame) -> None:
    full = metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat, method='ML')
    red = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, method='ML')
    out = metafopy.anova(full, red, digits=2)
    assert out.digits == 2
    assert '0.' in str(out)
    assert len(str(out).split('\n')[1].split()[2].split('.')[1]) == 2


def test_digits_robust(res1: metafopy.RMAResult) -> None:
    rb = metafopy.robust(res1, 'alloc', digits=2)
    assert rb.digits == 2
    assert estimate_cells(rb.summary())[1].startswith('-0.03')
    assert metafopy.robust(res1, 'alloc').digits == 4
    assert metafopy.robust(metafopy.rma('yi', 'vi', mods='~ ablat', data=res1.data.data, digits=3), 'alloc').digits == 3


def test_digits_robust_mv(mv_fit: typing.Any) -> None:
    assert metafopy.robust(mv_fit, 'district', digits=2).digits == 2


# -- significance stars -----------------------------------------------------------------------------------------------------------------------------


def test_signif_stars_uni(res1: metafopy.RMAResult) -> None:
    full = res1.summary()
    assert full.rstrip().endswith("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    assert estimate_cells(full)[-1] == '***'
    bare = res1.summary(signif_stars=False)
    assert '***' not in bare
    assert 'Signif. codes' not in bare
    assert '---' not in bare
    assert bare.rstrip().endswith('-0.0150')
    assert 'Signif. codes' in res1.summary(signif_stars=False, signif_legend=True)
    assert 'Signif. codes' not in res1.summary(signif_legend=False)
    assert estimate_cells(res1.summary(signif_legend=False))[-1] == '***'


def test_signif_stars_option(res1: metafopy.RMAResult, reset_options: None) -> None:
    metafopy.setmfopt(signif_stars=False)
    assert 'Signif. codes' not in res1.summary()
    assert '***' not in str(res1)
    assert '***' in res1.summary(signif_stars=True)
    assert metafopy.getmfopt('signif_stars') is False
    with pytest.raises(metafopy.errors.InputError):
        metafopy.setmfopt(signif_stars='yes')


def test_signif_stars_robust_keeps_footnote(res1: metafopy.RMAResult) -> None:
    rb = metafopy.robust(res1, 'alloc')
    text = rb.summary(signif_legend=False)
    assert 'Signif. codes' not in text
    assert text.rstrip().endswith('df: residual method)')
    assert '---' in text
    both = rb.summary()
    assert both.index('Signif. codes') < both.index('1) results based on cluster-robust inference')
    assert 'Signif. codes' not in rb.summary(signif_stars=False)


def test_signif_stars_mv(mv_fit: typing.Any) -> None:
    assert 'Signif. codes' in mv_fit.summary()
    bare = mv_fit.summary(signif_stars=False)
    assert 'Signif. codes' not in bare
    assert bare.rstrip().endswith('0.0238')
    assert ' *' not in bare.split('Model Results:')[1]


def test_signif_stars_location_scale(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', scale='~ ablat', data=dat)
    assert 'Signif. codes' in res.summary()
    bare = res.summary(signif_stars=False)
    assert 'Signif. codes' not in bare
    assert '*' not in bare.split('Model Results (Location):')[1].replace('***', '')


def test_signif_stars_anova_x(res1: metafopy.RMAResult) -> None:
    out = metafopy.anova(res1, X=[1, 30])
    assert out.summary().split('Results:')[1].split('\n')[2].endswith('*')
    assert '*' not in out.summary(signif_stars=False).split('Results:')[1].split('Test of')[0]


# -- anova(X=) ------------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def club(res1: metafopy.RMAResult, res2: metafopy.RMAResult) -> dict[str, typing.Any]:
    return {'rb1': metafopy.robust(res1, 'alloc', club_sandwich=True), 'rb2': metafopy.robust(res2, 'alloc', club_sandwich=True)}


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
X_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'x_vec': ('rb1', {'X': [1, 30]}),
    'x_mat': ('rb1', {'X': [[1, 30], [1, 40]]}),
    'x_noint': ('rb1', {'X': [30]}),
    'x_rank': ('rb1', {'X': [[1, 30], [2, 60]]}),
    'l_vec': ('rb1', {'L': [1, 30]}),
    'adjust_x': ('rb1', {'X': [[1, 30], [1, 40]], 'adjust': 'holm'}),
    'p3_x': ('rb2', {'X': [[1, 30, 70], [1, 10, 70]]}),
    'p3_x_short': ('rb2', {'X': [[30, 70], [10, 70]]}),
}


@pytest.mark.parametrize('name', list(X_CASES))
def test_anova_x_print_matches_r(club: dict[str, typing.Any], name: str) -> None:
    key, kwargs = X_CASES[name]
    got = metafopy.anova(club[key], **kwargs)
    assert refutil.r_print(str(got)).strip() == refutil.r_print(REF['anova_uni'][name]['print']).strip()


def test_anova_x_rank_deficient_has_no_qm_line(club: dict[str, typing.Any]) -> None:
    text = str(metafopy.anova(club['rb1'], X=[[1, 30], [2, 60]]))
    assert 'Omnibus' not in text
    assert 'QM' not in text
    assert 'NA' not in text
    assert text.split('\n')[0] == 'Hypotheses:'


def test_anova_x_rhs_hypothesis(res1: metafopy.RMAResult) -> None:
    assert metafopy.anova(res1, X=[1, 30], rhs=1).hyp == ('intrcpt + 30*ablat = 1',)
    assert metafopy.anova(res1, X=[[1, 30], [1, 40]], rhs=[1, 2]).hyp == ('intrcpt + 30*ablat = 1.0000', 'intrcpt + 40*ablat = 2.0000')
    assert metafopy.anova(res1, X=[[1, -1]]).hyp == ('intrcpt - ablat = 0',)


def test_anova_x_print_z_test(res1: metafopy.RMAResult) -> None:
    text = str(metafopy.anova(res1, X=[[1, 30], [1, 40]]))
    assert text.split('\n')[0] == 'Hypotheses:'
    assert 'zval' in text
    assert 'df' not in text.split('Results:')[1].split('\n')[1]
    assert 'Omnibus Test of Hypotheses:\nQM(df = 2) = ' in text
    one = str(metafopy.anova(res1, X=[1, 30]))
    assert one.split('\n')[0] == 'Hypothesis:'
    assert 'Test of Hypothesis:\nQM(df = 1) = ' in one


def test_anova_x_print_mv(mv_fit: typing.Any) -> None:
    got = metafopy.anova(metafopy.robust(mv_fit, 'district', club_sandwich=True), X=[[1, -5], [1, 5]])
    assert refutil.r_print(str(got)).strip() == refutil.r_print(REF['anova_mv']['x_mat']['print']).strip()


def test_wald_result_to_frame_x(club: dict[str, typing.Any]) -> None:
    got = metafopy.anova(club['rb1'], X=[[1, 30], [1, 40]])
    frame = got.to_frame()
    assert list(frame.columns) == ['hyp', 'estimate', 'se', 'tval', 'df', 'pval']
    assert list(frame.index) == ['1:', '2:']
    assert list(frame['hyp']) == ['intrcpt + 30*ablat = 0', 'intrcpt + 40*ablat = 0']
    refutil.close(frame['estimate'], REF['anova_uni']['x_mat']['Xb'], 'estimate', rtol=1e-6)
    refutil.close(frame['se'], REF['anova_uni']['x_mat']['se'], 'se', rtol=1e-6)
    refutil.close(frame['pval'], REF['anova_uni']['x_mat']['pval'], 'pval', rtol=1e-6)
    numpy.testing.assert_allclose(frame['df'], [1.83, 1.71])


def test_wald_result_to_frame_z(res1: metafopy.RMAResult) -> None:
    frame = metafopy.anova(res1, X=[1, 30]).to_frame()
    assert list(frame.columns) == ['hyp', 'estimate', 'se', 'zval', 'pval']


def test_wald_result_to_frame_btt(res1: metafopy.RMAResult, club: dict[str, typing.Any]) -> None:
    frame = metafopy.anova(res1, btt=[0, 1]).to_frame()
    assert list(frame.columns) == ['coefs', 'QM', 'df', 'pval']
    assert frame.loc[0, 'coefs'] == '1:2'
    assert frame.loc[0, 'df'] == 2
    robust = metafopy.anova(club['rb1']).to_frame()
    assert list(robust.columns) == ['coefs', 'Fval', 'df1', 'df2', 'pval']
    refutil.close(robust.loc[0, 'Fval'], REF['anova_uni']['default']['QM'], 'Fval', rtol=1e-6)
    assert robust.loc[0, 'df2'] == 1.52


# -- list btt -------------------------------------------------------------------------------------------------------------------------------------


def test_anova_btt_list_result(res2: metafopy.RMAResult) -> None:
    out = metafopy.anova(res2, btt=[[0], [1, 2]])
    assert isinstance(out, list)
    assert isinstance(out, metafopy.methods.anova.ListAnovaResult)
    assert len(out) == 2
    assert out.names == ('1', '2:3')
    assert all(isinstance(r, metafopy.methods.anova.WaldResult) for r in out)
    frame = out.to_frame()
    assert list(frame.columns) == ['coefs', 'QM', 'df', 'pval']
    assert list(frame['coefs']) == ['1', '2:3']
    lines = str(out).split('\n')
    assert lines[0].split() == ['coefs', 'QM', 'df', 'pval']
    assert lines[1].split()[:2] == ['1', '1']
    assert lines[2].split()[1] == '2:3'
    assert len(lines) == 3


def test_anova_btt_list_names_and_adjust(res2: metafopy.RMAResult) -> None:
    out = metafopy.anova(res2, btt=['ablat', 'year'], adjust='bonferroni')
    assert isinstance(out, metafopy.methods.anova.ListAnovaResult)
    assert out.names == ('ablat', 'year')
    frame = out.to_frame()
    assert list(frame.columns) == ['spec', 'coefs', 'QM', 'df', 'pval']
    plain = metafopy.anova(res2, btt=['ablat', 'year'])
    numpy.testing.assert_allclose(frame['pval'], numpy.minimum(1, 2 * numpy.array([r.QMp for r in plain])))


def test_anova_btt_list_f_test(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat, test='knha')
    out = metafopy.anova(res, btt=[[0], [1]])
    assert list(out.to_frame().columns) == ['coefs', 'Fval', 'df1', 'df2', 'pval']
    assert str(out).split('\n')[0].split() == ['coefs', 'Fval', 'df1', 'df2', 'pval']


def test_anova_btt_list_mv(mv_fit: typing.Any) -> None:
    out = metafopy.anova(mv_fit, btt=[[0], [1]], adjust='hochberg')
    assert isinstance(out, metafopy.methods.anova.ListAnovaResult)
    assert len(out) == 2
    assert 'QM' in str(out)
    assert out.to_frame().shape == (2, 4)


# -- confint of several components ------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def two_components(dat: pandas.DataFrame) -> typing.Any:
    data = dat.copy()
    data['id'] = range(1, 14)
    return metafopy.rma_mv('yi', 'vi', random=['~ 1 | alloc', '~ 1 | id'], data=data)


def test_confint_mv_list_result(two_components: typing.Any) -> None:
    out = metafopy.confint(two_components)
    assert isinstance(out, list)
    assert isinstance(out, metafopy.methods.confint.ConfintList)
    assert len(out) == 2
    assert all(isinstance(r, metafopy.methods.confint.ConfintResult) for r in out)
    text = str(out)
    assert text.count('estimate') == 2
    assert 'sigma^2.1' in text
    assert 'sigma.2' in text
    assert '\n\n' in text
    frame = out.to_frame()
    assert list(frame.index) == ['sigma^2.1', 'sigma.1', 'sigma^2.2', 'sigma.2']
    assert list(frame.columns) == ['estimate', 'ci.lb', 'ci.ub']
    assert '0.0132' in text
    assert '0.0132' not in out.summary(digits=2)
    assert '0.01' in out.summary(digits=2)


def test_confint_result_to_frame(res1: metafopy.RMAResult) -> None:
    out = metafopy.confint(res1, fixed=True)
    frame = out.to_frame()
    assert list(frame.columns) == ['estimate', 'ci.lb', 'ci.ub']
    assert list(frame.index) == ['intrcpt', 'ablat', 'tau^2', 'tau', 'I^2(%)', 'H^2']
    numpy.testing.assert_allclose(frame.loc['ablat'], [out.fixed.loc['ablat', c] for c in frame.columns])
    only = metafopy.confint(res1)
    assert list(only.to_frame().index) == ['tau^2', 'tau', 'I^2(%)', 'H^2']


def test_confint_digits_follow_fit(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, digits=2)
    row = str(metafopy.confint(res)).split('\n')[1].split()
    assert row[0] == 'tau^2'
    assert row[1] == '0.08'
    assert str(metafopy.confint(res, digits=3)).split('\n')[1].split()[1] == '0.076'


# -- progbar --------------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def intercept_only(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', data=dat)


def assert_bar(captured: str, total: int) -> None:
    assert '100%' in captured
    assert '|' + '+' * 50 + '|' in captured
    assert captured.count('%') >= total or total > 5


def test_progbar_influence_family(res1: metafopy.RMAResult, capsys: pytest.CaptureFixture[str]) -> None:
    for fn in (metafopy.influence, metafopy.rstudent, metafopy.cooks_distance, metafopy.dfbetas):
        capsys.readouterr()
        fn(res1)
        assert capsys.readouterr().err == ''
        got: typing.Any = fn(res1, progbar=True)
        assert_bar(capsys.readouterr().err, 13)
        plain: typing.Any = fn(res1)
        if isinstance(got, pandas.Series | pandas.DataFrame):
            pandas.testing.assert_frame_equal(pandas.DataFrame(got), pandas.DataFrame(plain))
        else:
            assert str(got) == str(plain)


def test_progbar_leave1out_cumul(intercept_only: metafopy.RMAResult, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    got = metafopy.leave1out(intercept_only, progbar=True)
    assert_bar(capsys.readouterr().err, 13)
    assert str(got) == str(metafopy.leave1out(intercept_only))
    got = metafopy.cumul(intercept_only, progbar=True)
    assert_bar(capsys.readouterr().err, 13)
    assert str(got) == str(metafopy.cumul(intercept_only))
    assert capsys.readouterr().err == ''


def test_progbar_binary_fits(dat: pandas.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    mh = metafopy.rma_mh(ai=dat['tpos'], bi=dat['tneg'], ci=dat['cpos'], di=dat['cneg'], measure='RR')
    capsys.readouterr()
    for fn in (metafopy.leave1out, metafopy.cumul, metafopy.rstudent):
        fn(mh, progbar=True)
        assert_bar(capsys.readouterr().err, 13)


def test_progbar_permutest(res1: metafopy.RMAResult, dat: pandas.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    got: typing.Any = metafopy.permutest(res1, iter=20, seed=1, progbar=True)
    assert_bar(capsys.readouterr().err, 20)
    plain: typing.Any = metafopy.permutest(res1, iter=20, seed=1)
    assert capsys.readouterr().err == ''
    numpy.testing.assert_array_equal(got.pval, plain.pval)
    small = metafopy.rma('yi', 'vi', data=dat.iloc[:6])
    exact: typing.Any = metafopy.permutest(small, exact=True, progbar=True)
    assert_bar(capsys.readouterr().err, 64)
    assert exact.exact_iter == 64


def test_progbar_permutest_location_scale(dat: pandas.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', scale='~ ablat', data=dat)
    capsys.readouterr()
    metafopy.permutest(res, iter=3, seed=1, progbar=True)
    assert '100%' in capsys.readouterr().err


def test_progbar_baujat(res1: metafopy.RMAResult, intercept_only: metafopy.RMAResult, capsys: pytest.CaptureFixture[str]) -> None:
    import matplotlib

    matplotlib.use('Agg')
    capsys.readouterr()
    metafopy.baujat(intercept_only, progbar=True)
    assert_bar(capsys.readouterr().err, 13)
    metafopy.baujat(res1)
    assert capsys.readouterr().err == ''


def test_progbar_tes(capsys: pytest.CaptureFixture[str]) -> None:
    yi = numpy.array([0.4, 0.5, 0.3, 0.6, 0.2, 0.7])
    vi = numpy.array([0.04, 0.05, 0.03, 0.06, 0.02, 0.07])
    capsys.readouterr()
    got = metafopy.tes(yi, vi, test='exact', find_lim=False, progbar=True)
    assert_bar(capsys.readouterr().err, 6)
    plain = metafopy.tes(yi, vi, test='exact', find_lim=False)
    assert capsys.readouterr().err == ''
    assert got.pval == plain.pval
    metafopy.tes(yi, vi, test='chi2', find_lim=False, progbar=True)
    assert capsys.readouterr().err == ''
