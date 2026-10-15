"""Hat matrices, summary() and the fit statistics of several models, checked against R metafor 5.2-1 (tests/r/generate_audit3.R)."""

import collections.abc
import typing

import matplotlib
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors

matplotlib.use('Agg')

REF: dict[str, typing.Any] = refutil.load_phase12('audit3')


@pytest.fixture(scope='module')
# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture helpers
def escalc_frame() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


def uni(escalc_frame: pandas.DataFrame, **kwargs: typing.Any) -> metafopy.RMAResult:
    return typing.cast(metafopy.RMAResult, metafopy.rma('yi', 'vi', data=escalc_frame, **kwargs))


def hat_close(actual: numpy.typing.NDArray[numpy.float64], spec: typing.Any, what: str, rtol: float) -> None:
    expected = numpy.asarray(spec, dtype=float)
    assert actual.shape == expected.shape, what
    refutil.close(actual.reshape(-1), expected.reshape(-1).tolist(), what, rtol=rtol)


def test_hat_matrix(escalc_frame: pandas.DataFrame) -> None:
    ref = REF['hat']
    mods = uni(escalc_frame, mods='~ ablat')
    hat_close(mods.hatvalues(type='matrix'), ref['uni_mods'], 'uni_mods', 1e-8)
    refutil.close(mods.hatvalues(), ref['uni_mods_diag'], 'diag', rtol=1e-8)
    hat_close(uni(escalc_frame, method='EE').hatvalues(type='matrix'), ref['uni_ee'], 'ee', 1e-8)
    weighted = uni(escalc_frame, mods='~ ablat', weights=1 / numpy.sqrt(escalc_frame['vi'].to_numpy()))
    hat_close(weighted.hatvalues(type='matrix'), ref['uni_w'], 'weights', 1e-8)
    with pytest.raises(metafopy.errors.InputError, match='type'):
        mods.hatvalues(type='full')  # type: ignore[arg-type]


def test_hat_matrix_ls(escalc_frame: pandas.DataFrame) -> None:
    ref = REF['hat']
    ls = metafopy.rma('yi', 'vi', scale='~ ablat', data=escalc_frame)
    hat_close(ls.hatvalues(type='matrix'), ref['ls'], 'ls', 1e-6)
    refutil.close(ls.hatvalues(), ref['ls_diag'], 'ls diag', rtol=1e-6)


def test_hat_matrix_trimfill_and_robust(escalc_frame: pandas.DataFrame) -> None:
    ref = REF['hat']
    tf = metafopy.trimfill(uni(escalc_frame))
    hat_close(tf.hatvalues(type='matrix'), ref['trimfill'], 'trimfill', 1e-8)
    rb = metafopy.robust(uni(escalc_frame, mods='~ ablat'), escalc_frame['trial'].to_numpy())
    hat_close(rb.hatvalues(type='matrix'), ref['robust'], 'robust', 1e-8)
    refutil.close(metafopy.cooks_distance(rb), ref['cooks_robust'], 'cooks robust', rtol=1e-6)
    refutil.close(metafopy.cooks_distance(uni(escalc_frame, mods='~ ablat')), ref['cooks_uni'], 'cooks uni', rtol=1e-6)


def mh() -> typing.Any:
    return metafopy.rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


def peto() -> typing.Any:
    return metafopy.rma_peto(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


def mv(escalc_frame: pandas.DataFrame, **kwargs: typing.Any) -> typing.Any:
    return metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=escalc_frame, **kwargs)


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
SUMMARIES: dict[str, collections.abc.Callable[[pandas.DataFrame], typing.Any]] = {
    'uni_re': lambda d: uni(d),
    'uni_mods': lambda d: uni(d, mods='~ ablat'),
    'uni_ml': lambda d: uni(d, method='ML'),
    'uni_ee': lambda d: uni(d, method='EE'),
    'uni_knha': lambda d: uni(d, test='knha'),
    'uni_tau2': lambda d: uni(d, tau2=0.1),
    'selmodel': lambda d: metafopy.selmodel(uni(d), type='halfnorm'),
    'mh': lambda d: mh(),
    'peto': lambda d: peto(),
    'mv': lambda d: mv(d),
    'mv_mods': lambda d: mv(d, mods='~ ablat'),
    'ls': lambda d: metafopy.rma('yi', 'vi', scale='~ ablat', data=d),
    'trimfill': lambda d: metafopy.trimfill(uni(d)),
}


@pytest.mark.parametrize('key', sorted(SUMMARIES))
def test_summary(escalc_frame: pandas.DataFrame, key: str) -> None:
    out = SUMMARIES[key](escalc_frame).summary()
    expected = REF['summary'][key]
    if key == 'ls':
        # the scale standard errors agree with R to about 1e-5 (see test_ls_se), so the last printed digit of the scale rows may differ
        cut = expected.index('Model Results (Scale)')
        out, expected = out[: out.index('Model Results (Scale)')], expected[:cut]
    assert refutil.same_print(out, expected), out


def test_print_omits_fit_table(escalc_frame: pandas.DataFrame) -> None:
    tf = metafopy.trimfill(uni(escalc_frame))
    assert refutil.same_print(str(tf), REF['summary']['print_trimfill'])
    assert 'logLik' not in str(uni(escalc_frame))
    assert 'logLik' in uni(escalc_frame).summary()


def frame_of(spec: dict[str, typing.Any]) -> pandas.DataFrame:
    return pandas.DataFrame({name: refutil.as_list(col) for name, col in spec['columns'].items()}, index=spec['rownames'])


def check_frame(actual: typing.Any, spec: dict[str, typing.Any], what: str, rtol: float = 1e-6) -> None:
    expected = frame_of(spec)
    assert actual.shape == expected.shape, what
    for got, want in zip(actual.to_numpy().T, expected.to_numpy().T, strict=True):
        refutil.close(got, want.tolist(), what, rtol=rtol)


def fits(escalc_frame: pandas.DataFrame) -> dict[str, typing.Any]:
    return {
        'r1': uni(escalc_frame),
        'r2': uni(escalc_frame, mods='~ ablat'),
        'r3': uni(escalc_frame, mods='~ ablat + year'),
        'l1': metafopy.rma('yi', 'vi', scale='~ 1', data=escalc_frame),
        'l2': metafopy.rma('yi', 'vi', scale='~ ablat', data=escalc_frame),
        'm1': mv(escalc_frame),
        'm2': mv(escalc_frame, mods='~ ablat'),
        's1': metafopy.selmodel(uni(escalc_frame), type='halfnorm'),
        's2': metafopy.selmodel(uni(escalc_frame), type='negexp'),
    }


def test_fitstats(escalc_frame: pandas.DataFrame) -> None:
    ref, f = REF['fitstats'], fits(escalc_frame)
    check_frame(f['r1'].fitstats(f['r2'], f['r3']), ref['uni'], 'uni')
    check_frame(f['r1'].fitstats(f['r2'], f['r3'], REML=False), ref['uni_ml'], 'uni ml')
    refutil.close(f['r2'].fitstats(), ref['uni_single'], 'single')
    ml1, ml2 = uni(escalc_frame, method='ML'), uni(escalc_frame, mods='~ ablat', method='ML')
    check_frame(ml1.fitstats(ml2), ref['uni_ml_first'], 'ml first')
    check_frame(f['l1'].fitstats(f['l2']), ref['ls'], 'ls')
    check_frame(f['m1'].fitstats(f['m2']), ref['mv'], 'mv')
    check_frame(f['s1'].fitstats(f['s2']), ref['sel'], 'sel')


def test_aic_bic(escalc_frame: pandas.DataFrame) -> None:
    ref, f = REF['fitstats'], fits(escalc_frame)
    check_frame(f['r1'].AIC(f['r2'], f['r3']), ref['aic_uni'], 'aic')
    check_frame(f['r1'].AIC(f['r2'], f['r3'], correct=True), ref['aic_uni_correct'], 'aic correct')
    refutil.close(f['r2'].AIC(correct=True), ref['aic_single_correct'], 'single correct')
    refutil.close(f['r2'].AIC(k=3), ref['aic_single_k'], 'k is ignored as in metafor')
    check_frame(f['l1'].AIC(f['l2']), ref['aic_ls'], 'aic ls')
    check_frame(f['m1'].AIC(f['m2'], correct=True), ref['aic_mv'], 'aic mv')
    check_frame(f['s1'].AIC(f['s2']), ref['aic_sel'], 'aic sel')
    check_frame(f['r1'].BIC(f['r2'], f['r3']), ref['bic_uni'], 'bic')
    check_frame(f['l1'].BIC(f['l2']), ref['bic_ls'], 'bic ls')
    check_frame(f['m1'].BIC(f['m2']), ref['bic_mv'], 'bic mv')
    check_frame(f['s1'].BIC(f['s2']), ref['bic_sel'], 'bic sel')


def test_different_data_warns(escalc_frame: pandas.DataFrame) -> None:
    with pytest.warns(UserWarning, match='same data'):
        uni(escalc_frame).AIC(uni(escalc_frame, subset='ablat > 20'))
