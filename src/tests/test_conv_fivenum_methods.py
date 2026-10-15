"""The 'qe', 'bc', 'mln' and 'blue' methods of conv_fivenum() against R metafor 5.2-1 with estmeansd 1.0.1 and metaBLUE 1.0.0.

Fixtures come from tests/r/generate_fivenum_methods.R. 'bc' and 'mln' use R's random number generator, so they reproduce R's results for the
same seed; 'qe' minimises with SciPy's L-BFGS-B instead of R's, so it agrees to the optimiser's tolerance.
"""

import json
import pathlib
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.core.rng
import metafopy.effects.fivenum_estimators
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase10' / 'conv_fivenum_methods.json').read_text(encoding='utf-8'))
# auditme: accept[non-alphabetical-dict-keys] order follows the argument order of the R function
COLS: dict[str, typing.Any] = {'min': 'min', 'q1': 'q1', 'median': 'median', 'q3': 'q3', 'max': 'max', 'n': 'n'}
SEED = 42


def scenario(args: dict[str, float]) -> tuple[numpy.typing.NDArray[numpy.float64], str]:
    quants = numpy.array([args[k] for k in ('min.val', 'q1.val', 'med.val', 'q3.val', 'max.val') if k in args])
    return quants, {3: 'S1' if 'min.val' in args else 'S2', 5: 'S3'}[len(quants)]


def check(key: str, source: str, **kwargs: typing.Any) -> pandas.DataFrame:
    ref = REF[key]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = metafopy.conv_fivenum(**COLS, data=refutil.r_frame(REF[source]), **kwargs)
    rtol = 1e-4 if kwargs.get('method') == 'qe' else 1e-9
    refutil.assert_frame(out, ref['data'], key, rtol=rtol)
    refutil.close(out.attrs['tval'], ref['tval'], f'{key}: tval', rtol=1e-12)
    refutil.close(out.attrs['crit'], ref['crit'], f'{key}: crit', rtol=1e-12)
    assert [None if pandas.isna(v) else bool(v) for v in out.attrs['sig']] == refutil.as_list(ref['sig'])
    assert out.attrs['dist'] == refutil.as_list(ref['dist'])
    return out


@pytest.mark.parametrize('key', sorted(REF['single']))
def test_metablue(key: str) -> None:
    case = REF['single'][key]
    x, n, kind = case['X'], case['n'], case['type']
    est = metafopy.effects.fivenum_estimators.blue_s(x, n, kind)
    refutil.close(est.alpha, case['alpha'], f'{key}: alpha', rtol=1e-10)
    refutil.close(est.B.ravel(), numpy.array(case['B']['values']).ravel().tolist(), f'{key}: B', rtol=1e-10)
    for name, value in (('muhat', est.muhat), ('sigmahat', est.sigmahat), ('var_mu', est.var_mu), ('var_sigma', est.var_sigma)):
        refutil.close(value, case[name], f'{key}: {name}', rtol=1e-9)
    refutil.close(metafopy.effects.fivenum_estimators.luo_mean(x, n, kind), case['luo'], f'{key}: Luo.mean', rtol=1e-12)
    refutil.close(metafopy.effects.fivenum_estimators.wan_std(x, n, kind), case['wan'], f'{key}: Wan.std', rtol=1e-12)


@pytest.mark.parametrize('key', sorted(REF['estimators']))
def test_estimators(key: str) -> None:
    case = REF['estimators'][key]
    quants, kind = scenario(case['args'])
    n = case['args']['n']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        qe = metafopy.effects.fivenum_estimators.qe_mean_sd(quants, n, kind)
        bc = metafopy.effects.fivenum_estimators.bc_mean_sd(quants, n, kind, metafopy.core.rng.RRandom(SEED))
        mln = metafopy.effects.fivenum_estimators.mln_mean_sd(quants, n, kind, metafopy.core.rng.RRandom(SEED))
    assert qe.dist == case['qe']['dist']
    refutil.close([qe.mean, qe.sd], [case['qe']['mean'], case['qe']['sd']], f'{key}: qe', rtol=1e-4)
    refutil.close([bc.mean, bc.sd], [case['bc']['mean'], case['bc']['sd']], f'{key}: bc', rtol=1e-9)
    refutil.close([mln.mean, mln.sd], [case['mln']['mean'], case['mln']['sd']], f'{key}: mln', rtol=1e-9)


def test_qe_fits() -> None:
    """Every candidate distribution fitted by L-BFGS-B ends near R's (up to the flat optimum of the least-squares criterion)."""
    for key, case in REF['estimators'].items():
        quants, kind = scenario(case['args'])
        fits = metafopy.effects.fivenum_estimators.qe_fit(quants, case['args']['n'], kind)
        for name, ref in case['qe']['fits'].items():
            fit = fits[name]
            if ref['value'] is None:
                # R drops a fit that raises a warning (qbeta's, for instance); SciPy returns a poor fit, which is never selected
                assert name == 'beta' or fit is None, f'{key}: {name}'
                continue
            assert fit is not None, f'{key}: {name}'
            refutil.close(metafopy.effects.fivenum_estimators.mean_sd(fit[0], name), [ref['mean'], ref['sd']], f'{key}: {name}', rtol=5e-3)


@pytest.mark.parametrize('method', ['qe', 'bc', 'mln', 'blue'])
def test_methods(method: str) -> None:
    check(method, 'input', method=method, test=False, seed=1234)
    check(f'{method}_test', 'input', method=method, test=True, seed=1234)


def test_seed() -> None:
    check('bc_seed2', 'input', method='bc', seed=7)
    check('mln_seed2', 'input', method='mln', seed=7)
    first = metafopy.conv_fivenum(**COLS, data=refutil.r_frame(REF['input']), method='bc', seed=3)
    second = metafopy.conv_fivenum(**COLS, data=refutil.r_frame(REF['input']), method='bc', seed=3)
    pandas.testing.assert_frame_equal(first, second)


def test_unseeded_uses_the_module_generator() -> None:
    metafopy.core.rng.set_seed(1234)
    out = metafopy.conv_fivenum(**COLS, data=refutil.r_frame(REF['input']), method='mln')
    refutil.assert_frame(out, REF['mln']['data'], 'mln', rtol=1e-9)


def test_noappend() -> None:
    out = check('noappend', 'input', method='qe', append=False)
    assert list(out.columns) == ['mean', 'sd']


def test_negative_values() -> None:
    """With negative values 'qe' fits the normal distribution; its skewness test, once switched on, stays on after a skipped study (as in R)."""
    check('qe_neg', 'input_neg', method='qe')
    check('qe_neg_notest', 'input_neg', method='qe', test=False)
    check('blue_neg', 'input_neg', method='blue', test=False)


def test_errors() -> None:
    errors = REF['errors']
    with pytest.raises(metafopy.errors.InputError, match='non-positive'):
        metafopy.conv_fivenum(**COLS, data=refutil.r_frame(REF['input_neg']), method='bc')
    assert errors['bc_neg'].startswith('Cannot use method with non-positive values')
    for method in (['qe', 'bc'], ['blue', 'wan2014']):
        with pytest.raises(metafopy.errors.InputError, match='same'):
            metafopy.conv_fivenum(**COLS, data=refutil.r_frame(REF['input']), method=method)
    assert errors['mixed'].startswith('Must use the same')


def test_blue_singular(monkeypatch: pytest.MonkeyPatch) -> None:
    def singular(_: typing.Any) -> typing.NoReturn:
        raise numpy.linalg.LinAlgError('Singular matrix')

    monkeypatch.setattr(numpy.linalg, 'inv', singular)
    with pytest.raises(metafopy.errors.EstimationError, match='singular'):
        metafopy.effects.fivenum_estimators.blue_s([1.0, 2.0, 3.0], 20, 'S1')
