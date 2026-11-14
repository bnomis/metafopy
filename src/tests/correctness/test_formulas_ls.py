"""Unit checks of the location-scale functions of ``formulas``: the analytic score and information against numerical derivatives, and the failure mode.

The data come from the bundled datasets only; the effect sizes of ``dat.bcg`` are computed with ``formulas``, not with metafopy.
"""

import itertools
import pathlib
import tomllib

import numpy
import numpy.typing
import pytest

import correctness.formulas
import metafopy.datasets

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Data = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


def bcg() -> Data:
    f = metafopy.datasets.load('dat.bcg')
    a, b = f['tpos'].to_numpy(dtype=float), f['tneg'].to_numpy(dtype=float)
    c, d = f['cpos'].to_numpy(dtype=float), f['cneg'].to_numpy(dtype=float)
    yi = correctness.formulas.log_risk_ratio_yi(a, b, c, d)
    vi = correctness.formulas.log_risk_ratio_vi(a, b, c, d)
    X = numpy.column_stack([numpy.ones(len(yi)), f['ablat'].to_numpy(dtype=float)])
    Z = numpy.column_stack([numpy.ones(len(yi)), (f['year'].to_numpy(dtype=float) - 1950) / 10])
    return yi, vi, X, Z


def bangert() -> Data:
    f = metafopy.datasets.load('dat.bangertdrowns2004').dropna(subset=['length']).reset_index(drop=True)
    yi, vi = f['yi'].to_numpy(dtype=float), f['vi'].to_numpy(dtype=float)
    length = f['length'].to_numpy(dtype=float)
    scaled = (length - length.mean()) / length.std()
    X = numpy.column_stack([numpy.ones(len(yi)), scaled])
    Z = numpy.column_stack([numpy.ones(len(yi)), scaled])
    return yi, vi, X, Z


DATASETS = {'bangertdrowns2004': bangert, 'bcg': bcg}
ALPHAS = {
    'identity': [numpy.array([0.3, 0.02]), numpy.array([0.5, 0.0]), numpy.array([0.2, 0.05])],
    'log': [numpy.array([-1.0, 0.2]), numpy.array([-2.0, -0.3]), numpy.array([0.0, 0.1])],
}
METHODS = sorted({'REML', 'ML'})
CASES = list(itertools.product(DATASETS, METHODS, ('log', 'identity')))


def numerical_hessian(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> FloatArray:
    """The Jacobian of the analytic score, by forward differences."""
    step = 1e-7
    base = correctness.formulas.ls_score(yi, vi, X, Z, alpha, link, method)
    columns = []
    for ell in range(len(alpha)):
        shifted = alpha.copy()
        shifted[ell] += step
        columns.append((correctness.formulas.ls_score(yi, vi, X, Z, shifted, link, method) - base) / step)
    return numpy.column_stack(columns)


def numerical_score(yi: FloatArray, vi: FloatArray, X: FloatArray, Z: FloatArray, alpha: FloatArray, link: str, method: str) -> FloatArray:
    """The gradient of the log-likelihood by forward differences."""
    step = 1e-7
    base = correctness.formulas.ls_loglik(yi, vi, X, Z, alpha, link, method)
    out = numpy.zeros(len(alpha))
    for j in range(len(alpha)):
        shifted = alpha.copy()
        shifted[j] += step
        out[j] = (correctness.formulas.ls_loglik(yi, vi, X, Z, shifted, link, method) - base) / step
    return out


@pytest.mark.parametrize(('name', 'method', 'link'), CASES)
def test_score_and_information_match_numerical_derivatives(name: str, method: str, link: str) -> None:
    yi, vi, X, Z = DATASETS[name]()
    checked = 0
    for alpha in ALPHAS[link]:
        if correctness.formulas.ls_loglik(yi, vi, X, Z, alpha, link, method) == -numpy.inf:
            continue
        checked += 1
        score = correctness.formulas.ls_score(yi, vi, X, Z, alpha, link, method)
        assert score == pytest.approx(numerical_score(yi, vi, X, Z, alpha, link, method), rel=1e-4, abs=1e-4)
        info = correctness.formulas.ls_information(yi, vi, X, Z, alpha, link, method)
        assert info == pytest.approx(-numerical_hessian(yi, vi, X, Z, alpha, link, method), rel=1e-4, abs=1e-4)
    assert checked > 0


def test_information_of_the_wrong_sign_is_caught() -> None:
    yi, vi, X, Z = bcg()
    alpha = ALPHAS['log'][0]
    info = correctness.formulas.ls_information(yi, vi, X, Z, alpha, 'log', 'ML')
    assert info != pytest.approx(numerical_hessian(yi, vi, X, Z, alpha, 'log', 'ML'), rel=1e-4, abs=1e-4)


@pytest.mark.parametrize('method', METHODS)
def test_reference_fit_has_a_zero_score_and_a_covariance(method: str) -> None:
    yi, vi, X, Z = bangert()
    fit = correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', method)
    assert correctness.formulas.ls_score(yi, vi, X, Z, fit['alpha'], 'log', method) == pytest.approx(0.0, abs=1e-5)
    assert fit['ll'] == pytest.approx(correctness.formulas.ls_loglik(yi, vi, X, Z, fit['alpha'], 'log', method), rel=1e-12)
    assert numpy.all(numpy.linalg.eigvalsh(fit['va']) > 0)


def test_reference_fit_raises_when_nothing_converges() -> None:
    yi, vi, X, Z = bcg()
    with pytest.raises(RuntimeError, match='no start converged'):
        correctness.formulas.ls_reference_fit(yi, vi, X, Z, 'log', 'ML', maxiter=1)


def test_inline_columns_are_the_recoding_of_the_dataset() -> None:
    """The inline ``yi``, ``vi`` and ``ni100`` of the no-intercept heterogeneity case are the dataset's ``yi`` and ``vi`` and ``ni / 100``."""
    path = pathlib.Path(__file__).parent / 'cases' / 'location_scale.toml'
    cases = tomllib.loads(path.read_text(encoding='utf-8'))['case']
    inline = next(c['data'] for c in cases if c['id'] == 'ls.heterogeneity.bdni')
    frame = metafopy.datasets.load('dat.bangertdrowns2004')
    assert inline['yi'] == frame['yi'].tolist()
    assert inline['vi'] == frame['vi'].tolist()
    assert inline['ni100'] == pytest.approx((frame['ni'] / 100).tolist(), rel=1e-15)
