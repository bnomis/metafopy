"""rma_glmm with ``control={'package': 'GLMMadaptive'}`` and ``{'package': 'glmmTMB'}`` checked against R metafor 5.2-1.

The fixtures come from ``r/generate_glmm_packages.R``.

GLMMadaptive is reproduced step by step (EM stage, then BFGS with R's ``optim`` scaling), so the results follow R's, including where R stops short of
the maximum; the exceptions in ``ADAPTIVE_TOLERANCE`` are fits whose long BFGS path is chaotic or that sit on a tie of
the stopping rule. glmmTMB maximises the Laplace likelihood with exact
derivatives, so only the small optimiser differences of ``TMB_TOLERANCE`` remain. See ``docs/differences.md``.
"""

import json
import pathlib
import typing
import warnings

import glmm_package_cases
import numpy
import pytest
import refutil

import metafopy
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase8' / 'glmm_packages.json').read_text(encoding='utf-8'))

ADAPTIVE_DEFAULT = 1e-4
TMB_DEFAULT = 5e-3
TMB_TOLERANCE: dict[str, float] = {}

# fits whose BFGS path is long enough for a 1e-12 change of the start to move the result (rs_ml_mod), or whose R run stops at a different iterate
CHAOTIC = frozenset({'rs_ml_mod', 'rs_collins'})
ADAPTIVE_TOLERANCE: dict[str, float] = {
    'rs_ml_mod': 2.5e-1,
    # the EM stage stops as soon as the log-likelihood rises by less than 1e-8 relative, and at iterations 5 to 9 the iterates tie to the last
    # bit, so a rounding difference between platforms stops it at the first fixed point (-0.532) instead of after the nodes are re-centred (-0.541, R's)
    'al_nagq1': 2e-2,
    # the confidence bound is a few 1e-3 from zero, so one unit in the last place of the optimiser shows
    'rs_ml_cor': 1e-3,
    'rs_collins': 3e-2,
}

SCALARS = ('tau2', 'sigma2', 'I2', 'QE_Wld', 'QE_LRT', 'QM')


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float) -> None:
    """Relative differences scaled by ``max(|expected|, 1e-3)``; NaN must match NaN."""
    exp = numpy.atleast_1d(numpy.array([numpy.nan if v is None else v for v in refutil.as_list(expected)], dtype=float))
    act = numpy.atleast_1d(numpy.asarray(actual, dtype=float))
    assert act.shape == exp.shape, what
    assert numpy.array_equal(numpy.isnan(act), numpy.isnan(exp)), f'{what}: {act} vs {exp}'
    mask = ~numpy.isnan(exp)
    worst = float(numpy.max(numpy.abs(act[mask] - exp[mask]) / numpy.maximum(numpy.abs(exp[mask]), 1e-3), initial=0.0))
    assert worst <= rtol, f'{what}: relative difference {worst:.2e} > {rtol:.0e}\n{act}\n{exp}'


def params(package: str) -> list[typing.Any]:
    """The cases that R fits with the package; the others fail in R (see the ``error`` entries of the fixture)."""
    return [name for name in glmm_package_cases.all_cases(package) if 'error' not in REF[package][name]]


def check(package: str, name: str, tolerance: float) -> None:
    ref = REF[package][name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = glmm_package_cases.all_cases(package)[name]({'package': package})
    close(res.beta, ref['beta'], 'beta', tolerance)
    close(res.se, ref['se'], 'se', tolerance)
    close(res.vb.ravel('F'), ref['vb'], 'vb', 2 * tolerance)
    if name not in CHAOTIC:
        # a bound is a difference of nearly equal numbers when the estimate is chaotic
        close(res.ci_lb, ref['ci_lb'], 'ci_lb', tolerance)
        close(res.ci_ub, ref['ci_ub'], 'ci_ub', tolerance)
    for key in SCALARS:
        close(getattr(res, key), ref[key], key, tolerance)
    close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], 'fit ML', tolerance)
    assert (res.k, res.p, res.parms, res.QE_df, res.model, res.method) == (ref['k'], ref['p'], ref['parms'], ref['QE_df'], ref['model'], ref['method'])


@pytest.mark.parametrize('name', params('GLMMadaptive'))
def test_glmmadaptive(name: str) -> None:
    check('GLMMadaptive', name, ADAPTIVE_TOLERANCE.get(name, ADAPTIVE_DEFAULT))


@pytest.mark.parametrize('name', params('glmmTMB'))
def test_glmmtmb(name: str) -> None:
    check('glmmTMB', name, TMB_TOLERANCE.get(name, TMB_DEFAULT))


def test_partial_package_name() -> None:
    ref = REF['partial']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = glmm_package_cases.bcg(model='UM.FS')({'package': 'glmmT'})
    close(res.beta, ref['beta'], 'beta', TMB_DEFAULT)


def test_unknown_package() -> None:
    with pytest.raises(metafopy.errors.InputError) as info:
        glmm_package_cases.bcg(model='UM.FS')({'package': 'nlme'})
    assert str(info.value) == REF['errors']['bad_package']
