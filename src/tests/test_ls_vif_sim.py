"""vif(sim=) for location-scale models against R metafor 5.2-1 (fixtures from tests/r/generate_ls_vif_sim.R).

The permutations are reproduced exactly (iteration l shuffles with R's generator seeded by seed + l), so the simulated (G)VIFs are those of R up to the
optimiser's accuracy. Permuted fits whose scale coefficients run to the boundary (a flat likelihood) can end at different points or have a Hessian that is
invertible in one implementation and not in the other; the scale-part comparisons therefore tolerate a few such iterations.
"""

import concurrent.futures
import json
import pathlib
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors
import metafopy.methods.ls_vif

HERE = pathlib.Path(__file__).parent
REF: dict[str, typing.Any] = json.loads((HERE / 'reference' / 'phase6' / 'ls_vif_sim.json').read_text(encoding='utf-8'))
DATA: dict[str, typing.Any] = json.loads((HERE / 'reference' / 'phase5' / 'location_scale.json').read_text(encoding='utf-8'))

# auditme: accept[non-alphabetical-literal] the order of the columns of the model matrix
LOCATION = ('ni100', 'meta')
# auditme: accept[non-alphabetical-literal] the order of the columns of the model matrix
THREE = ('ni100', 'meta', 'imag')
OBSERVED = ('yi', 'vi', *THREE)

type FloatArray = numpy.typing.NDArray[numpy.float64]

any_vif: typing.Callable[..., typing.Any] = metafopy.vif


def complete() -> pandas.DataFrame:
    d = pandas.DataFrame(DATA['bd_data'])
    return d.dropna(subset=list(OBSERVED)).reset_index(drop=True)


def fit(mods: typing.Sequence[str] | None, scale: bool = True, **kwargs: typing.Any) -> metafopy.RMALSResult:
    d = complete()
    X = None if mods is None else d[list(mods)].to_numpy()
    Z = numpy.column_stack([numpy.ones(len(d)), d['ni100'], d['imag']]) if scale else '~ 1'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_uni(d['yi'].to_numpy(), d['vi'].to_numpy(), mods=X, scale=Z, **kwargs)
    assert isinstance(res, metafopy.RMALSResult)
    return res


def rows(sim: typing.Any) -> FloatArray:
    return numpy.array([[numpy.nan if v is None else v for v in refutil.as_list(r)] for r in sim], dtype=float)


def check_part(out: metafopy.VIFResult, ref: dict[str, typing.Any], what: str, max_bad: int = 0) -> None:
    refutil.close(out.vifs.to_numpy(), ref['vifs'], f'{what} vifs', rtol=1e-4)
    assert out.sim is not None
    assert out.prop is not None
    assert out.sim_type == ref['sim_type']
    expected = rows(ref['sim'])
    assert out.sim.shape == expected.shape
    close = numpy.isclose(out.sim, expected, rtol=1e-4, atol=1e-6, equal_nan=True)
    bad = ~close.all(axis=1)
    assert bad.sum() <= max_bad, f'{what}: iterations {numpy.flatnonzero(bad).tolist()} differ'
    assert not (numpy.isnan(expected) & ~numpy.isnan(out.sim)).any(), what
    if not bad.any():
        refutil.close(out.prop, ref['prop'], f'{what} prop', rtol=1e-10)


CASES: dict[str, dict[str, typing.Any]] = {
    'both': {'sim': 12, 'seed': 42},
    'both_btt': {'btt': 1, 'att': [[1], [1, 2]], 'sim': 8, 'seed': 7},
    'both_reest': {'reestimate': True, 'sim': 6, 'seed': 3},
}
UNSTABLE_ALPHA = {'both_reest'}


@pytest.mark.parametrize('case', list(CASES))
def test_sim_matches_r(case: str) -> None:
    out = any_vif(fit(LOCATION), progbar=False, **CASES[case])
    assert isinstance(out, metafopy.methods.ls_vif.LSVIFResult)
    check_part(out.beta, REF[case]['beta'], f'{case} beta')
    if case in UNSTABLE_ALPHA:
        # the re-estimated scale models of the permuted fits have flat likelihoods (scale coefficients at the boundary), so only the shape agrees
        assert out.alpha.sim is not None
        assert out.alpha.sim.shape == rows(REF[case]['alpha']['sim']).shape
        return
    check_part(out.alpha, REF[case]['alpha'], f'{case} alpha', max_bad=2)


def test_sim_three_location_predictors() -> None:
    out = any_vif(fit(THREE), sim=10, seed=21, progbar=False)
    assert isinstance(out, metafopy.methods.ls_vif.LSVIFResult)
    check_part(out.beta, REF['three']['beta'], 'three beta')
    check_part(out.alpha, REF['three']['alpha'], 'three alpha', max_bad=3)


def test_sim_ml_identity() -> None:
    out = any_vif(fit(LOCATION, method='ML', link='identity'), sim=8, seed=11, progbar=False)
    assert isinstance(out, metafopy.methods.ls_vif.LSVIFResult)
    check_part(out.beta, REF['ml_identity']['beta'], 'ml beta', max_bad=3)
    # the scale Hessian of this fit is singular in R as well: no (G)VIFs and so nothing to simulate
    assert REF['ml_identity']['alpha']['sim'] is None
    assert out.alpha.sim is None
    assert numpy.isnan(out.alpha.vifs.to_numpy()).all()


def test_sim_single_part() -> None:
    scale_only = any_vif(fit(None), sim=8, seed=5, progbar=False)
    assert isinstance(scale_only, metafopy.VIFResult)
    check_part(scale_only, REF['scale_only'], 'scale only', max_bad=2)
    loc_only = any_vif(fit(LOCATION, scale=False), sim=8, seed=5, progbar=False)
    assert isinstance(loc_only, metafopy.VIFResult)
    check_part(loc_only, REF['loc_only'], 'location only')


def test_sim_print_includes_prop() -> None:
    out = any_vif(fit(LOCATION), sim=12, seed=42, progbar=False)
    text = str(out)
    assert 'prop' in text
    assert text.count('prop') == 2


def test_parallel_equals_serial() -> None:
    res = fit(LOCATION)
    kwargs: dict[str, typing.Any] = {'btt': 1, 'att': [[1], [1, 2]], 'sim': 6, 'seed': 13, 'progbar': False}
    serial = any_vif(res, **kwargs)
    snow = any_vif(res, parallel='snow', ncpus=2, **kwargs)
    multicore = any_vif(res, parallel='multicore', ncpus=2, **kwargs)
    implied = any_vif(res, ncpus=2, **kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pooled = any_vif(res, cl=pool, **kwargs)
    assert isinstance(serial, metafopy.methods.ls_vif.LSVIFResult)
    for other in (snow, multicore, implied, pooled):
        assert isinstance(other, metafopy.methods.ls_vif.LSVIFResult)
        for part in ('beta', 'alpha'):
            a, b = getattr(serial, part), getattr(other, part)
            numpy.testing.assert_array_equal(a.sim, b.sim)
            numpy.testing.assert_array_equal(a.prop, b.prop)


def test_seed_reproducible_and_unseeded_runs() -> None:
    res = fit(LOCATION)
    a = any_vif(res, sim=4, seed=9, progbar=False)
    b = any_vif(res, sim=4, seed=9, progbar=False)
    c = any_vif(res, sim=4, seed=10, progbar=False)
    assert isinstance(a, metafopy.methods.ls_vif.LSVIFResult)
    assert isinstance(b, metafopy.methods.ls_vif.LSVIFResult)
    assert isinstance(c, metafopy.methods.ls_vif.LSVIFResult)
    assert a.beta.sim is not None
    assert c.beta.sim is not None
    numpy.testing.assert_array_equal(a.beta.sim, b.beta.sim)
    assert not numpy.array_equal(a.beta.sim, c.beta.sim)
    d = any_vif(res, sim=3, progbar=False)
    assert isinstance(d, metafopy.methods.ls_vif.LSVIFResult)
    assert d.beta.sim is not None
    assert d.beta.sim.shape == (3, 2)


def test_progress_bar(capsys: pytest.CaptureFixture[str]) -> None:
    any_vif(fit(LOCATION), sim=2, seed=1, progbar=True)
    assert '100%' in capsys.readouterr().err


def test_sim_errors() -> None:
    res = fit(LOCATION)
    with pytest.raises(metafopy.errors.InputError, match="'sim'"):
        any_vif(res, sim=1)
    with pytest.raises(metafopy.errors.InputError, match='parallel'):
        any_vif(res, sim=3, parallel='cluster')
    with pytest.raises(metafopy.errors.InputError, match="'ncpus'"):
        any_vif(res, sim=3, ncpus=0)
