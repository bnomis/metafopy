"""simulate() against R (src/tests/r/generate_simulate.R): seeded draws reproduce R's."""

import functools
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors

REF = refutil.load_phase7('simulate')
# location-scale and rma.mv estimates come from an optimiser; the draws scale with them, so they agree to its tolerance
OPTIM = 1e-5


@functools.cache
def uni() -> metafopy.RMAResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=refutil.bcg_na_yi())


def close(got: pandas.DataFrame, key: str, rtol: float = refutil.RTOL) -> None:
    refutil.close(got.to_numpy().ravel(), numpy.asarray(REF[key], dtype=float).ravel().tolist(), key, rtol=rtol)


def test_simulate_uni() -> None:
    res = uni()
    sims = metafopy.simulate(res, nsim=3, seed=1234)
    close(sims, 'uni')
    assert list(sims.columns) == ['sim_1', 'sim_2', 'sim_3']
    assert list(res.simulate(seed=1).index) == REF['uni_rows']
    assert sims.attrs['seed'] == 1234
    close(res.simulate(nsim=2, seed=42, olim=(0, -1.5)), 'uni_olim')
    close(metafopy.simulate(res, nsim=2, seed=7, withvb=True), 'uni_withvb')


def test_simulate_global_rng() -> None:
    res = uni()
    rng = metafopy.set_seed(99)
    close(metafopy.simulate(res, nsim=2), 'uni_global')
    refutil.close(rng.rnorm(2), REF['uni_global_next'], 'next draws')
    # a seed leaves the global generator alone, as R restores .Random.seed
    rng = metafopy.set_seed(99)
    metafopy.simulate(res, seed=3)
    close(metafopy.simulate(res, nsim=2), 'uni_global')


def test_simulate_ls_mv() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ls = metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ year', data=refutil.bcg_na_yi())
    close(metafopy.simulate(ls, nsim=2, seed=5), 'ls', rtol=OPTIM)
    mv = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=metafopy.datasets.load('konstantopoulos2011'))
    close(mv.simulate(nsim=2, seed=11), 'mv', rtol=OPTIM)


def test_simulate_errors() -> None:
    glmm = metafopy.rma_glmm(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'))
    with pytest.raises(metafopy.errors.InputError, match='not available'):
        metafopy.simulate(glmm)  # type: ignore[arg-type]
    with pytest.raises(metafopy.errors.InputError, match='nsim'):
        metafopy.simulate(uni(), nsim=0)
    with pytest.raises(metafopy.errors.InputError, match='olim'):
        metafopy.simulate(uni(), olim=[1.0])
