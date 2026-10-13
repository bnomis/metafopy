"""profile() of selection models with metafor's ``parallel``, ``ncpus`` and ``cl`` arguments: the results equal those of the serial run (and of R)."""

import concurrent.futures
import typing
import warnings

import numpy
import pytest
import test_selmodel

import metafopy
import metafopy.core.parallel
import metafopy.errors

CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'delta': ('hack_stepfun_b', {'delta': 1, 'steps': 6}),
    'delta_xlim': ('hack_stepfun_b', {'delta': 1, 'xlim': [0.1, 1.5], 'steps': 5}),
    'mods': ('rau_halfnorm_mods', {'delta': 0, 'steps': 4}),
    'tau2': ('hack_stepfun_b', {'tau2': True, 'steps': 6}),
}


def run(name: str, **kwargs: typing.Any) -> metafopy.SelProfileResult:
    case, args = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = metafopy.profile(test_selmodel.fit(case), **args, **kwargs)
    assert isinstance(out, metafopy.SelProfileResult)
    return out


def same(a: metafopy.SelProfileResult, b: metafopy.SelProfileResult) -> None:
    numpy.testing.assert_array_equal(a.tau2, b.tau2)
    numpy.testing.assert_array_equal(a.ll, b.ll)
    numpy.testing.assert_array_equal(a.beta.to_numpy(), b.beta.to_numpy())
    numpy.testing.assert_array_equal(a.ci_lb.to_numpy(), b.ci_lb.to_numpy())
    numpy.testing.assert_array_equal(a.ci_ub.to_numpy(), b.ci_ub.to_numpy())
    assert (a.comp, a.delta_pos, a.vc, a.maxll, a.xlim, a.ylim) == (b.comp, b.delta_pos, b.vc, b.maxll, b.xlim, b.ylim)


@pytest.mark.parametrize(('name', 'parallel'), [('tau2', 'snow'), ('mods', 'snow'), ('delta', 'multicore'), ('delta_xlim', 'multicore')])
def test_process_pool_equals_serial(name: str, parallel: typing.Literal['snow', 'multicore']) -> None:
    same(run(name), run(name, parallel=parallel, ncpus=2))


def test_ncpus_implies_snow_and_one_cpu_is_serial() -> None:
    serial = run('tau2')
    same(serial, run('tau2', parallel='snow', ncpus=1))
    same(serial, run('tau2', parallel='multicore', ncpus=1))
    same(serial, run('tau2', parallel='no', ncpus=1))
    assert metafopy.core.parallel.resolve_parallel('no', 3, None) == 'snow'
    assert metafopy.core.parallel.resolve_parallel('multicore', 3, None) == 'multicore'
    assert metafopy.core.parallel.resolve_parallel('no', 1, concurrent.futures.ThreadPoolExecutor()) == 'snow'
    same(serial, run('tau2', ncpus=2))


def test_executor_given_as_cl() -> None:
    serial = run('delta')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as threads:
        same(serial, run('delta', cl=threads))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as processes:
        same(serial, run('delta', cl=processes))
        # the executor stays usable: it is not shut down by profile()
        same(serial, run('delta', cl=processes))


def test_parallel_matches_r() -> None:
    ref = test_selmodel.REF['profile']['hack_stepfun_b_delta']['prof']
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as threads:
        pr = run('delta', cl=threads)
    test_selmodel.close(pr.grid, ref['grid'], 'grid')
    test_selmodel.close(pr.ll, ref['ll'], 'll', rtol=test_selmodel.RTOL_SE)


def test_all_components_in_parallel() -> None:
    res = test_selmodel.fit('hack_stepfun_a')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        serial = res.profile(steps=4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as threads:
            pooled = res.profile(steps=4, cl=threads)
    assert isinstance(serial, list)
    assert isinstance(pooled, list)
    assert len(serial) == len(pooled) == 3
    for a, b in zip(serial, pooled, strict=True):
        same(a, b)


def test_progress_bar(capsys: pytest.CaptureFixture[str]) -> None:
    run('delta')
    assert capsys.readouterr().err == ''
    run('delta', progbar=True)
    assert '100%' in capsys.readouterr().err
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as threads:
        run('delta', progbar=True, cl=threads)
    assert '100%' in capsys.readouterr().err


def test_argument_errors() -> None:
    res = test_selmodel.fit('hack_stepfun_b')
    with pytest.raises(metafopy.errors.InputError, match='parallel'):
        metafopy.profile(res, delta=1, parallel='cluster')
    with pytest.raises(metafopy.errors.InputError, match="'ncpus'"):
        metafopy.profile(res, delta=1, ncpus=0)
