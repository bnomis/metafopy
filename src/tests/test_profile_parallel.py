"""profile() of rma_uni and location-scale fits and gosh() with metafor's ``parallel``, ``ncpus``, ``cl`` and ``progbar`` arguments.

The fits in a process pool are the same deterministic refits as in the serial run, so the results are identical; the serial results are
checked against R elsewhere (test_diagnostics.py, test_rma_ls.py).
"""

import concurrent.futures
import functools
import typing
import warnings

import numpy
import pytest
import refutil

import metafopy
import metafopy.errors


@functools.cache
def uni() -> metafopy.RMAResult:
    return metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=refutil.bcg())


@functools.cache
def ls() -> metafopy.RMALSResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metafopy.rma_uni('yi', 'vi', scale='~ ablat', data=refutil.bcg())


def frames_equal(a: typing.Any, b: typing.Any) -> None:
    for name in ('ll', 'beta', 'ci_lb', 'ci_ub'):
        numpy.testing.assert_array_equal(numpy.asarray(getattr(a, name)), numpy.asarray(getattr(b, name)))


# auditme: accept[literal-duplication] the process-pool choices are listed per test
@pytest.mark.parametrize('parallel', ['snow', 'multicore'])
# auditme: accept[literal-duplication] the process-pool choices are listed per test
def test_profile_uni(parallel: typing.Literal['snow', 'multicore']) -> None:
    serial = metafopy.profile(uni(), steps=6)
    pooled = metafopy.profile(uni(), steps=6, parallel=parallel, ncpus=2)
    frames_equal(serial, pooled)
    numpy.testing.assert_array_equal(serial.I2, pooled.I2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as threads:
        frames_equal(serial, metafopy.profile(uni(), steps=6, cl=threads))


def test_profile_ls() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        serial: object = metafopy.profile(ls(), steps=5)
        pooled: object = metafopy.profile(ls(), steps=5, parallel='snow', ncpus=2)
    assert isinstance(serial, list)
    assert isinstance(pooled, list)
    for a, b in zip(serial, pooled, strict=True):
        frames_equal(a, b)


def test_gosh() -> None:
    d = refutil.bcg().iloc[:8]
    res = metafopy.rma_uni('yi', 'vi', data=d)
    serial = metafopy.gosh(res)
    pooled = metafopy.gosh(res, parallel='multicore', ncpus=2)
    numpy.testing.assert_array_equal(serial.res.to_numpy(), pooled.res.to_numpy())
    reml = metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=d)
    a = metafopy.gosh(reml, subsets=50, seed=3)
    b = metafopy.gosh(reml, subsets=50, seed=3, parallel='snow', ncpus=2)
    numpy.testing.assert_array_equal(a.res.to_numpy(), b.res.to_numpy())
    numpy.testing.assert_array_equal(a.incl, b.incl)


def test_progbar_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    metafopy.profile(uni(), steps=4, progbar=True)
    assert '100%' in capsys.readouterr().err
    metafopy.gosh(metafopy.rma_uni('yi', 'vi', data=refutil.bcg().iloc[:5]), progbar=True)
    assert '100%' in capsys.readouterr().err
    with pytest.raises(metafopy.errors.InputError):
        metafopy.profile(uni(), parallel='threads')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.gosh(uni(), ncpus=0)
