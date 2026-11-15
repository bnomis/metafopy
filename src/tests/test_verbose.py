"""verbose= output of the model fitters, confint, trimfill and fsn against R (src/tests/r/generate_verbose.R).

R's ``cat()`` output is compared with stdout and its ``message()`` output with stderr.
"""

import collections.abc
import re
import typing
import warnings

import pytest
import refutil

import metafopy

REF = refutil.load_phase7('verbose')
CELLS: dict[str, typing.Any] = dict(ai='tpos', bi='tneg', ci='cpos', di='cneg')


def norm(text: str) -> str:
    """Drop the leading and trailing blank lines, which ``capture.output()`` does not keep reliably."""
    return text.strip('\n')


def run(capsys: pytest.CaptureFixture[str], fn: collections.abc.Callable[[], object]) -> tuple[str, str]:
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fn()
    got = capsys.readouterr()
    return got.out, got.err


def fits() -> dict[str, collections.abc.Callable[[], object]]:
    d = refutil.bcg()
    raw = metafopy.datasets.load('bcg')
    res = metafopy.rma_uni('yi', 'vi', data=d)
    hack = metafopy.datasets.load('hackshaw1998')
    return {
        'uni_reml_1': lambda: metafopy.rma_uni('yi', 'vi', data=d, verbose=True),
        'uni_reml_2': lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=d, verbose=2),
        'uni_pm': lambda: metafopy.rma_uni('yi', 'vi', data=d, method='PM', verbose=True),
        'uni_dl': lambda: metafopy.rma_uni('yi', 'vi', data=d, method='DL', verbose=2),
        'mh': lambda: metafopy.rma_mh(measure='OR', **CELLS, data=raw, verbose=True),
        'peto': lambda: metafopy.rma_peto(**CELLS, data=raw, verbose=True),
        'confint_qp': lambda: metafopy.confint(res, verbose=True),
        'confint_pl': lambda: metafopy.confint(res, type='pl', verbose=True),
        'trimfill': lambda: metafopy.trimfill(metafopy.rma_uni('yi', 'vi', data=hack, method='FE'), verbose=True),
        'fsn': lambda: metafopy.fsn(res, verbose=True),
    }


@pytest.mark.parametrize('key', ['uni_reml_1', 'uni_reml_2', 'uni_pm', 'uni_dl', 'mh', 'peto', 'confint_qp', 'confint_pl', 'trimfill'])
def test_verbose_matches_r(key: str, capsys: pytest.CaptureFixture[str]) -> None:
    out, err = run(capsys, fits()[key])
    assert norm(out) == norm(REF[key]['stdout'])
    assert norm(err) == norm(REF[key]['stderr'])


def test_verbose_fsn(capsys: pytest.CaptureFixture[str]) -> None:
    # metafopy searches the integers, so it tries other values of N than R's uniroot: the lines have R's form, the N values are integers, and
    # the search ends at the returned N or a neighbour (docs/differences.md, "fsn(): Fisher pooling and the General type")
    res = metafopy.rma_uni('yi', 'vi', data=refutil.bcg())
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = metafopy.fsn(res, verbose=True)
    got = capsys.readouterr()
    assert norm(got.err) == norm(REF['fsn']['stderr'])
    lines = [line for line in norm(got.out).splitlines() if line.startswith('fsnum')]
    assert lines
    pattern = re.compile(r'^fsnum = +(\d+)  est = [ -]\d\.\d+  tau2 = \d\.\d+  pval = \d\.\d+  alpha = \d\.\d+  diff = [ -]\d\.\d+ $')
    tried = []
    for line in lines:
        matched = pattern.match(line)
        assert matched is not None, line
        tried.append(int(matched.group(1)))
    assert abs(tried[-1] - result.fsnum) <= 1
    assert result.fsnum in tried


def test_verbose_glmm(capsys: pytest.CaptureFixture[str]) -> None:
    # the trace of glm()/glmer() has no counterpart, so only the progress messages are compared
    _, err = run(capsys, lambda: metafopy.rma_glmm(measure='OR', **CELLS, data=metafopy.datasets.load('bcg'), verbose=1))
    assert norm(err) == norm(REF['glmm']['stderr'])
    _, err = run(capsys, lambda: metafopy.rma_glmm(measure='OR', **CELLS, data=metafopy.datasets.load('bcg'), verbose=2))
    assert err.splitlines()[0] == 'Extracting the data and computing yi/vi values ...'
    assert 'Preparing the output ...' in err


def test_verbose_mv(capsys: pytest.CaptureFixture[str]) -> None:
    # the optimiser differs from R's nlminb, so the trace is checked for its key lines only
    d = metafopy.datasets.load('konstantopoulos2011')
    out, err = run(capsys, lambda: metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=d, verbose=True))
    assert 'Variance Components in Model:' in out
    assert err == ''
    _, err = run(capsys, lambda: metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=d, verbose=2))
    assert 'Model fitting ...' in err
    res = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=d)
    out, _ = run(capsys, lambda: metafopy.confint(res, verbose=True))
    assert 'Obtaining CI for sigma2 = 1' in out
    assert 'Obtaining CI for sigma2 = 2' in out
    assert 'LRT - objective =' in out


def test_verbose_off(capsys: pytest.CaptureFixture[str]) -> None:
    out, err = run(capsys, lambda: metafopy.rma_uni('yi', 'vi', data=refutil.bcg()))
    assert (out, err) == ('', '')
