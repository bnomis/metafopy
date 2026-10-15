"""Helpers for comparing emmeans summaries with the fixtures written by tests/r/generate_emmeans.R."""

import json
import math
import pathlib
import typing

import numpy
import pandas

import metafopy
import metafopy.methods.emm_summary

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase9' / 'emmeans.json').read_text(encoding='utf-8'))
RTOL = 1e-7
# rma.mv, rma.glmm and rma.ls estimates come from an optimiser, so they agree with R to its convergence tolerance only
OPTIM = 1e-4
# rma.glmm fits agree with R's lme4 to about 1e-3
GLMM = 2e-3
GLMM_ATOL = 1e-3
# the mvt adjustment integrates with a quasi-Monte Carlo rule in both R and SciPy
MVT = 5e-3


def as_list(value: typing.Any) -> list[typing.Any]:
    return value if isinstance(value, list) else [value]


def check(got: metafopy.methods.emm_summary.EmmSummary, key: str, rtol: float = RTOL, atol: float = 1e-9) -> None:
    exp = REF[key]
    frame = got.frame
    assert list(frame.columns) == exp['names'], key
    for name, col in zip(exp['names'], exp['columns'], strict=True):
        expected = as_list(col)
        actual = frame[name].tolist()
        assert len(actual) == len(expected), f'{key}.{name}'
        if (all(v is None or isinstance(v, int | float) for v in expected) and any(v is not None for v in expected)) or name == 'df':
            fill = math.inf if name == 'df' else math.nan
            numpy.testing.assert_allclose(
                numpy.array(actual, dtype=float),
                numpy.array([fill if v is None else v for v in expected], dtype=float),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                err_msg=f'{key}.{name}',
            )
        elif any(v is not None for v in expected):
            assert [str(v) for v in actual] == [str(v) for v in expected], f'{key}.{name}'


def bcg_lat() -> pandas.DataFrame:
    """The BCG trials with the latitude split into ``lat``, as in the R script."""
    bcg = metafopy.datasets.load('bcg')
    out = metafopy.escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg)
    out['lat'] = numpy.where(out['ablat'] >= 33, 'high', 'low')
    return out


def nested_data() -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """The BCG trials with the factors ``grp`` (in two levels), ``grp3`` (``'c'`` only for high latitudes) and, in the second frame,
    a ``grp`` and a ``sub`` that are nested in ``lat`` and in each other."""
    dat = bcg_lat()
    n = len(dat)
    pos = numpy.arange(1, n + 1)
    dat['grp'] = (['a', 'b'] * (n // 2 + 1))[:n]
    dat['grp3'] = numpy.where(dat['lat'] == 'high', numpy.array(['a', 'b', 'c'])[pos % 3], numpy.array(['a', 'b'])[pos % 2])
    cells = dat.copy()
    grp = [''] * n
    sub = [''] * n
    for lat in ('high', 'low'):
        for k, row in enumerate(numpy.flatnonzero(cells['lat'] == lat)):
            grp[row] = ['a', 'a', 'b', 'b'][k % 4]
            sub[row] = ['x', 'y', 'x', 'y'][k % 4]
    cells['grp'] = grp
    cells['sub'] = sub
    return dat, cells
