"""Helpers for comparing results with the JSON fixtures written by tests/r/generate_*.R."""

import functools
import json
import pathlib
import re
import typing

import numpy
import pandas

import metafopy.datasets

PHASE2_DIR = pathlib.Path(__file__).parent / 'reference' / 'phase2'
RTOL = 1e-8


@functools.cache
def load_phase2(name: str) -> dict[str, typing.Any]:
    data: dict[str, typing.Any] = json.loads((PHASE2_DIR / f'{name}.json').read_text(encoding='utf-8'))
    return data


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def as_list(value: typing.Any) -> list[typing.Any]:
    """jsonlite's ``auto_unbox`` turns length-one vectors into scalars."""
    return value if isinstance(value, list) else [value]


def close(actual: typing.Any, expected: typing.Any, what: str = '', rtol: float = RTOL, atol: float = 1e-9) -> None:
    exp = numpy.atleast_1d(numpy.array([numpy.nan if v is None else v for v in as_list(expected)], dtype=float))
    numpy.testing.assert_allclose(numpy.atleast_1d(numpy.asarray(actual, dtype=float)), exp, rtol=rtol, atol=atol, equal_nan=True, err_msg=what)


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def r_print(text: str) -> str:
    """R pads printed tables with trailing spaces."""
    return '\n'.join(line.rstrip() for line in text.split('\n'))


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def bcg() -> pandas.DataFrame:
    d = metafopy.datasets.load('bcg')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    a, b, c, e = (d[col].to_numpy(float) for col in ('tpos', 'tneg', 'cpos', 'cneg'))
    d['yi'] = numpy.log(a / (a + b)) - numpy.log(c / (c + e))
    d['vi'] = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + e)
    d['n'] = a + b + c + e
    d['label'] = d['author'] + ' ' + d['year'].astype(str)
    return d


def bcg_na() -> pandas.DataFrame:
    """The bcg data with ``yi`` of study 3 and ``ablat`` of study 7 set to NA."""
    d = bcg()
    d.loc[2, 'yi'] = numpy.nan
    d.loc[6, 'ablat'] = numpy.nan
    return d


def same_print(actual: str, expected: str) -> bool:
    """Compare printed tables; R's formatC shows tiny negative numbers as ``-0.0000`` while NumPy noise may flip the sign."""
    a, e = r_print(actual), r_print(expected)
    if a == e:
        return True

    def norm(text: str) -> str:
        return '\n'.join(' '.join(re.sub(r'-(0\.0+)(?![0-9])', r'\1', line).split()) for line in text.split('\n'))

    return norm(a) == norm(e)
