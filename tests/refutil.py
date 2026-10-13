"""Helpers for comparing results with the JSON fixtures written by tests/r/generate_*.R."""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metafopy import datasets

PHASE2_DIR = Path(__file__).parent / 'reference' / 'phase2'
RTOL = 1e-8


@cache
def load_phase2(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((PHASE2_DIR / f'{name}.json').read_text())
    return data


def as_list(value: Any) -> list[Any]:
    """jsonlite's ``auto_unbox`` turns length-one vectors into scalars."""
    return value if isinstance(value, list) else [value]


def close(actual: Any, expected: Any, what: str = '', rtol: float = RTOL, atol: float = 1e-9) -> None:
    exp = np.atleast_1d(np.array([np.nan if v is None else v for v in as_list(expected)], dtype=float))
    np.testing.assert_allclose(np.atleast_1d(np.asarray(actual, dtype=float)), exp, rtol=rtol, atol=atol, equal_nan=True, err_msg=what)


def r_print(text: str) -> str:
    """R pads printed tables with trailing spaces."""
    return '\n'.join(line.rstrip() for line in text.split('\n'))


def bcg() -> pd.DataFrame:
    d = datasets.load('bcg')
    a, b, c, e = (d[col].to_numpy(float) for col in ('tpos', 'tneg', 'cpos', 'cneg'))
    d['yi'] = np.log(a / (a + b)) - np.log(c / (c + e))
    d['vi'] = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + e)
    d['n'] = a + b + c + e
    d['label'] = d['author'] + ' ' + d['year'].astype(str)
    return d


def bcg_na() -> pd.DataFrame:
    """The bcg data with ``yi`` of study 3 and ``ablat`` of study 7 set to NA."""
    d = bcg()
    d.loc[2, 'yi'] = np.nan
    d.loc[6, 'ablat'] = np.nan
    return d


def same_print(actual: str, expected: str) -> bool:
    """Compare printed tables; R's formatC shows tiny negative numbers as ``-0.0000`` while NumPy noise may flip the sign."""
    a, e = r_print(actual), r_print(expected)
    if a == e:
        return True

    def norm(text: str) -> str:
        return '\n'.join(' '.join(re.sub(r'-(0\.0+)(?![0-9])', r'\1', line).split()) for line in text.split('\n'))

    return norm(a) == norm(e)
