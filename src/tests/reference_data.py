"""Access to the JSON fixtures generated from R metafor by src/tests/r/generate_fixtures.R."""

import functools
import json
import pathlib
import typing

REFERENCE_DIR = pathlib.Path(__file__).parent / 'reference'


NOT_CASES = frozenset({'formula_names', 'formula_rank'})  # R model.matrix names and rank-deficient fits; read by their own tests, hold no case fits


def case_names() -> list[str]:
    return sorted(p.stem for p in REFERENCE_DIR.glob('*.json') if p.stem not in NOT_CASES)


@functools.cache
def load_case(name: str) -> dict[str, typing.Any]:
    with (REFERENCE_DIR / f'{name}.json').open() as fh:
        data: dict[str, typing.Any] = json.load(fh)
    return data
