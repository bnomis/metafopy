"""Loading of the bundled datasets."""

import functools
import importlib.resources
import json
import typing

import pandas

import metafopy.errors
import metafopy.metadata

# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
__all__ = ['list_datasets', 'load']

PACKAGE = 'metafopy.datasets.data'

# character (or factor) columns of metadat whose values look like numbers, which must not be read as numbers (yusuf1985 `id` is "5.10")
TEXT_COLUMNS = {'dat.bauer2002': ['threshold'], 'dat.yusuf1985': ['id']}


def list_datasets() -> list[str]:
    """Names of the bundled datasets, e.g. ``"dat.bcg"``."""
    files = importlib.resources.files(PACKAGE).iterdir()
    return sorted(f.name.removesuffix('.csv') for f in files if f.name.endswith('.csv'))


@functools.cache
def read(name: str) -> pandas.DataFrame:
    path = importlib.resources.files(PACKAGE).joinpath(f'{name}.csv')
    if not path.is_file():
        raise metafopy.errors.NotFoundError(f'unknown dataset {name!r}; see metafopy.datasets.list_datasets()')
    with path.open('rb') as fh:
        return pandas.read_csv(fh, dtype={column: str for column in TEXT_COLUMNS.get(name, [])})


@functools.cache
def escalc_attributes() -> dict[str, dict[str, typing.Any]]:
    """The ``escalc`` attributes of the datasets that metadat ships as ``escalc`` objects (measure, ``ni``, ``digits``, ``yi.names``, ``vi.names``)."""
    with importlib.resources.files(PACKAGE).joinpath('escalc-attrs.json').open('rb') as fh:
        loaded: dict[str, dict[str, typing.Any]] = json.load(fh)
    return loaded


def load(name: str) -> pandas.DataFrame:
    """Load a bundled metadat dataset by its R name (``"dat.bcg"``) or ``"bcg"``.

    Datasets that metadat stores as ``escalc`` objects (for example ``dat.konstantopoulos2011``) carry the attributes of such a frame in
    ``DataFrame.attrs`` as :func:`metafopy.escalc` stores them, so that ``rma`` takes the effect size measure and the sample sizes from them, as in
    R, where ``rma(yi, vi, data = dat.konstantopoulos2011)`` has ``measure`` ``"SMD"``.
    """
    if not name.startswith('dat.'):
        name = f'dat.{name}'
    frame = read(name).copy()
    stored = escalc_attributes().get(name)
    if stored is not None:
        digits = stored.get('digits')
        frame.attrs['measure'] = stored.get('measure')
        frame.attrs['slab'] = None
        frame.attrs['ni'] = stored.get('ni')
        frame.attrs[metafopy.metadata.ROWS_KEY] = list(frame.index)
        frame.attrs['digits'] = None if not digits else (digits[0] if len(set(digits)) == 1 else digits)
        frame.attrs['yi_names'] = stored['yi_names']
        frame.attrs['vi_names'] = stored['vi_names']
    return frame
