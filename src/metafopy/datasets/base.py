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

# character columns of metadat that hold strings pandas reads as missing ("None", "NA", ""); the value is the strings that are missing in the column
# (R's NA is an empty field in the CSV), all other strings are kept as they are
MISSING_STRINGS: dict[str, dict[str, tuple[str, ...]]] = {
    'dat.axfors2021': {'blinding_exact': ()},
    'dat.bakdash2021': {'DTIC.link': ('',)},
    'dat.linde2015': {'treatment3': ()},
    'dat.michael2013': {'Compensation': ()},
    'dat.stowe2010': {'t3': ()},
}


def list_datasets() -> list[str]:
    """Names of the bundled datasets.

    The datasets are those of the R package metadat that metafor's documentation uses, see https://wviechtb.github.io/metadat/.

    Returns:
        The sorted names, each with the `dat.` prefix, for example `'dat.bcg'`. Pass them to [load][metafopy.datasets.load].
    """
    files = importlib.resources.files(PACKAGE).iterdir()
    return sorted(f.name.removesuffix('.csv') for f in files if f.name.endswith('.csv'))


@functools.cache
def read(name: str) -> pandas.DataFrame:
    path = importlib.resources.files(PACKAGE).joinpath(f'{name}.csv')
    if not path.is_file():
        raise metafopy.errors.NotFoundError(f'unknown dataset {name!r}; see metafopy.datasets.list_datasets()')
    with path.open('rb') as fh:
        frame = pandas.read_csv(fh, dtype={column: str for column in TEXT_COLUMNS.get(name, [])})
    for column, missing in MISSING_STRINGS.get(name, {}).items():
        with path.open('rb') as fh:
            frame[column] = pandas.read_csv(fh, usecols=[column], dtype=str, keep_default_na=False, na_values=list(missing))[column]
    return frame


@functools.cache
def escalc_attributes() -> dict[str, dict[str, typing.Any]]:
    """The ``escalc`` attributes of the datasets that metadat ships as ``escalc`` objects (measure, ``ni``, ``digits``, ``yi.names``, ``vi.names``)."""
    with importlib.resources.files(PACKAGE).joinpath('escalc-attrs.json').open('rb') as fh:
        loaded: dict[str, dict[str, typing.Any]] = json.load(fh)
    return loaded


def load(name: str) -> pandas.DataFrame:
    """Load a bundled metadat dataset.

    Datasets that metadat stores as `escalc` objects (for example `dat.konstantopoulos2011`) carry the attributes of such a frame in
    `DataFrame.attrs`, as [escalc][metafopy.escalc] stores them, so that `rma` takes the effect size measure and the sample sizes from
    them, as in R.

    Args:
        name: name of the dataset, with or without the `dat.` prefix (`'dat.bcg'` and `'bcg'` are the same). See
            [list_datasets][metafopy.datasets.list_datasets].

    Returns:
        A new data frame with the columns of the R dataset (a copy that can be changed freely).

    Raises:
        metafopy.errors.NotFoundError: there is no dataset of that name.
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
