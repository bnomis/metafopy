"""Loading of the bundled datasets."""

import functools
import importlib.resources

import pandas

import metafopy.errors

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


def load(name: str) -> pandas.DataFrame:
    """Load a bundled metadat dataset by its R name (``"dat.bcg"``) or ``"bcg"``."""
    if not name.startswith('dat.'):
        name = f'dat.{name}'
    return read(name).copy()
