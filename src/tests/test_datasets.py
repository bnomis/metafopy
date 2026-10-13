import pytest

import metafopy.datasets


def test_list_and_load() -> None:
    names = metafopy.datasets.list_datasets()
    assert 'dat.bcg' in names
    bcg = metafopy.datasets.load('bcg')
    assert list(bcg.columns[:3]) == ['trial', 'author', 'year']
    assert len(bcg) == 13


def test_load_returns_copy() -> None:
    a = metafopy.datasets.load('dat.bcg')
    a.loc[0, 'tpos'] = -1
    assert metafopy.datasets.load('dat.bcg').loc[0, 'tpos'] != -1


def test_unknown_dataset() -> None:
    with pytest.raises(KeyError):
        metafopy.datasets.load('dat.nope')
