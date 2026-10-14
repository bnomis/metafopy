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


def test_gallery_datasets() -> None:
    pignon = metafopy.datasets.load('dat.pignon2000')
    assert pignon.shape == (65, 5)
    assert pignon.iloc[0].to_dict() == {'id': 44, 'trial': 'Pitie-74', 'OmE': -1.8, 'V': 14.5, 'grp': 1}
    damico = metafopy.datasets.load('dat.damico2009')
    assert damico.shape == (16, 8)
    assert damico.iloc[0].to_dict() == {'study': 'Abele-Horn', 'year': 1997, 'xt': 13, 'nt': 58, 'xc': 23, 'nc': 30, 'conceal': 0, 'blind': 0}
