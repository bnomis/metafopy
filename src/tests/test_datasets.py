import pytest

import metafopy.core.labelled
import metafopy.datasets


def test_list_and_load() -> None:
    names = metafopy.datasets.list_datasets()
    assert 'dat.bcg' in names
    bcg = metafopy.datasets.load('bcg')
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
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


def test_escalc_dataset_prints_as_r() -> None:
    """metadat ships ``dat.konstantopoulos2011`` as an ``escalc`` object, so R's ``print.escalc`` numbers its rows from 1 and gives ``yi`` and ``vi``
    the dataset's three decimals (R 4.6.2, metadat for metafor 5.2-1, printed 2026-11-06); audit cases ``matrix/identity/ks_dat`` and
    ``matrix/print/ks_dat``."""
    text = [line.rstrip() for line in str(metafopy.datasets.load('dat.konstantopoulos2011')).splitlines()]
    assert text[1:3] == ['   district school study year     yi    vi', '1        11      1     1 1976 -0.180 0.118']


def test_escalc_dataset_keeps_digits_by_kind() -> None:
    """``dat.mccurdy2020`` gives variances five decimals and the other kinds four, as its named ``digits`` attribute says: R prints ``1    0.8790 0.00068``."""
    dat = metafopy.datasets.load('dat.mccurdy2020')
    assert dat.attrs['digits'] == {'est': 4, 'se': 4, 'test': 4, 'pval': 4, 'ci': 4, 'var': 5, 'sevar': 4, 'fit': 4, 'het': 4}
    effects = metafopy.core.labelled.escalc_frame(dat[dat.attrs['yi_names'] + dat.attrs['vi_names']], dat.attrs)
    assert '1    0.8790 0.00068' in [line.rstrip() for line in str(effects).splitlines()]
