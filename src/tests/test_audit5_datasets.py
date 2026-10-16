"""Audit 5 (area E): the metadat datasets added to the bundle.

The CSVs were produced by parsing the ``.rda`` files of metadat 1.6-0 in plain Python (the same reader reproduces the existing CSVs byte for byte).
Column sums are from that independent parse; ``tau2`` of aloe2013 is the value in the metafor test suite.
"""

import pytest

import metafopy
import metafopy.datasets

SHAPES = {
    'aloe2013': (5, 5),
    'baskerville2012': (23, 18),
    'bonett2010': (9, 6),
    'collins1985b': (9, 16),
    'furukawa2003': (17, 8),
    'hahn2001': (12, 5),
    'kalaian1996': (67, 12),
    'lehmann2018': (81, 49),
    'nielweise2008': (9, 7),
    'viechtbauer2021': (20, 6),
}

SUMS: dict[str, dict[str, float]] = {
    'aloe2013': {'n': 1247, 'tval': 15.26, 'preds': 51, 'R2': 1.639},
    'baskerville2012': {'year': 46042, 'score': 178, 'fumonths': 386, 'retention': 2044.3, 'smd': 15.2, 'se': 7.31},
    'bonett2010': {'ni': 1930, 'ai': 8.13, 'mi': 180},
    'collins1985b': {'pre.nti': 3759, 'pre.nci': 3183, 'pre.xti': 291, 'pre.xci': 345, 'ped.xti': 63, 'ped.xci': 60, 'ned.xti': 22, 'ned.xci': 21},
    'furukawa2003': {'Ne': 479, 'Me': 201.09, 'Se': 150.27, 'Nc': 423, 'Mc': 304.14, 'Sc': 141.97},
    'hahn2001': {'ai': 98, 'n1i': 1049, 'ci': 147, 'n2i': 1036},
    'kalaian1996': {'n1i': 3279, 'n2i': 5366, 'yi': 7.36, 'vi': 5.4236, 'hrs': 1093.3},
    'lehmann2018': {'Total.SampleSize': 5700, 'Red_N': 3093, 'Control_N': 3246},
    'nielweise2008': {'x1i': 48, 't1i': 25778, 'x2i': 77, 't2i': 25605},
    'viechtbauer2021': {'nTi': 3385, 'nCi': 3409, 'xTi': 1450, 'xCi': 1137, 'dose': 2350},
}


@pytest.mark.parametrize('name', sorted(SHAPES))
def test_listed_and_shape(name: str) -> None:
    assert f'dat.{name}' in metafopy.datasets.list_datasets()
    assert metafopy.datasets.load(name).shape == SHAPES[name]


@pytest.mark.parametrize('name', sorted(SUMS))
def test_column_sums(name: str) -> None:
    dat = metafopy.datasets.load(name)
    for col, total in SUMS[name].items():
        assert dat[col].sum() == pytest.approx(total, abs=1e-9)


def test_columns_and_first_rows() -> None:
    aloe = metafopy.datasets.load('dat.aloe2013')
    assert list(aloe.columns) == ['study', 'n', 'tval', 'preds', 'R2']
    assert aloe.iloc[0].to_dict() == {'study': 'Abu-Bader (2000)', 'n': 218, 'tval': 4.61, 'preds': 4, 'R2': 0.24}
    coll = metafopy.datasets.load('dat.collins1985b')
    assert list(coll.columns[:5]) == ['id', 'author', 'year', 'pre.nti', 'pre.nci']
    assert coll.iloc[0]['author'] == 'Weseley & Douglas'
    kal = metafopy.datasets.load('dat.kalaian1996')
    assert kal.iloc[0].to_dict() == {
        'id': 1,
        'study': 'Alderman & Powers (A)',
        'year': 1980,
        'n1i': 28,
        'n2i': 22,
        'outcome': 'verbal',
        'yi': 0.22,
        'vi': 0.0817,
        'hrs': 7,
        'ets': 1,
        'homework': 1,
        'type': 1,
    }
    bon = metafopy.datasets.load('dat.bonett2010')
    assert bon.iloc[0]['source'] == 'Atienza & Parris Stephens (2000)'
    assert list(metafopy.datasets.load('dat.lehmann2018').columns[-3:]) == ['pooled', 'yi', 'vi']


def test_aloe2013_tau2_matches_metafor_tests() -> None:
    dat = metafopy.escalc(measure='PCOR', ni='n', ti='tval', mi='preds', data=metafopy.datasets.load('dat.aloe2013'))
    res = metafopy.rma('yi', 'vi', data=dat, method='REML')
    assert round(float(res.tau2), 4) == 0.0298


def test_hackshaw_molloy_verified() -> None:
    assert metafopy.datasets.load('dat.hackshaw1998').shape == (37, 11)
    assert metafopy.datasets.load('dat.molloy2014').shape == (16, 10)
