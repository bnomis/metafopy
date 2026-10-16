"""Audit 7: every data frame of metadat 1.6-0 is bundled; the shapes are those R reports."""

import pytest

import metafopy.datasets

SHAPES = {
    'anand1999': (34, 9),
    'anker2025': (6, 23),
    'axfors2021': (33, 13),
    'bakdash2021': (678, 12),
    'baker2009': (94, 6),
    'bartos2023': (48, 7),
    'bassler2004': (10, 6),
    'bauer2002': (5, 10),
    'begg1989': (20, 6),
    'berkey1998r': (10, 9),
    'besson2016': (473, 67),
    'bornmann2007': (66, 13),
    'bourassa1996': (96, 14),
    'cannon2006': (4, 16),
    'chiarito2020': (9, 29),
    'crede2010': (97, 8),
    'crisafulli2020': (26, 7),
    'curtin2002': (21, 6),
    'dagostino1998': (72, 16),
    'demir2022': (33, 10),
    'dogliotti2014': (44, 5),
    'dong2013': (99, 4),
    'dumouchel1994': (9, 7),
    'feng2019': (19, 13),
    'fine1993': (17, 11),
    'franchini2012': (7, 13),
    'frank2008': (4, 7),
    'greb2008': (14, 5),
    'greenland1992': (16, 4),
    'gurusamy2011': (29, 4),
    'hannum2020': (35, 11),
    'hartmannboyce2018': (136, 6),
    'higgins2008': (17, 7),
    'huber2000': (601, 3),
    'karner2014': (22, 33),
    'kearon1998': (34, 8),
    'landenberger2005': (58, 32),
    'lau1992': (33, 6),
    'linde2015': (66, 24),
    'linde2016': (201, 7),
    'lloyd2010': (16, 9),
    'lopez2019': (172, 23),
    'mccurdy2020': (1653, 26),
    'meeker2017': (10, 5),
    'michael2013': (12, 13),
    'moore1998': (37, 12),
    'nakagawa2007': (15, 4),
    'nissen2007': (42, 15),
    'obrien2003': (43, 13),
    'poole2003': (19, 9),
    'quan2000': (11, 5),
    'riley2003': (98, 5),
    'rivera2025': (10, 18),
    'roever2022': (37, 5),
    'spooner2002': (17, 10),
    'steurer2006': (4, 9),
    'stowe2010': (29, 14),
    'tannersmith2016': (113, 8),
    'thompson1999': (28, 9),
    'tian2009': (96, 16),
    'ursino2021': (49, 5),
    'vanhowe1999': (33, 6),
    'white2020': (186, 15),
    'woods2010': (8, 4),
}


@pytest.mark.parametrize('name', sorted(SHAPES))
def test_shape(name: str) -> None:
    assert f'dat.{name}' in metafopy.datasets.list_datasets()
    assert metafopy.datasets.load(name).shape == SHAPES[name]


def test_numeric_looking_text_columns_stay_text() -> None:
    assert metafopy.datasets.load('dat.yusuf1985')['id'].tolist()[:12:3] == ['1.1', '1.4', '1.7', '1.10']
    assert all(isinstance(v, str) for v in metafopy.datasets.load('dat.bauer2002')['threshold'])
