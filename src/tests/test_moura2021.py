"""Multilevel phylogenetic meta-analysis of dat.moura2021 with rma_mv() checked against R metafor 5.2-1 (the tree's own branch lengths are used)."""

import json
import pathlib
import typing

import numpy
import pytest
import refutil

import metafopy
import metafopy.datasets

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase4' / 'moura2021.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def tree() -> metafopy.datasets.PhyloTree:
    return metafopy.datasets.load_tree('moura2021')


def test_vcv(tree: metafopy.datasets.PhyloTree) -> None:
    ref = REF['vcv_check']
    cov = tree.vcv()
    assert cov.shape == tuple(int(v) for v in ref['dim'])
    assert list(cov.index[:3]) == ref['tips']
    refutil.close(numpy.diag(cov.to_numpy())[:5], ref['diag_head'], 'diag(vcv)', rtol=1e-12)
    refutil.close(tree.vcv(corr=True).to_numpy()[:3, :3].reshape(-1, order='F'), ref['cor_corner'], 'cor corner', rtol=1e-10)


def test_fit(tree: metafopy.datasets.PhyloTree) -> None:
    data = metafopy.escalc(measure='ZCOR', ri='ri', ni='ni', data=metafopy.datasets.load('moura2021'))
    data['species.id.phy'] = data['species.id']
    res = metafopy.rma_mv(
        'yi',
        'vi',
        random=['~ 1 | study.id', '~ 1 | effect.size.id', '~ 1 | species.id', '~ 1 | species.id.phy'],
        R={'species.id.phy': tree.vcv(corr=True)},
        data=data,
        sparse=True,
    )
    assert res.k == REF['k']
    assert list(res.s_nlevels) == [int(v) for v in REF['s_nlevels']]
    assert list(res.s_names) == REF['s_names']
    refutil.close(res.fit_stats_reml.ll, REF['loglik'], 'logLik', rtol=1e-7)
    refutil.close(res.sigma2, REF['sigma2'], 'sigma2', rtol=1e-3, atol=1e-5)
    refutil.close(res.beta, REF['beta'], 'beta', rtol=1e-4)
    refutil.close(res.se, REF['se'], 'se', rtol=1e-3)
    refutil.close(res.QE, REF['QE'], 'QE', rtol=1e-6)
