"""vcalc, rcalc, bldiag and blsplit (and the nearPD port they use) checked against R metafor 5.2-1.

The fixture ``reference/phase4/vcalc.json`` is written by ``src/tests/r/generate_vcalc.R``.
"""

import collections.abc
import json
import pathlib
import typing
import warnings

import numpy
import numpy.typing
import pandas
import pytest
import refutil
import scipy.sparse

import metafopy
import metafopy.core.linalg
import metafopy.datasets
import metafopy.errors

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase4' / 'vcalc.json').read_text(encoding='utf-8'))
RTOL = 1e-10
ATOL = 1e-12

type Case = collections.abc.Callable[[], typing.Any]

# auditme: accept[literal-duplication] test cases spell out their own R arguments
RVARS = ('v1i', 'v2i')
# auditme: accept[non-alphabetical-literal] the order is that of the rows and columns of the correlation matrices
DIMNAMES = ('perf', 'acog', 'asom', 'conf')
TYPE_NAMES = tuple('ab')


def ref_frame(cols: dict[str, typing.Any]) -> pandas.DataFrame:
    """Rebuild a synthetic data frame written by the generator."""
    return pandas.DataFrame({name: values for name, values in cols.items()})


def named(key: str, names: collections.abc.Sequence[typing.Any]) -> pandas.DataFrame:
    """A fixture matrix as a data frame whose rows and columns carry ``names`` (R: dimnames)."""
    return pandas.DataFrame(numpy.array(REF[key], dtype=float), index=list(names), columns=list(names))


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float = RTOL) -> None:
    dense = actual.toarray() if isinstance(actual, scipy.sparse.csr_array) else actual
    got: numpy.typing.NDArray[numpy.float64] = numpy.asarray(dense, dtype=numpy.float64)
    got = numpy.where(numpy.isfinite(got), got, numpy.nan)  # jsonlite writes infinities as null
    numpy.testing.assert_allclose(got, numpy.array(expected, dtype=numpy.float64), rtol=rtol, atol=ATOL, equal_nan=True, err_msg=what)


def check_frame(frame: pandas.DataFrame | None, ref: dict[str, typing.Any], what: str) -> None:
    assert frame is not None
    assert list(frame.columns) == list(ref)
    for name, expected in ref.items():
        column = frame[name].to_numpy()
        values = expected if isinstance(expected, list) else [expected]
        if any(isinstance(v, str) for v in values):
            assert [str(v) for v in column] == values, f'{what}: {name}'
        else:
            close(column, values, f'{what}: {name}')


def recorded[T](call: collections.abc.Callable[[], T]) -> tuple[T, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call()
    return result, [str(w.message) for w in caught]


def assink() -> pandas.DataFrame:
    return metafopy.datasets.load('assink2016')


def berkey() -> pandas.DataFrame:
    return metafopy.datasets.load('berkey1998')


def craft() -> pandas.DataFrame:
    return metafopy.datasets.load('craft2003')


def knapp() -> pandas.DataFrame:
    return metafopy.datasets.load('knapp2017')


def cov2cor(x: numpy.ndarray) -> numpy.ndarray:
    d = numpy.sqrt(numpy.diag(x))
    out: numpy.ndarray = x / numpy.outer(d, d)
    numpy.fill_diagonal(out, 1.0)
    return out


def as_dense(x: typing.Any) -> numpy.ndarray:
    assert isinstance(x, numpy.ndarray)
    return x


def syn_time() -> pandas.DataFrame:
    return ref_frame(REF['syn_time'])


def syn_to() -> pandas.DataFrame:
    return ref_frame(REF['syn_to'])


def syn_grp() -> pandas.DataFrame:
    return ref_frame(REF['syn_grp'])


def syn_sub() -> pandas.DataFrame:
    return ref_frame(REF['syn_sub'])


def syn_na() -> pandas.DataFrame:
    return ref_frame(REF['syn_na'])


def syn_pd() -> pandas.DataFrame:
    return ref_frame(REF['syn_pd'])


def rho_obs() -> pandas.DataFrame:
    return named('rho_obs', [1, 2])


def rho_type() -> pandas.DataFrame:
    return named('rho_type', TYPE_NAMES)


def rho_bad() -> pandas.DataFrame:
    return named('rho_bad', [1, 2, 3])


def rho_bad4() -> pandas.DataFrame:
    return named('rho_bad4', [1, 2, 3, 4])


def knapp_vcalc() -> typing.Any:
    rho = named('knapp_rho', REF['knapp_rho_names'])
    return metafopy.vcalc('vi', 'study', grp1='group1', grp2='group2', w1='n_sz', w2='n_hc', obs=REF['knapp_task_diff'], rho=rho, data=knapp())


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
VCALC: dict[str, Case] = {
    'assink_rho06': lambda: metafopy.vcalc('vi', 'study', obs='esid', data=assink(), rho=0.6),
    'assink_rho0': lambda: metafopy.vcalc('vi', 'study', obs='esid', data=assink(), rho=0),
    'assink_cluster_only': lambda: metafopy.vcalc('vi', 'study', data=assink()),
    'assink_type_obs': lambda: metafopy.vcalc('vi', 'study', type='deltype', obs='esid', data=assink(), rho=numpy.array([0.7, 0.5])),
    'assink_type_obs_list': lambda: metafopy.vcalc('vi', 'study', type='deltype', obs='esid', data=assink(), rho=[0.7, 0.5]),
    'assink_type_only': lambda: metafopy.vcalc('vi', 'study', type='deltype', data=assink(), rho=0.3),
    'assink_vi1': lambda: metafopy.vcalc(1, 'study', obs='esid', data=assink(), rho=0.6),
    'assink_vectors': lambda: metafopy.vcalc(assink()['vi'].to_numpy(), assink()['study'].to_numpy(), obs=assink()['esid'].to_numpy(), rho=0.6),
    'assink_subset_cols': lambda: metafopy.vcalc('vi', 'study', subgroup='pubstatus', obs='esid', data=assink(), rho=0.6),
    'assink_nopdcheck': lambda: metafopy.vcalc('vi', 'study', obs='esid', data=assink(), rho=0.6, checkpd=False),
    'konst_obs': lambda: metafopy.vcalc('vi', 'district', obs='school', data=metafopy.datasets.load('konstantopoulos2011'), rho=0.4),
    'konst_obs_nearpd': lambda: metafopy.vcalc('vi', 'district', obs='school', data=metafopy.datasets.load('konstantopoulos2011'), rho=0.9, nearpd=True),
    'berkey_rvars': lambda: metafopy.vcalc(1, 'author', rvars=RVARS, data=berkey()),
    'berkey_rvars_colon': lambda: metafopy.vcalc(1, 'author', rvars=slice('v1i', 'v2i'), data=berkey()),
    'berkey_type': lambda: metafopy.vcalc('vi', 'trial', type='outcome', data=berkey(), rho=0.4),
    'berkey_rvars_ignored': lambda: metafopy.vcalc(1, 'author', rvars=RVARS, data=berkey(), rho=0.3, phi=0.5),
    'knapp': knapp_vcalc,
    'knapp_grp1_only': lambda: metafopy.vcalc('vi', 'study', grp1='group1', w1='n_sz', data=knapp()),
    'time_phi': lambda: metafopy.vcalc('vi', 'study', time1='time', phi=0.7, data=syn_time()),
    'time_phi_neg': lambda: metafopy.vcalc('vi', 'study', time1='time', phi=-0.5, data=syn_time()),
    'type_obs_time': lambda: metafopy.vcalc('vi', 'study', type='type', obs='obs', time1='time', phi=0.8, data=syn_to(), rho=[0.6, 0.3]),
    'type_obs_matrices': lambda: metafopy.vcalc('vi', 'study', type='type', obs='obs', data=syn_to(), rho=[rho_obs(), rho_type()]),
    'type_only_matrix': lambda: metafopy.vcalc('vi', 'study', type='type', data=syn_to(), rho=rho_type()),
    'obs_only_matrix': lambda: metafopy.vcalc('vi', 'study', obs='obs', data=syn_to(), rho=rho_obs()),
    'obs_only_list': lambda: metafopy.vcalc('vi', 'study', obs='obs', data=syn_to(), rho=[0.45]),
    'grp_w': lambda: metafopy.vcalc('vi', 'study', grp1='g1', grp2='g2', w1='n1', w2='n2', data=syn_grp()),
    'grp_w_same': lambda: metafopy.vcalc('vi', 'study', grp1='g1', grp2='g2', w1='n1', data=syn_grp()),
    'grp_w_time': lambda: metafopy.vcalc('vi', 'study', grp1='g1', grp2='g2', w1='n1', w2='n2', time1='t1', time2='t2', phi=0.6, data=syn_grp()),
    'grp_w_time_obs': lambda: metafopy.vcalc(
        'vi', 'study', grp1='g1', grp2='g2', w1='n1', w2='n2', time1='t1', time2='t2', phi=0.6, obs='t1', rho=0.5, data=syn_grp()
    ),
    'grp_time1_only': lambda: metafopy.vcalc('vi', 'study', grp1='g1', time1='t1', phi=0.6, data=syn_grp()),
    'vi_na': lambda: metafopy.vcalc('vi', 's', obs='o', rho=0.5, data=syn_na()),
    'vi_inf': lambda: metafopy.vcalc('vi', 's', obs='o', rho=0.5, data=syn_na().assign(vi=[0.1, numpy.inf, 0.3, 0.4, 0.5])),
    'vi_neg': lambda: metafopy.vcalc('vi', 's', obs='o', rho=0.5, data=syn_na().assign(vi=[0.1, -0.2, 0.3, 0.4, 0.5])),
    'subgroup': lambda: metafopy.vcalc('vi', 'study', subgroup='sub', obs='obs', data=syn_sub(), rho=0.5),
    'pd_warn': lambda: metafopy.vcalc('vi', 'study', obs='obs', data=syn_pd(), rho=rho_bad4()),
    'pd_nocheck': lambda: metafopy.vcalc('vi', 'study', obs='obs', data=syn_pd(), rho=rho_bad4(), checkpd=False),
    'pd_nearpd': lambda: metafopy.vcalc('vi', 'study', obs='obs', data=syn_pd(), rho=rho_bad4(), nearpd=True),
    'pd_nearpd_nocheck': lambda: metafopy.vcalc('vi', 'study', obs='obs', data=syn_pd(), rho=rho_bad4(), nearpd=True, checkpd=False),
    'pd_one_cluster': lambda: metafopy.vcalc('vi', 'study', obs='obs', data=syn_pd().iloc[:3], rho=rho_bad(), nearpd=True),
}


@pytest.mark.parametrize('name', sorted(VCALC))
def test_vcalc(name: str) -> None:
    result, caught = recorded(VCALC[name])
    assert isinstance(result, numpy.ndarray)
    assert result.dtype == numpy.float64
    close(result, REF[name]['V'], name)
    assert caught == refutil.as_list(REF[name]['warnings']), name


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
SPARSE: dict[str, Case] = {
    'assink_sparse': lambda: metafopy.vcalc('vi', 'study', obs='esid', data=assink(), rho=0.6, sparse=True),
    'vi_na_sparse': lambda: metafopy.vcalc('vi', 's', obs='o', rho=0.5, data=syn_na(), sparse=True),
    'grp_w_sparse': lambda: metafopy.vcalc('vi', 'study', grp1='g1', grp2='g2', w1='n1', w2='n2', data=syn_grp(), sparse=True, checkpd=False),
    'subgroup_sparse': lambda: metafopy.vcalc('vi', 'study', subgroup='sub', obs='obs', data=syn_sub(), rho=0.5, sparse=True),
}


@pytest.mark.parametrize('name', sorted(SPARSE))
def test_vcalc_sparse(name: str) -> None:
    result = SPARSE[name]()
    assert isinstance(result, scipy.sparse.csr_array)
    assert REF[name]['is_sparse']
    close(result, REF[name]['V'], name)


def test_vcalc_sparse_equals_dense() -> None:
    dense = metafopy.vcalc('vi', 'study', obs='esid', data=assink(), rho=0.6)
    sparse = SPARSE['assink_sparse']()
    numpy.testing.assert_array_equal(as_dense(dense), sparse.toarray())
    assert sparse.nnz < dense.size


def test_vcalc_rvars_ignored_warning() -> None:
    _, one = recorded(lambda: metafopy.vcalc(1, 'author', rvars=RVARS, data=berkey(), rho=0.3))
    assert one == refutil.as_list(REF['warnings_vcalc']['rvars_ignored_one'])


def test_vcalc_rvars_missing_warnings() -> None:
    d = berkey()
    d.loc[1, 'v1i'] = numpy.nan
    _, caught = recorded(lambda: metafopy.vcalc(1, 'author', rvars=RVARS, data=d))
    assert caught == refutil.as_list(REF['warnings_vcalc']['rvars_missing'])
    d = berkey()
    d.loc[0, 'v2i'] = numpy.nan
    _, caught = recorded(lambda: metafopy.vcalc(1, 'author', rvars=RVARS, data=d))
    assert caught == refutil.as_list(REF['warnings_vcalc']['rvars_missing_lower'])


def test_vcalc_rvars_clusters_not_contiguous() -> None:
    """metafor places the blocks in order of first appearance (and so misaligns them); metafopy puts a block at its cluster's rows."""
    d = berkey()
    sorted_v = as_dense(metafopy.vcalc(1, 'author', rvars=RVARS, data=d))
    order = numpy.array([0, 2, 1, 3, 4, 6, 5, 7, 8, 9])
    shuffled = d.iloc[order].reset_index(drop=True)
    v = as_dense(metafopy.vcalc(1, 'author', rvars=RVARS, data=shuffled))
    inverse = numpy.argsort(order)
    close(v[numpy.ix_(inverse, inverse)], sorted_v, 'shuffled rows')
    close(v[numpy.ix_(inverse, inverse)][:2, :2], sorted_v[:2, :2], 'first cluster')


def test_vcalc_rvars_by_position() -> None:
    by_name = metafopy.vcalc(1, 'author', rvars=RVARS, data=berkey())
    by_position = metafopy.vcalc(1, 'author', rvars=[7, 8], data=berkey())
    numpy.testing.assert_array_equal(as_dense(by_name), as_dense(by_position))


@pytest.mark.parametrize('key', ['assink_retdat', 'subgroup_retdat'])
def test_vcalc_retdat(key: str) -> None:
    if key == 'assink_retdat':
        d = assink()
        d = d[d['study'].isin([1, 2, 3])].reset_index(drop=True)
        result = metafopy.vcalc('vi', 'study', obs='esid', data=d, rho=0.6, retdat=True)
    else:
        result = metafopy.vcalc('vi', 'study', subgroup='sub', obs='obs', data=syn_sub(), rho=0.5, retdat=True)
    assert isinstance(result, pandas.DataFrame)
    ref = REF[key]
    assert list(result.columns[:10]) == list(ref['cols'])
    check_frame(result.iloc[:, :10], ref['cols'], key)
    close(result.iloc[:, 10:].to_numpy(), ref['V'], key)
    assert list(result.columns[10:]) == ref['vnames']


def test_vcalc_data_mapping() -> None:
    d = syn_to()
    from_frame = metafopy.vcalc('vi', 'study', obs='obs', data=d, rho=0.5, checkpd=False)
    from_dict = metafopy.vcalc('vi', 'study', obs='obs', data={c: d[c].to_numpy() for c in d.columns}, rho=0.5, checkpd=False)
    numpy.testing.assert_array_equal(as_dense(from_frame), as_dense(from_dict))


def test_vcalc_blsplit() -> None:
    v = as_dense(metafopy.vcalc('vi', 'study', obs='esid', data=assink(), rho=0.6))
    study = assink()['study'].to_numpy()
    parts = metafopy.blsplit(v, study, numpy.round, [4])
    for key, expected in REF['assink_blsplit'].items():
        close(parts[int(key)], expected, f'blsplit {key}')
    close(metafopy.blsplit(v, study, cov2cor)[16], REF['assink_blsplit_cov2cor']['16'], 'cov2cor')
    sorted_parts = metafopy.blsplit(v, study, numpy.round, [3], sort=True)
    for key, expected in REF['assink_blsplit_sort'].items():
        close(sorted_parts[int(key)], expected, f'blsplit sorted {key}')
    assert [str(k) for k in metafopy.blsplit(v, study)] == REF['assink_blsplit_names']
    assert [str(k) for k in metafopy.blsplit(v, study, sort=True)] == REF['assink_blsplit_names_sorted']


def test_vcalc_berkey_blsplit_fun() -> None:
    v = as_dense(VCALC['berkey_rvars']())
    parts = metafopy.blsplit(v, berkey()['author'].to_numpy(), lambda x: numpy.round(cov2cor(x), 2))
    for key, expected in REF['berkey_blsplit_fun'].items():
        close(parts[key], expected, key)


def test_vcalc_knapp_cov2cor() -> None:
    v = as_dense(knapp_vcalc())
    close(cov2cor(metafopy.blsplit(v, knapp()['study'].to_numpy())[3]), REF['knapp_cov2cor_study3'], 'knapp study 3')


def test_vcalc_knapp_task_diff() -> None:
    """The task.diff variable of the vcalc() example can be rebuilt with pandas (factor levels sort as in the C locale)."""
    d = knapp()
    out: list[str] = []
    for _, x in d.groupby('study', sort=True):
        task = pandas.factorize(x['task'], sort=True)[0] + 1
        diff = (pandas.factorize(x['difficulty'], sort=True, use_na_sentinel=True)[0] + 1).astype(float)
        diff[diff == 0] = 1
        out.extend(f'{t}.{int(df)}' for t, df in zip(task, diff, strict=True))
    assert out == REF['knapp_task_diff']


def test_nearpd_direct() -> None:
    ref = REF['nearpd_direct']
    res = metafopy.core.linalg.near_pd(numpy.array(REF['rho_bad4'], dtype=float), corr=True)
    close(res.mat, ref['mat'], 'nearPD mat')
    close(res.eigenvalues, ref['eigenvalues'], 'nearPD eigenvalues')
    close(res.norm_f, ref['normF'], 'nearPD normF')
    close(res.rel_tol, ref['rel_tol'], 'nearPD rel.tol')
    assert res.iterations == ref['iterations']
    assert res.converged == ref['converged']
    assert metafopy.core.linalg.is_positive_definite(res.mat)


def test_nearpd_keep_diag_and_asymmetric() -> None:
    ref = REF['nearpd_direct_keepdiag']
    res = metafopy.core.linalg.near_pd(numpy.array([[2, -1, 0], [-1, 2, -1.5], [0, -1.5, 1]], dtype=float), keep_diag=True)
    close(res.mat, ref['mat'], 'keepDiag mat')
    close(res.eigenvalues, ref['eigenvalues'], 'keepDiag eigenvalues')
    assert res.iterations == ref['iterations']
    ref = REF['nearpd_direct_asym']
    res = metafopy.core.linalg.near_pd(numpy.array([[1, 0.7, 0.3], [0.8, 1, 0.95], [0.3, 0.9, 1]], dtype=float), corr=True)
    close(res.mat, ref['mat'], 'asymmetric mat')
    assert res.iterations == ref['iterations']
    assert res.converged == ref['converged']


def test_nearpd_negative_semidefinite() -> None:
    with pytest.raises(metafopy.errors.EstimationError, match='negative semi-definite'):
        metafopy.core.linalg.near_pd(-numpy.eye(3))


def test_nearpd_did_not_converge_warns() -> None:
    with pytest.warns(UserWarning, match=r"'nearPD\(\)' did not converge in 1 iterations"):
        metafopy.core.linalg.near_pd(numpy.array(REF['rho_bad4'], dtype=float), corr=True, maxit=1)


def test_vcalc_nearpd_failure_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    # metafor wraps nearPD() in try(); a failure leaves the matrix as it is and warns

    def fail(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise metafopy.errors.EstimationError('Matrix seems negative semi-definite')

    monkeypatch.setattr(metafopy.core.linalg, 'near_pd', fail)
    result, caught = recorded(VCALC['pd_nearpd'])
    assert caught == ['Using nearPD() failed in cluster 1.', 'Using nearPD() failed in cluster 2.']
    close(result, REF['pd_nocheck']['V'], 'unchanged')


def test_vcalc_properties() -> None:
    v = as_dense(metafopy.vcalc('vi', 'study', obs='esid', data=assink(), rho=0.6))
    d = assink()
    numpy.testing.assert_allclose(numpy.diag(v), d['vi'].to_numpy(), rtol=1e-14)
    numpy.testing.assert_array_equal(v, v.T)
    cross = d['study'].to_numpy()[:, None] != d['study'].to_numpy()[None, :]
    assert not bool(numpy.any(v[cross]))


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
RCALC_CRAFT: dict[str, Case] = {
    'craft': lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft()),
    'craft_rtoz': lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft(), rtoz=True),
    'craft_harmonic': lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft(), nfun='harmonic'),
    'craft_mean': lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft(), nfun='mean'),
    'craft_nfun_function': lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft(), nfun=lambda x: float(x.max()) - 1),
    'craft_sparse': lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft(), sparse=True),
    'craft_shuffled': lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft().iloc[numpy.array(REF['craft_shuffled']['order']) - 1]),
    'craft_noid': lambda: metafopy.rcalc('ri ~ var1 + var2', ni='ni', data=craft()[craft()['study'] == 1]),
    'craft_partial': lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft().drop(index=[1, 3])),
    'craft_string_ids': lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft().assign(study=lambda x: 's' + x['study'].astype(str))),
}


def craft_with(**changes: dict[int, typing.Any]) -> pandas.DataFrame:
    """The craft2003 data with the given values put at the given (0-based) row positions of the given columns."""
    d = craft()
    for column, edits in changes.items():
        if any(isinstance(v, str) or v is None for v in edits.values()):
            d[column] = d[column].astype(object)
        else:
            d[column] = d[column].astype(float)
        for position, value in edits.items():
            d.loc[position, column] = value
    return d


@pytest.mark.parametrize('name', sorted(RCALC_CRAFT))
def test_rcalc_formula(name: str) -> None:
    result = RCALC_CRAFT[name]()
    assert isinstance(result, metafopy.RCalcResult)
    if name == 'craft':
        check_frame(result.dat, REF['craft_dat']['cols'], name)
        assert list(typing.cast(pandas.DataFrame, result.dat).columns) == REF['craft_dat']['names']
        close(result.V, REF['craft_V'], name)
        return
    if name == 'craft_string_ids':
        check_frame(result.dat, REF[name]['cols'], name)
        return
    ref = REF[name]
    check_frame(result.dat, ref['cols'], name)
    if name == 'craft_noid':
        assert list(typing.cast(pandas.DataFrame, result.dat).columns) == ref['names']
    if name == 'craft_sparse':
        assert isinstance(result.V, scipy.sparse.csr_array)
        assert ref['is_sparse']
    else:
        assert isinstance(result.V, numpy.ndarray)
    close(result.V, ref['V'], name)


def test_rcalc_formula_with_missing_values() -> None:
    d = craft()
    d['ri'] = d['ri'].astype(float)
    d['ni'] = d['ni'].astype(float)
    d.loc[4, 'ri'] = numpy.nan
    d.loc[2, 'ni'] = numpy.nan
    result = metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=d)
    assert isinstance(result, metafopy.RCalcResult)
    ref = REF['craft_na']
    check_frame(result.dat, ref['cols'], 'craft_na')
    close(result.V, ref['V'], 'craft_na')


def test_rcalc_ni_vector() -> None:
    d = craft()
    result = metafopy.rcalc('ri ~ var1 + var2 | study', ni=d['ni'].to_numpy(), data=d)
    assert isinstance(result, metafopy.RCalcResult)
    close(result.V, REF['craft_ni_vector'], 'ni vector')


def test_rcalc_blocks_match_documented_example() -> None:
    result = metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft())
    assert isinstance(result, metafopy.RCalcResult)
    assert result.dat is not None
    parts = metafopy.blsplit(as_dense(result.V), result.dat['study'].to_numpy(), numpy.round, [4])
    assert list(parts) == REF['craft_blsplit_names']
    for key, expected in REF['craft_blsplit'].items():
        close(parts[key], expected, f'craft study {key}')


def test_rcalc_dimension_names_are_in_dat() -> None:
    result = metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft())
    assert isinstance(result, metafopy.RCalcResult)
    assert result.dat is not None
    names = [f'{s}.{p}' for s, p in zip(result.dat['study'], result.dat['var1.var2'], strict=True)]
    assert names == [f'{s}.{p}' for s, p in zip(REF['craft_dat']['cols']['study'], REF['craft_dat']['cols']['var1.var2'], strict=True)]
    assert REF['craft_V_dimnames'] == list(result.dat['var1.var2'])


def r1() -> pandas.DataFrame:
    return named('R1', DIMNAMES)


def r3() -> pandas.DataFrame:
    return named('R3', DIMNAMES)


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
RCALC_MATRIX: dict[str, Case] = {
    'rcalc_matrix': lambda: metafopy.rcalc(r1(), ni=142),
    'rcalc_matrix_rtoz': lambda: metafopy.rcalc(r1(), ni=142, rtoz=True),
    'rcalc_matrix_upper': lambda: metafopy.rcalc(r1(), ni=142, upper=True),
    'rcalc_small_n': lambda: metafopy.rcalc(r1(), ni=4),
    'rcalc_nonames': lambda: metafopy.rcalc(numpy.array([[1, 0.4], [0.4, 1]]), ni=30),
    'rcalc_colnames_unsorted': lambda: metafopy.rcalc(pandas.DataFrame(numpy.array(REF['R4'], dtype=float), columns=['b', 'a', 'c']), ni=40),
}


@pytest.mark.parametrize('name', sorted(RCALC_MATRIX))
def test_rcalc_matrix(name: str) -> None:
    result = RCALC_MATRIX[name]()
    assert isinstance(result, metafopy.RCalcResult)
    ref = REF[name]
    check_frame(result.dat, ref['cols'], name)
    close(result.V, ref['V'], name)


def test_rcalc_matrix_dimension_names() -> None:
    result = RCALC_MATRIX['rcalc_matrix']()
    assert isinstance(result, metafopy.RCalcResult)
    assert result.dat is not None
    assert list(result.dat['var1.var2']) == REF['rcalc_matrix']['names']


def test_rcalc_asymmetric_matrix_uses_one_triangle() -> None:
    x = pandas.DataFrame(numpy.array(REF['Rasym'], dtype=float), index=DIMNAMES, columns=DIMNAMES)
    for upper, key in ((False, 'lower'), (True, 'upper')):
        result = metafopy.rcalc(x, ni=50, upper=upper)
        assert isinstance(result, metafopy.RCalcResult)
        ref = REF['rcalc_asym'][key]
        check_frame(result.dat, ref['cols'], key)
        close(result.V, ref['V'], key)


def test_rcalc_one_by_one_matrix() -> None:
    result = metafopy.rcalc(numpy.ones((1, 1)), ni=10)
    assert isinstance(result, metafopy.RCalcResult)
    assert REF['rcalc_1x1'] == {'dat_null': True, 'v_null': True}
    assert result.dat is None
    assert result.V is None


def test_rcalc_list() -> None:
    result = metafopy.rcalc({'1': r1(), '3': r3()}, ni=[142, 37])
    assert isinstance(result, metafopy.RCalcResult)
    ref = REF['rcalc_list']
    check_frame(result.dat, ref['cols'], 'list')
    close(result.V, ref['V'], 'list')
    assert result.dat is not None
    assert list(result.dat['var1.var2']) == ref['names']
    by_sequence = metafopy.rcalc([r1(), r3()], ni=[142, 37])
    assert isinstance(by_sequence, metafopy.RCalcResult)
    check_frame(by_sequence.dat, REF['rcalc_list_unnamed']['cols'], 'unnamed list')


def test_rcalc_list_sparse() -> None:
    result = metafopy.rcalc({'1': r1(), '3': r3()}, ni=[142, 37], sparse=True)
    assert isinstance(result, metafopy.RCalcResult)
    assert isinstance(result.V, scipy.sparse.csr_array)
    assert REF['rcalc_list_sparse']['is_sparse']
    close(result.V, REF['rcalc_list_sparse']['V'], 'sparse list')


def test_rcalc_list_no_simplify() -> None:
    result = metafopy.rcalc({'1': r1(), '3': r3()}, ni=[142, 37], simplify=False)
    assert isinstance(result, dict)
    assert list(result) == list(REF['rcalc_list_nosimplify'])
    for key, ref in REF['rcalc_list_nosimplify'].items():
        check_frame(result[key].dat, ref['cols'], key)
        close(result[key].V, ref['V'], key)


def test_rcalc_list_with_one_by_one_matrix() -> None:
    result = metafopy.rcalc({'a': r1(), 'b': numpy.ones((1, 1)), 'c': r3()}, ni=[142, 10, 37])
    assert isinstance(result, metafopy.RCalcResult)
    ref = REF['rcalc_list_with_1x1']
    check_frame(result.dat, ref['cols'], '1x1 in list')
    close(result.V, ref['V'], '1x1 in list')


def test_rcalc_matches_documented_example_for_study_one() -> None:
    """The matrix route and the formula route agree for study 1 (the Olkin and Finn covariances are symmetric in the variable order)."""
    via_matrix = metafopy.rcalc(r1(), ni=142)
    via_formula = metafopy.rcalc('ri ~ var1 + var2', ni='ni', data=craft()[craft()['study'] == 1])
    assert isinstance(via_matrix, metafopy.RCalcResult)
    assert isinstance(via_formula, metafopy.RCalcResult)
    assert via_matrix.dat is not None
    assert via_formula.dat is not None
    assert sorted(via_matrix.dat['var1.var2']) == sorted(via_formula.dat['var1.var2'])
    close(numpy.sort(as_dense(via_matrix.V).ravel()), numpy.sort(as_dense(via_formula.V).ravel()), 'same entries')


# bldiag and blsplit


def a_mat() -> numpy.ndarray:
    return numpy.array(REF['bl_A'], dtype=float)


def b_mat() -> numpy.ndarray:
    return numpy.array(REF['bl_B'], dtype=float)


def c_mat() -> numpy.ndarray:
    return numpy.array(REF['bl_C'], dtype=float)


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
BLDIAG: dict[str, Case] = {
    'bldiag_two': lambda: metafopy.bldiag(a_mat(), b_mat()),
    'bldiag_list': lambda: metafopy.bldiag([a_mat(), b_mat(), a_mat()]),
    'bldiag_scalar': lambda: metafopy.bldiag(numpy.ones((3, 3)), 2),
    'bldiag_vector': lambda: metafopy.bldiag(numpy.array([1, 2, 3]), a_mat()),
    'bldiag_rect': lambda: metafopy.bldiag(a_mat(), c_mat()),
    'bldiag_empty': lambda: metafopy.bldiag(numpy.zeros((0, 0)), a_mat()),
    'bldiag_single': lambda: metafopy.bldiag(a_mat()),
    'bldiag_sparse_input': lambda: metafopy.bldiag(scipy.sparse.csr_array(a_mat()), b_mat()),
}


@pytest.mark.parametrize('name', sorted(BLDIAG))
def test_bldiag(name: str) -> None:
    result = BLDIAG[name]()
    assert isinstance(result, numpy.ndarray)
    numpy.testing.assert_array_equal(result, numpy.array(REF[name], dtype=float))


def test_bldiag_all_empty() -> None:
    result = metafopy.bldiag(numpy.zeros((0, 0)), numpy.zeros((0, 3)))
    assert list(result.shape) == REF['bldiag_all_empty']
    assert metafopy.bldiag().shape == (0, 0)


def test_bldiag_order() -> None:
    cluster = REF['bldiag_order']['cluster']
    numpy.testing.assert_array_equal(metafopy.bldiag(a_mat(), b_mat(), order=cluster), numpy.array(REF['bldiag_order']['V'], dtype=float))
    permutation = numpy.argsort(cluster, kind='stable')
    numpy.testing.assert_array_equal(
        metafopy.bldiag(a_mat(), b_mat(), order=permutation, order_is_permutation=True), numpy.array(REF['bldiag_order_perm'], dtype=float)
    )
    numpy.testing.assert_array_equal(metafopy.bldiag(a_mat(), b_mat(), order=['b', 'a', 'b', 'a']), numpy.array(REF['bldiag_order_str'], dtype=float))


def test_bldiag_errors() -> None:
    with pytest.raises(metafopy.errors.InputError) as excinfo:
        metafopy.bldiag(a_mat(), c_mat(), order=[1, 2, 3, 4, 5])
    assert str(excinfo.value) == REF['errors_bldiag']['order_not_square']
    with pytest.raises(metafopy.errors.InputError) as excinfo:
        metafopy.bldiag(a_mat(), b_mat(), order=[1, 2, 3])
    assert str(excinfo.value) == REF['errors_bldiag']['order_length']
    with pytest.raises(metafopy.errors.ArgumentError):
        metafopy.bldiag(numpy.ones((2, 2, 2)))


def v4() -> numpy.ndarray:
    return numpy.array(REF['V4'], dtype=float)


CL2 = ['p', 'p', 'q', 'q']


def test_blsplit() -> None:
    plain = metafopy.blsplit(v4(), CL2)
    assert list(plain) == ['p', 'q']
    for key, expected in REF['blsplit_plain'].items():
        close(plain[key], expected, key)
    for key, expected in REF['blsplit_round'].items():
        close(metafopy.blsplit(v4(), CL2, numpy.round, [1])[key], expected, key)
    for key, expected in REF['blsplit_fun'].items():
        close(metafopy.blsplit(v4(), CL2, lambda x: x / 2)[key], expected, key)
    dets = metafopy.blsplit(v4(), CL2, numpy.linalg.det)
    close([dets['p'], dets['q']], [REF['blsplit_det']['p'], REF['blsplit_det']['q']], 'det')


def test_blsplit_interleaved_and_sorted() -> None:
    parts = metafopy.blsplit(v4(), [2, 1, 2, 1])
    # auditme: accept[literal-duplication] test cases spell out their own R arguments
    assert [str(k) for k in parts] == ['2', '1']
    for key, expected in REF['blsplit_interleaved'].items():
        close(parts[int(key)], expected, key)
    assert [str(k) for k in metafopy.blsplit(v4(), [2, 1, 2, 1], sort=True)] == REF['blsplit_sorted']
    assert [str(k) for k in metafopy.blsplit(v4(), [10, 3, 10, 3])] == REF['blsplit_numeric_names']


def test_blsplit_sparse() -> None:
    parts = metafopy.blsplit(scipy.sparse.csr_array(v4()), CL2)
    assert all(isinstance(p, scipy.sparse.csr_array) for p in parts.values())
    assert all(refutil.as_list(REF['blsplit_sparse_class']))
    for key, expected in REF['blsplit_sparse'].items():
        close(parts[key], expected, key)


def test_blsplit_cluster_types() -> None:
    parts = metafopy.blsplit(v4(), numpy.array([1.0, 1.0, 2.0, 2.0]))
    assert list(parts) == [1.0, 2.0]
    assert all(isinstance(k, float) for k in parts)


# error paths: the messages are those of R


def message(call: Case, error: type[Exception]) -> str:
    with pytest.raises(error) as excinfo:
        call()
    return str(excinfo.value.args[0])


def err_d() -> pandas.DataFrame:
    return syn_to()


def with_na(values: typing.Any, where: typing.Any) -> numpy.ndarray:
    out = numpy.array(values, dtype=object)
    out[numpy.asarray(where)] = None
    return out


def obs_is_1() -> numpy.ndarray:
    return (err_d()['obs'] == 1).to_numpy()


def vc(**kwargs: typing.Any) -> Case:
    args: dict[str, typing.Any] = {'vi': 'vi', 'cluster': 'study', 'data': err_d()}
    args.update(kwargs)
    return lambda: metafopy.vcalc(**args)


def na_float(column: str) -> numpy.ndarray:
    return numpy.where(obs_is_1(), numpy.nan, err_d()[column].to_numpy())


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
ERRORS_VCALC: dict[str, tuple[Case, type[Exception]]] = {
    'no_vi': (lambda: metafopy.vcalc(cluster='study', data=err_d()), metafopy.errors.InputError),
    'no_cluster': (lambda: metafopy.vcalc('vi', data=err_d()), metafopy.errors.InputError),
    'rvars_no_data': (lambda: metafopy.vcalc(1, 'study', rvars=['vi', 'vi']), metafopy.errors.InputError),
    'k1': (lambda: metafopy.vcalc(0.1, 1), metafopy.errors.InputError),
    'na_cluster': (vc(cluster=na_float('study')), metafopy.errors.InputError),
    'cluster_length': (vc(cluster=[1, 2, 3]), metafopy.errors.InputError),
    'na_subgroup': (vc(subgroup=na_float('obs')), metafopy.errors.InputError),
    'subgroup_length': (vc(subgroup=[1, 2]), metafopy.errors.InputError),
    'type_no_rho': (vc(type='type'), metafopy.errors.InputError),
    'na_type': (vc(type=with_na(err_d()['type'], obs_is_1()), rho=0.5), metafopy.errors.InputError),
    'type_length': (vc(type=TYPE_NAMES, rho=0.5), metafopy.errors.InputError),
    'obs_no_rho': (vc(obs='obs'), metafopy.errors.InputError),
    'na_obs': (vc(obs=na_float('obs'), rho=0.5), metafopy.errors.InputError),
    'obs_length': (vc(obs=[1, 2], rho=0.5), metafopy.errors.InputError),
    'grp2_only': (vc(grp2='type'), metafopy.errors.InputError),
    'na_grp1': (vc(grp1=with_na(err_d()['type'], obs_is_1())), metafopy.errors.InputError),
    'na_grp2': (vc(grp1='type', grp2=with_na(err_d()['type'], obs_is_1())), metafopy.errors.InputError),
    'grp1_length': (vc(grp1=[1, 2]), metafopy.errors.InputError),
    'grp2_length': (vc(grp1='type', grp2=[1, 2]), metafopy.errors.InputError),
    'time2_only': (vc(time2='time', phi=0.5), metafopy.errors.InputError),
    'time2_no_grp2': (vc(time1='time', time2='time', phi=0.5), metafopy.errors.InputError),
    'time_no_phi': (vc(time1='time'), metafopy.errors.InputError),
    'phi_range': (vc(time1='time', phi=-1.2), metafopy.errors.InputError),
    'na_time1': (vc(time1=na_float('time'), phi=0.5), metafopy.errors.InputError),
    'na_time2': (vc(grp1='type', grp2='type', time1='time', time2=na_float('time'), phi=0.5), metafopy.errors.InputError),
    'time1_length': (vc(time1=[1, 2], phi=0.5), metafopy.errors.InputError),
    'time1_not_numeric': (vc(time1='type', phi=0.5), metafopy.errors.InputError),
    'w2_only': (vc(w2='time'), metafopy.errors.InputError),
    'w2_no_grp2': (vc(w1='time', w2='time'), metafopy.errors.InputError),
    'na_w1': (vc(w1=na_float('time')), metafopy.errors.InputError),
    'na_w2': (vc(grp1='type', grp2='type', w1='time', w2=na_float('time')), metafopy.errors.InputError),
    'w1_length': (vc(w1=[1, 2]), metafopy.errors.InputError),
    'w2_length': (vc(grp1='type', grp2='type', w1='time', w2=[1, 2]), metafopy.errors.InputError),
    'w1_not_numeric': (vc(w1='type'), metafopy.errors.InputError),
    'both_vector_length': (vc(type='type', obs='obs', rho=0.5), metafopy.errors.InputError),
    'both_matrix': (vc(type='type', obs='obs', rho=rho_obs()), metafopy.errors.InputError),
    'both_list_length': (vc(type='type', obs='obs', rho=[rho_obs()]), metafopy.errors.InputError),
    'type_vector_length': (vc(type='type', rho=[0.5, 0.3]), metafopy.errors.InputError),
    'type_list_length': (vc(type='type', rho=[0.5, rho_type()]), metafopy.errors.InputError),
    'obs_vector_length': (vc(obs='obs', rho=[0.5, 0.3]), metafopy.errors.InputError),
    'obs_list_length': (vc(obs='obs', rho=[0.5, rho_obs()]), metafopy.errors.InputError),
    'rho_not_square': (vc(obs='obs', rho=pandas.DataFrame(numpy.full((2, 3), 0.5), index=[1, 2], columns=[1, 2, 3])), metafopy.errors.InputError),
    'rho_range': (vc(obs='obs', rho=1.2), metafopy.errors.InputError),
    'rho_matrix_range': (vc(obs='obs', rho=pandas.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=[1, 2], columns=[1, 2])), metafopy.errors.InputError),
    'rho_no_dimnames': (vc(obs='obs', rho=numpy.array([[1.0, 0.5], [0.5, 1.0]])), metafopy.errors.InputError),
    'rho_obs_missing': (vc(obs='obs', rho=pandas.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=[1, 3], columns=[1, 3])), metafopy.errors.InputError),
    'rho_type_missing': (vc(type='type', rho=pandas.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=['a', 'c'], columns=['a', 'c'])), metafopy.errors.InputError),
    'rvars_not_found': (vc(vi=1, rvars=['nope1', 'nope2']), metafopy.errors.InputError),
    'rvars_too_few': (vc(vi=1, rvars=['vi']), metafopy.errors.InputError),
    'rvars_too_many': (vc(vi=1, rvars=['study', 'type', 'obs', 'time', 'vi']), metafopy.errors.InputError),
    'missing_column': (vc(cluster='nope'), metafopy.errors.NotFoundError),
}


@pytest.mark.parametrize('name', sorted(ERRORS_VCALC))
def test_vcalc_errors(name: str) -> None:
    call, error = ERRORS_VCALC[name]
    assert message(call, error) == REF['errors_vcalc'][name]


def test_vcalc_rho_of_unsupported_type() -> None:
    with pytest.raises(metafopy.errors.ArgumentError, match="Argument 'rho' must either be a vector, a matrix, or a list"):
        metafopy.vcalc('vi', 'study', obs='obs', data=err_d(), rho={'a': 0.5})  # type: ignore[arg-type]


def test_vcalc_column_without_data() -> None:
    with pytest.raises(metafopy.errors.InputError, match='names a column, but no data was given'):
        metafopy.vcalc('vi', [1, 1])


def test_vcalc_phi_without_time_is_ignored() -> None:
    plain = metafopy.vcalc('vi', 'study', obs='obs', rho=0.5, data=err_d(), checkpd=False)
    ignored = metafopy.vcalc('vi', 'study', obs='obs', rho=0.5, phi=0.3, data=err_d(), checkpd=False)
    numpy.testing.assert_array_equal(as_dense(plain), as_dense(ignored))


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
ERRORS_RCALC: dict[str, tuple[Case, type[Exception]]] = {
    'bad_x': (lambda: metafopy.rcalc(numpy.arange(1.0, 4.0), ni=10), metafopy.errors.ArgumentError),
    'no_ni': (lambda: metafopy.rcalc(r1()), metafopy.errors.InputError),
    'no_data': (lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni'), metafopy.errors.InputError),
    'bad_nfun': (lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft(), nfun='median'), metafopy.errors.InputError),
    'four_vars': (lambda: metafopy.rcalc('ri ~ var1 + var2 + sport | study', ni='ni', data=craft()), metafopy.errors.InputError),
    'two_vars': (lambda: metafopy.rcalc('ri ~ var1', ni='ni', data=craft()), metafopy.errors.InputError),
    'na_var': (lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft_with(var1={1: None})), metafopy.errors.InputError),
    'ni_length': (lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni=[1, 2, 3], data=craft()), metafopy.errors.InputError),
    'na_id': (lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft_with(study={1: numpy.nan})), metafopy.errors.InputError),
    'ni_zero': (lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft_with(ni={1: 0})), metafopy.errors.InputError),
    'nfun_not_single': (
        lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft(), nfun=typing.cast(typing.Any, lambda x: numpy.array([1.0, 2.0]))),
        metafopy.errors.InputError,
    ),
    'identical_pair': (lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft_with(var2={1: 'asom'})), metafopy.errors.InputError),
    'r_range': (lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft_with(ri={1: 1.5})), metafopy.errors.InputError),
    'matrix_not_square': (lambda: metafopy.rcalc(numpy.full((2, 3), 0.5), ni=10), metafopy.errors.InputError),
    'matrix_range': (lambda: metafopy.rcalc(numpy.array([[1.0, 1.5], [1.5, 1.0]]), ni=10), metafopy.errors.InputError),
    'matrix_ni': (lambda: metafopy.rcalc(r1(), ni=0), metafopy.errors.InputError),
    'list_ni_length': (lambda: metafopy.rcalc([r1(), r3()], ni=10), metafopy.errors.InputError),
    'list_not_matrix': (lambda: metafopy.rcalc([r1(), numpy.arange(1.0, 4.0)], ni=[10, 10]), metafopy.errors.ArgumentError),
}


@pytest.mark.parametrize('name', sorted(ERRORS_RCALC))
def test_rcalc_errors(name: str) -> None:
    call, error = ERRORS_RCALC[name]
    assert message(call, error) == REF['errors_rcalc'][name]


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
PAIR_ERRORS = {
    'identical_pairs': craft_with(var2={1: 'asom', 2: 'conf'}),
    'duplicated_pair': craft_with(var1={1: 'acog'}, var2={1: 'perf'}),
    'duplicated_pairs': craft_with(var1={1: 'acog', 2: 'acog'}, var2={1: 'perf', 2: 'perf'}),
}


@pytest.mark.parametrize('name', sorted(PAIR_ERRORS))
def test_rcalc_pair_errors_with_several_pairs(name: str) -> None:
    data = PAIR_ERRORS[name]
    assert message(lambda: metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=data), metafopy.errors.InputError) == REF['errors_rcalc'][name]


def test_rcalc_rejects_non_matrix_x() -> None:
    with pytest.raises(metafopy.errors.ArgumentError, match="Argument 'x' must be either a formula, a matrix, or a list of matrices"):
        metafopy.rcalc(5, ni=10)  # type: ignore[arg-type]


def test_rcalc_ni_must_be_scalar_for_matrix() -> None:
    with pytest.raises(metafopy.errors.ArgumentError, match='single sample size'):
        metafopy.rcalc(r1(), ni=[10, 20])


def test_rcalc_nfun_prefix() -> None:
    full = metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft(), nfun='harmonic')
    short = metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=craft(), nfun='har')
    assert isinstance(full, metafopy.RCalcResult)
    assert isinstance(short, metafopy.RCalcResult)
    close(short.V, full.V, 'nfun prefix')


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
ERRORS_BLSPLIT: dict[str, tuple[Case, type[Exception]]] = {
    'no_cluster': (lambda: metafopy.blsplit(v4(), None), metafopy.errors.InputError),  # type: ignore[arg-type]
    'not_matrix': (lambda: metafopy.blsplit(numpy.arange(1.0, 5.0), CL2), metafopy.errors.ArgumentError),
    'not_symmetric': (lambda: metafopy.blsplit(numpy.array([[1.0, 3.0], [2.0, 4.0]]), CL2[:2]), metafopy.errors.InputError),
    'asymmetric': (lambda: metafopy.blsplit(numpy.arange(1.0, 17.0).reshape(4, 4), CL2), metafopy.errors.InputError),
    'cluster_length': (lambda: metafopy.blsplit(v4(), [1, 2, 3]), metafopy.errors.InputError),
}


@pytest.mark.parametrize('name', sorted(ERRORS_BLSPLIT))
def test_blsplit_errors(name: str) -> None:
    call, error = ERRORS_BLSPLIT[name]
    assert message(call, error) == REF['errors_blsplit'][name]


def test_package_exports() -> None:
    for name in ('RCalcResult', 'bldiag', 'blsplit', 'rcalc', 'vcalc'):
        assert name in metafopy.__all__
        assert hasattr(metafopy, name)
