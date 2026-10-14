"""profile(), ranef(), cooks_distance(), dfbetas(), rstudent(), rstandard() and hatvalues() of rma_mv fits checked against R metafor 5.2-1.

The fixture ``reference/phase7/mv_diagnostics.json`` is written by ``src/tests/r/generate_mv_diagnostics.R``.
"""

import collections.abc
import concurrent.futures
import functools
import json
import pathlib
import re
import typing
import warnings

import matplotlib
import matplotlib.pyplot
import numpy
import numpy.typing
import pandas
import pytest

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.methods.mv_diagnostics
import metafopy.methods.mv_profile

matplotlib.use('Agg')

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase7' / 'mv_diagnostics.json').read_text(encoding='utf-8'))

# the fits (and refits) agree with nlminb's only to optimiser tolerance, as in test_rma_mv.py
RTOL = 1e-4
ATOL = 1e-6
# closed-form functions of the full fit (hat matrix, BLUPs): only the fit's own variance components differ
CLOSED_RTOL = 1e-5

type Floats = numpy.typing.NDArray[numpy.float64]


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def floats(values: typing.Any) -> Floats:
    vals = values if isinstance(values, list) else [values]
    return numpy.array([numpy.nan if v is None else v for v in vals], dtype=float)


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float = RTOL, atol: float = ATOL) -> None:
    numpy.testing.assert_allclose(numpy.asarray(actual, dtype=float).reshape(-1), floats(expected), rtol=rtol, atol=atol, equal_nan=True, err_msg=what)


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def as_list(value: typing.Any) -> list[typing.Any]:
    return value if isinstance(value, list) else [value]


def frame_close(actual: pandas.DataFrame, ref: dict[str, typing.Any], what: str, rtol: float = RTOL, check_cols: bool = True, check_rows: bool = True) -> None:
    if check_rows:
        assert [str(i) for i in actual.index] == as_list(ref['rows']), what
    if check_cols:
        assert list(actual.columns) == as_list(ref['cols']), what
    for j, values in enumerate(ref['values'].values()):
        close(actual.iloc[:, j].to_numpy(dtype=float), values, f'{what} {actual.columns[j]}', rtol=rtol)


@functools.cache
# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def kon() -> pandas.DataFrame:
    return metafopy.datasets.load('konstantopoulos2011')


@functools.cache
def kres() -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon())


@functools.cache
def kmod() -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=kon())


@functools.cache
def kna() -> metafopy.RMAMVResult:
    data = kon().copy()
    data.loc[2, 'yi'] = numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=data)


@functools.cache
def bres() -> metafopy.RMAMVResult:
    data = metafopy.datasets.load('berkey1998')
    variances = [col for col in data.columns if re.fullmatch(r'v\di', col)]
    V = metafopy.bldiag([g[variances].to_numpy() for _, g in data.groupby('trial', sort=False)])
    return metafopy.rma_mv('yi', V, mods='~ 0 + outcome', random='~ outcome | trial', struct='UN', data=data)


MODELS: dict[str, collections.abc.Callable[[], metafopy.RMAMVResult]] = {'ber': bres, 'kmod': kmod, 'kna': kna, 'kon': kres}
CLUSTER = {'ber': 'trial', 'kmod': 'district', 'kna': 'district', 'kon': 'district'}
ALL = sorted(MODELS)
NA = {'ber': 'omit', 'kmod': 'omit', 'kna': 'exclude', 'kon': 'omit'}


def residuals_close(actual: typing.Any, ref: dict[str, typing.Any], what: str, rtol: float = RTOL) -> None:
    # auditme: accept[literal-duplication] the fields of metafor's list.rma
    for field in ('resid', 'se', 'z'):
        close(getattr(actual, field), ref[field], f'{what} {field}', rtol=rtol)
    assert list(actual.slab) == as_list(ref['slab']), what


def cluster_close(actual: metafopy.methods.mv_diagnostics.ClusterResidualResult, ref: dict[str, typing.Any], what: str, rtol: float = RTOL) -> None:
    residuals_close(actual.obs, ref['obs'], what, rtol=rtol)
    close(actual.X2, ref['cluster']['X2'], f'{what} X2', rtol=rtol)
    close(actual.k, ref['cluster']['k'], f'{what} k', rtol=0)
    assert list(actual.cluster_slab) == as_list(ref['cluster']['slab']), what


@pytest.mark.parametrize('name', ALL)
def test_cooks_cluster(name: str) -> None:
    out = metafopy.cooks_distance(MODELS[name](), cluster=CLUSTER[name], na_action=NA[name])
    ref = REF[name]['cooks_cluster']
    close(out.to_numpy(), ref['values'], 'cooks')
    assert list(out.index) == ref['names']


def test_cooks_cluster_fixed() -> None:
    out = metafopy.cooks_distance(kres(), cluster='district', reestimate=False)
    close(out.to_numpy(), REF['kon']['cooks_cluster_fixed']['values'], 'cooks')


@pytest.mark.parametrize('name', ['ber', 'kmod'])
def test_dfbetas_cluster(name: str) -> None:
    out = metafopy.dfbetas(MODELS[name](), cluster=CLUSTER[name])
    frame_close(out, REF[name]['dfbetas_cluster'], 'dfbetas', check_cols=False)
    assert list(out.columns) == list(MODELS[name]().coef_names)


def test_dfbetas_cluster_kon() -> None:
    frame_close(metafopy.dfbetas(kres(), cluster='district'), REF['kon']['dfbetas_cluster'], 'dfbetas')
    frame_close(metafopy.dfbetas(kres(), cluster=kon()['district'].to_numpy(), reestimate=False), REF['kon']['dfbetas_cluster_fixed'], 'dfbetas')


@pytest.mark.parametrize('name', ['ber', 'kmod', 'kna'])
def test_rstudent_cluster(name: str) -> None:
    out: object = metafopy.rstudent(MODELS[name](), cluster=CLUSTER[name], na_action=NA[name])
    assert isinstance(out, metafopy.methods.mv_diagnostics.ClusterResidualResult)
    cluster_close(out, REF[name]['rstudent_cluster'], name)


def test_rstudent_cluster_kon() -> None:
    for key, reestimate in (('rstudent_cluster', True), ('rstudent_cluster_fixed', False)):
        out: object = metafopy.rstudent(kres(), cluster='district', reestimate=reestimate)
        assert isinstance(out, metafopy.methods.mv_diagnostics.ClusterResidualResult)
        cluster_close(out, REF['kon'][key], key)
    assert isinstance(out, metafopy.methods.mv_diagnostics.ClusterResidualResult)
    assert '$cluster' in str(out)
    assert list(out.obs_frame().columns) == ['resid', 'se', 'z', 'cluster']


def test_rstudent_ber() -> None:
    residuals_close(metafopy.rstudent(bres()), REF['ber']['rstudent'], 'rstudent')


@pytest.mark.parametrize('name', ['ber', 'kmod', 'kon'])
def test_rstandard_cluster(name: str) -> None:
    out: object = metafopy.rstandard(MODELS[name](), cluster=CLUSTER[name])
    assert isinstance(out, metafopy.methods.mv_diagnostics.ClusterResidualResult)
    cluster_close(out, REF[name]['rstandard_cluster'], name, rtol=CLOSED_RTOL)


def test_rstandard() -> None:
    residuals_close(metafopy.rstandard(kres()), REF['kon']['rstandard'], 'kon', rtol=CLOSED_RTOL)
    residuals_close(metafopy.rstandard(kna(), na_action='exclude'), REF['kna']['rstandard'], 'kna', rtol=CLOSED_RTOL)


@pytest.mark.parametrize('name', ALL)
def test_hatvalues(name: str) -> None:
    x = MODELS[name]()
    close(x.hatvalues(na_action=NA[name]), REF[name]['hat']['values'], 'hat', rtol=CLOSED_RTOL)
    if 'hat_matrix' in REF[name]:
        close(x.hatvalues(type='matrix').T, REF[name]['hat_matrix'], 'hat matrix', rtol=CLOSED_RTOL)


def test_hatvalues_padding() -> None:
    assert kna().hatvalues('matrix', na_action='exclude').shape == (56, 56)
    with pytest.raises(metafopy.errors.InputError):
        kna().hatvalues(na_action='fail')
    with pytest.raises(metafopy.errors.InputError):
        kres().hatvalues('full')  # type: ignore[arg-type]


def ranef_close(out: typing.Any, ref: dict[str, typing.Any], what: str) -> None:
    assert isinstance(out, dict)
    assert list(out) == list(ref)
    for name, frame in ref.items():
        frame_close(out[name], frame, f'{what} {name}', rtol=CLOSED_RTOL)


@pytest.mark.parametrize('name', ALL)
def test_ranef(name: str) -> None:
    ranef_close(metafopy.ranef(MODELS[name](), na_action=NA[name]), REF[name]['ranef'], name)


def test_ranef_options() -> None:
    ranef_close(metafopy.ranef(kres(), expand=True), REF['kon']['ranef_expand'], 'expand')
    ranef_close(metafopy.ranef(kres(), transf=numpy.exp), REF['kon']['ranef_transf'], 'transf')
    ranef_close(metafopy.ranef(kres(), level=90), REF['kon']['ranef_level'], 'level')
    out: object = metafopy.ranef(kres(), vcov=True)
    assert isinstance(out, dict)
    assert out['vcov']['district'].shape == (11, 11)


def test_ranef_t() -> None:
    ranef_close(metafopy.ranef(metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon(), test='t')), REF['kt']['ranef'], 't')
    fit = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon(), test='t', dfs='contain')
    ranef_close(metafopy.ranef(fit), REF['kt']['ranef_contain'], 'contain')


def test_ranef_weights_and_fe() -> None:
    with pytest.raises(metafopy.errors.InputError, match='non-standard weights'):
        metafopy.ranef(metafopy.rma_mv('yi', 'vi', W=1 / kon()['vi'].to_numpy(), random='~ 1 | district/school', data=kon()))
    assert metafopy.ranef(metafopy.rma_mv('yi', 'vi', data=kon())) is None


def profile_close(out: typing.Any, ref: dict[str, typing.Any], what: str) -> None:
    assert isinstance(out, metafopy.methods.mv_profile.MVProfileResult)
    close(out.tau2, ref['vc'], f'{what} grid')
    close(out.ll, ref['ll'], f'{what} ll')
    close(out.ylim, ref['ylim'], f'{what} ylim')
    close(out.maxll, ref['maxll'], f'{what} maxll')
    close(out.vc, ref['est'], f'{what} vc')
    # auditme: accept[literal-duplication] the fields of metafor's profile.rma
    for key in ('beta', 'ci_lb', 'ci_ub'):
        frame_close(getattr(out, key), ref[key], f'{what} {key}', check_cols=False, check_rows=False)


def test_profile_kon() -> None:
    profile_close(metafopy.profile(kres(), sigma2=1, steps=5), REF['kon']['profile_sigma2_2'], 'sigma2')
    profile_close(metafopy.profile(kres(), sigma2=0, xlim=(0.01, 0.2), steps=4), REF['kon']['profile_xlim'], 'xlim')
    profile_close(metafopy.profile(kmod(), sigma2=0, steps=5), REF['kmod']['profile_sigma2_1'], 'mods')


def test_profile_all() -> None:
    out: object = metafopy.profile(kres(), steps=4)
    assert isinstance(out, list)
    assert [(p.comp, p.comp_pos) for p in out] == [('sigma2', 0), ('sigma2', 1)]
    profile_close(out[0], REF['kon']['profile_all']['1'], 'first')
    profile_close(out[1], REF['kon']['profile_all']['2'], 'second')


def test_profile_ber() -> None:
    profile_close(metafopy.profile(bres(), tau2=1, xlim=(0.003, 0.075), steps=4), REF['ber']['profile_tau2_2'], 'tau2')
    profile_close(metafopy.profile(bres(), rho=0, steps=5), REF['ber']['profile_rho'], 'rho')


def test_profile_parallel() -> None:
    serial = metafopy.profile(bres(), rho=0, steps=4)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        threaded = metafopy.profile(bres(), rho=0, steps=4, cl=pool)
    assert isinstance(serial, metafopy.methods.mv_profile.MVProfileResult)
    assert isinstance(threaded, metafopy.methods.mv_profile.MVProfileResult)
    numpy.testing.assert_array_equal(serial.ll, threaded.ll)


def test_profile_plot() -> None:
    fig = matplotlib.pyplot.figure()
    out = metafopy.profile(bres(), rho=0, steps=3, plot=True, ax=fig.gca())
    assert isinstance(out, metafopy.methods.mv_profile.MVProfileResult)
    assert fig.axes[0].get_title() == r'Profile Plot for $\rho$'
    matplotlib.pyplot.close(fig)
    fig = matplotlib.pyplot.figure()
    metafopy.profile(kres(), steps=3, plot=True, ax=fig.gca())
    assert [a.get_title() for a in fig.axes] == [r'Profile Plot for $\sigma_{1}^2$', r'Profile Plot for $\sigma_{2}^2$']
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize(
    ('kwargs', 'key'),
    [
        ({'tau2': 0}, 'err_profile_tau2'),
        ({'sigma2': 0, 'tau2': 0}, 'err_profile_two'),
        ({'sigma2': 2}, 'err_profile_none'),
        ({'sigma2': 0, 'xlim': (-1, 1)}, 'err_profile_xlim'),
    ],
)
def test_profile_errors(kwargs: dict[str, typing.Any], key: str) -> None:
    with pytest.raises(metafopy.errors.InputError, match=re.escape(REF['kon'][key])):
        metafopy.profile(kres(), **kwargs)


def test_errors() -> None:
    with pytest.raises(metafopy.errors.NotSupportedError, match=re.escape(REF['kon']['err_qqnorm'])):
        metafopy.qqnorm(kres())
    with pytest.raises(metafopy.errors.NotSupportedError, match="no applicable method for 'influence'"):
        metafopy.influence(kres())
    with pytest.raises(metafopy.errors.ArgumentError, match=re.escape('infl.rma.uni')):
        metafopy.plot_infl(kres())  # type: ignore[arg-type]
    with pytest.raises(metafopy.errors.InputError, match=re.escape(REF['kon']['err_cluster_len'])):
        metafopy.cooks_distance(kres(), cluster=[1, 2, 3])
    robust = metafopy.robust(kres(), cluster='district')
    with pytest.raises(metafopy.errors.NotSupportedError, match=re.escape('robust.rma')):
        metafopy.rstudent(robust)


def test_sparse() -> None:
    fit = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon(), sparse=True)
    ranef: object = metafopy.ranef(fit)
    assert isinstance(ranef, dict)
    frame_close(ranef['district'], REF['kon']['ranef']['district'], 'sparse ranef', rtol=CLOSED_RTOL)
    out: object = metafopy.rstandard(fit, cluster='district')
    assert isinstance(out, metafopy.methods.mv_diagnostics.ClusterResidualResult)
    cluster_close(out, REF['kon']['rstandard_cluster'], 'sparse', rtol=CLOSED_RTOL)


@pytest.mark.slow
def test_per_observation() -> None:
    close(metafopy.cooks_distance(kres()).to_numpy(), REF['kon']['cooks']['values'], 'cooks')
    frame_close(metafopy.dfbetas(kres()), REF['kon']['dfbetas'], 'dfbetas')
    residuals_close(metafopy.rstudent(kres()), REF['kon']['rstudent'], 'rstudent')


@pytest.mark.slow
def test_per_observation_na() -> None:
    out = metafopy.cooks_distance(kna(), na_action='exclude')
    close(out.to_numpy(), REF['kna']['cooks']['values'], 'cooks')
    assert list(out.index) == REF['kna']['cooks']['names']
    frame_close(metafopy.dfbetas(kna(), na_action='exclude'), REF['kna']['dfbetas'], 'dfbetas')
