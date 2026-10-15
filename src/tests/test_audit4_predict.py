"""predict() with predtype and newvi, newmods for models without moderators, na_action of fitted(), predict() and hatvalues(), and location-scale fits
with a missing scale value, checked against R metafor 5.2-1 (tests/r/generate_audit4_predict.R)."""

import collections.abc
import typing

import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.predict
import metafopy.models.rma_uni

matplotlib.use('Agg')

REF: dict[str, typing.Any] = refutil.load_phase12('audit4_predict')
ACTIONS: dict[str, metafopy.models.rma_uni.NAActionAll] = {'na.omit': 'omit', 'na.exclude': 'exclude', 'na.pass': 'pass'}
NEWMODS = [10, 30, 50]
FIELDS = ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub')
# auditme: accept[non-alphabetical-literal] the order follows the R fixture generator
TRANSFS = {'exp': numpy.exp}


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def dna(dat: pandas.DataFrame) -> pandas.DataFrame:
    out = dat.copy()
    out.loc[2, 'yi'] = numpy.nan
    out.loc[4, 'ablat'] = numpy.nan
    return out


@pytest.fixture(scope='module')
def bna() -> pandas.DataFrame:
    out = metafopy.datasets.load('dat.bcg').copy()
    out.loc[2, 'tpos'] = numpy.nan
    out.loc[4, 'cneg'] = numpy.nan
    return out


def close(actual: typing.Any, expected: typing.Any, what: str, rtol: float) -> None:
    """Compare with R's values (NA is None); a single R value stands for the same value in every row (R keeps ``pi.se`` of ``predtype="simple"`` as 0)."""
    got = numpy.atleast_1d(numpy.asarray(actual, dtype=float))
    if numpy.ndim(expected) >= 2:
        numpy.testing.assert_allclose(got, numpy.asarray(expected, dtype=float), rtol=rtol, atol=1e-9, equal_nan=True, err_msg=what)
        return
    if isinstance(expected, numpy.ndarray):
        want = expected.astype(float).reshape(-1)
    else:
        want = numpy.atleast_1d(numpy.array([numpy.nan if v is None else v for v in refutil.as_list(expected)], dtype=float))
    if want.size == 1 and got.size > 1:
        want = numpy.full(got.shape, want[0])
    numpy.testing.assert_allclose(got, want, rtol=rtol, atol=1e-9, equal_nan=True, err_msg=what)


def check_prediction(res: typing.Any, spec: dict[str, typing.Any], what: str, rtol: float, labels: bool = True) -> None:
    """Compare a prediction with R's; ``labels=False`` skips the labels."""
    for name in FIELDS:
        if name in ('pi_lb', 'pi_ub') and spec[name] is None and numpy.all(numpy.isnan(numpy.asarray(getattr(res, name), dtype=float))):
            continue
        close(getattr(res, name), spec[name], f'{what} {name}', rtol)
    # for the fitted values of a fit with studies left out R keeps pi.se for all rows of the subset (it does not drop the rows of the other values)
    if spec['pi_se'] is not None and res.pi_se is not None and numpy.size(spec['pi_se']) in (1, numpy.size(res.pi_se)):
        close(res.pi_se, spec['pi_se'], f'{what} pi_se', rtol)
    if spec['pi_dist'] is not None and res.pi_dist is not None:
        assert res.pi_dist == spec['pi_dist'], what
    if spec['pi_ddf'] is not None and res.pi_ddf is not None and numpy.size(spec['pi_ddf']) in (1, numpy.size(res.pred)):
        close(
            numpy.broadcast_to(res.pi_ddf, numpy.atleast_1d(res.pred).shape),
            numpy.broadcast_to(numpy.asarray(spec['pi_ddf'], dtype=float), numpy.atleast_1d(res.pred).shape),
            f'{what} pi_ddf',
            rtol,
        )
    if spec['prob'] is not None:
        close(res.prob, spec['prob'], f'{what} prob', rtol)
    slab = [spec['slab']] if isinstance(spec['slab'], str) else spec['slab']
    if labels and res.slab is not None and slab != ['']:
        assert list(res.slab) == slab, what


def kwargs_of(run: dict[str, typing.Any]) -> dict[str, typing.Any]:
    out: dict[str, typing.Any] = {}
    if run['args'] is None:
        return out
    names = run['args'] if isinstance(run['args'], list) else [run['args']]
    vals = run['vals']
    for name in names:
        value = vals[name]
        key = name.replace('.', '_')
        if key == 'transf':
            value = numpy.exp
        out[key] = value
    return out


def run_types(res: typing.Any, key: str, newmods: typing.Any, rtol: float) -> None:
    for i, run in enumerate(REF['types'][key]):
        kwargs = kwargs_of(run)
        if newmods is not None:
            kwargs['newmods'] = newmods
        check_prediction(metafopy.predict(res, **kwargs), run, f'{key} run {i} {run["args"]}', rtol)


def uni(frame: pandas.DataFrame, **kwargs: typing.Any) -> metafopy.RMAResult:
    return typing.cast(metafopy.RMAResult, metafopy.rma('yi', 'vi', data=frame, **kwargs))


KON = metafopy.datasets.load('konstantopoulos2011')


def models(dat: pandas.DataFrame) -> dict[str, collections.abc.Callable[[], tuple[typing.Any, typing.Any, float]]]:
    """The fits of the ``types`` fixture: the model, the ``newmods`` to predict for and the tolerance."""
    reml = uni(dat, mods='~ablat')
    mv_mods = metafopy.rma_mv('yi', 'vi', mods='~ablat', random='~1|trial', data=dat)
    cluster = dat['trial'].to_numpy()
    return {
        'uni_reml': lambda: (reml, NEWMODS, 1e-7),
        'uni_reml_fit': lambda: (reml, None, 1e-7),
        'uni_knha': lambda: (uni(dat, mods='~ablat', test='knha'), NEWMODS, 1e-7),
        'uni_ee': lambda: (uni(dat, mods='~ablat', method='EE'), NEWMODS, 1e-7),
        'uni_int': lambda: (uni(dat), None, 1e-7),
        'uni_fix_tau2': lambda: (uni(dat, mods='~ablat', tau2=0.1), NEWMODS, 1e-7),
        'mv_mods': lambda: (mv_mods, NEWMODS, 1e-5),
        'mv_t': lambda: (metafopy.rma_mv('yi', 'vi', mods='~ablat', random='~1|trial', data=dat, test='t'), NEWMODS, 1e-5),
        'mv_int': lambda: (metafopy.rma_mv('yi', 'vi', random='~1|trial', data=dat), None, 1e-5),
        'mv_kon': lambda: (
            metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=KON, test='t'),
            [-5, 0, 5],
            1e-4,
        ),
        'rob': lambda: (metafopy.robust(reml, cluster), NEWMODS, 1e-7),
        'rob_cs': lambda: (metafopy.robust(reml, cluster, club_sandwich=True), NEWMODS, 1e-6),
        'rob_mv': lambda: (metafopy.robust(mv_mods, cluster), NEWMODS, 1e-5),
        'sel': lambda: (metafopy.selmodel(reml, type='halfnorm'), NEWMODS, 1e-4),
        'sel_int': lambda: (metafopy.selmodel(uni(dat), type='halfnorm'), None, 1e-4),
        'mh': lambda: (metafopy.rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg')), None, 1e-7),
        'peto': lambda: (metafopy.rma_peto(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg')), None, 1e-7),
    }


@pytest.mark.parametrize('key', list(REF['types']))
def test_predtype_and_newvi(dat: pandas.DataFrame, key: str) -> None:
    res, newmods, rtol = models(dat)[key]()
    run_types(res, key, newmods, rtol)


def test_predtype_values(dat: pandas.DataFrame) -> None:
    """The prediction interval follows the formulas of ``predict.rma``: riley uses ``k - parms`` degrees of freedom, simple the normal distribution."""
    res = uni(dat, mods='~ablat')
    default = metafopy.predict(res, newmods=NEWMODS)
    riley = metafopy.predict(res, newmods=NEWMODS, predtype='riley')
    assert riley.pi_dist == 't'
    numpy.testing.assert_allclose(numpy.asarray(riley.pi_ddf), res.k - res.parms)
    numpy.testing.assert_allclose(riley.ci_lb, default.ci_lb)
    assert numpy.all(riley.pi_ub - riley.pi_lb > default.pi_ub - default.pi_lb)
    simple = metafopy.predict(res, newmods=NEWMODS, predtype='simple')
    assert simple.pi_dist == 'norm'
    both = metafopy.predict(res, newmods=NEWMODS, predtype='t', newvi=0.1)
    numpy.testing.assert_allclose(numpy.asarray(both.pi_se), numpy.sqrt(default.se**2 + res.tau2 + 0.1))
    spelling = metafopy.predict(res, newmods=NEWMODS, pi_type='riley')
    numpy.testing.assert_allclose(spelling.pi_lb, riley.pi_lb)
    override = metafopy.predict(res, newmods=NEWMODS, predtype='simple', pi_type='riley')
    numpy.testing.assert_allclose(override.pi_lb, simple.pi_lb)


def test_predtype_errors(dat: pandas.DataFrame) -> None:
    res = uni(dat, mods='~ablat')
    with pytest.raises(metafopy.errors.InputError, match='predtype'):
        metafopy.predict(res, predtype='bogus')
    with pytest.raises(metafopy.errors.InputError, match="Length of the 'newvi' argument"):
        metafopy.predict(res, newmods=NEWMODS, newvi=[0.1, 0.2])
    with pytest.raises(metafopy.errors.InputError, match="Length of the 'newvi' argument"):
        metafopy.predict(metafopy.rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg')), newvi=[0.1, 0.2])
    ls = metafopy.rma('yi', 'vi', mods='~ablat', scale='~ablat', data=dat)
    run = metafopy.predict(ls, newmods=NEWMODS, newscale=NEWMODS, predtype='riley', pi_type='simple')
    assert run.pi_dist == 't'


def test_forest_and_addpoly_pass_predtype(dat: pandas.DataFrame) -> None:
    """forest() and addpoly() hand predtype (and its old spelling pi_type) to predict(), as forest.rma and addpoly.rma do."""
    res = uni(dat)
    ref = REF['types']['uni_int'][2]
    assert ref['args'] == 'predtype'
    for spelling in ('predtype', 'pi_type'):
        fig, ax = matplotlib.pyplot.subplots()
        if spelling == 'predtype':
            metafopy.forest(res, addpred=True, ax=ax, predtype='riley')
        else:
            metafopy.forest(res, addpred=True, ax=ax, pi_type='riley')
        data = numpy.concatenate(
            [numpy.asarray(line.get_xdata(), dtype=float) for line in ax.lines] + [p.get_xy()[:, 0] for p in ax.patches if hasattr(p, 'get_xy')]
        )
        for bound in (float(ref['pi_lb']), float(ref['pi_ub'])):
            assert numpy.min(numpy.abs(data - bound)) < 1e-6, spelling
        matplotlib.pyplot.close(fig)
    fig, ax = matplotlib.pyplot.subplots()
    metafopy.forest(res, ax=ax)
    metafopy.addpoly(res, addpred=True, predtype='riley', ax=ax)
    data = numpy.concatenate(
        [numpy.asarray(line.get_xdata(), dtype=float) for line in ax.lines] + [p.get_xy()[:, 0] for p in ax.patches if hasattr(p, 'get_xy')]
    )
    assert numpy.min(numpy.abs(data - float(ref['pi_lb']))) < 1e-6
    matplotlib.pyplot.close(fig)


def test_parms(dat: pandas.DataFrame) -> None:
    ref = REF['parms']
    fits = models(dat)
    assert fits['uni_reml']()[0].parms == ref['uni']
    assert fits['mv_mods']()[0].parms == ref['mv']
    assert fits['mv_kon']()[0].parms == ref['mv_kon']
    assert fits['sel']()[0].parms == ref['sel']
    assert fits['rob']()[0].parms == ref['rob']


def berkey(kind: str) -> typing.Any:
    spec = REF['ber']
    frame = pandas.DataFrame({'yi': spec['yi'], 'outcome': spec['outcome'], 'trial': spec['trial']})
    V = numpy.asarray(spec['V'], dtype=float)
    return metafopy.rma_mv('yi', V, mods='~ 0 + outcome', random='~ outcome | trial', struct=kind, data=frame)


@pytest.mark.parametrize('kind', ['UN', 'DIAG'])
def test_predtype_several_tau2(kind: str) -> None:
    spec = REF['ber'][kind.lower()]
    res = berkey(kind)
    assert res.parms == spec['parms']
    for i, run in enumerate(spec['runs']):
        variants: list[dict[str, typing.Any]] = [{}, {'predtype': 'riley'}, {'predtype': 'simple'}, {'predtype': 't', 'newvi': 0.02}]
        kwargs = variants[i]
        check_prediction(metafopy.predict(res, newmods=numpy.eye(2), tau2_levels=[0, 1], **kwargs), run, f'{kind} run {i}', 1e-4)


def test_newmods_without_moderators(dat: pandas.DataFrame) -> None:
    """Models without moderators refuse new moderator values, with the message of metafor."""
    message = REF['int_only_errors']['uni']
    for key in ('mv', 'mh', 'peto', 'sel', 'rob', 'uni', 'mv_rob'):
        assert REF['int_only_errors'][key] == message
    cluster = dat['trial'].to_numpy()
    bcg = metafopy.datasets.load('dat.bcg')
    fits = {
        'mv': metafopy.rma_mv('yi', 'vi', random='~1|trial', data=dat),
        'mh': metafopy.rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg),
        'peto': metafopy.rma_peto(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg),
        'sel': metafopy.selmodel(uni(dat), type='halfnorm'),
        'rob': metafopy.robust(uni(dat), cluster),
        'uni': uni(dat),
        'mv_rob': metafopy.robust(metafopy.rma_mv('yi', 'vi', random='~1|trial', data=dat), cluster),
    }
    for res in fits.values():
        with pytest.raises(metafopy.errors.InputError, match=message):
            metafopy.predict(res, newmods=30)
    mods = metafopy.rma_mv('yi', 'vi', mods='~ablat', random='~1|trial', data=dat)
    close(metafopy.predict(mods, newmods=30).pred, REF['int_only_errors']['mv_ok'], 'mv with moderators', 1e-5)


def na_models(dna: pandas.DataFrame, bna: pandas.DataFrame) -> dict[str, collections.abc.Callable[[], tuple[typing.Any, typing.Any]]]:
    """The fits of the ``na`` fixture and the ``newmods`` they are predicted for."""
    new = [10, numpy.nan, 30]
    cluster = dna['trial'].to_numpy()
    mods = uni(dna, mods='~ablat')
    return {
        'uni_mods': lambda: (mods, new),
        'uni_int': lambda: (uni(dna), None),
        'uni_ee': lambda: (uni(dna, mods='~ablat', method='EE'), None),
        'mv_mods': lambda: (metafopy.rma_mv('yi', 'vi', mods='~ablat', random='~1|trial', data=dna), new),
        'mv_int': lambda: (metafopy.rma_mv('yi', 'vi', random='~1|trial', data=dna), None),
        'rob': lambda: (metafopy.robust(mods, cluster), new),
        'rob_cs': lambda: (metafopy.robust(mods, cluster, club_sandwich=True), new),
        'mh': lambda: (metafopy.rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bna), None),
        'peto': lambda: (metafopy.rma_peto(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bna), None),
        'sel': lambda: (metafopy.selmodel(mods, type='halfnorm'), None),
        'ls_loc': lambda: (metafopy.rma('yi', 'vi', mods='~ablat', scale='~1', data=dna), new),
        'ls_scale': lambda: (metafopy.rma('yi', 'vi', scale='~ablat', data=dna), None),
    }


def tolerance(key: str) -> float:
    return 1e-4 if key in ('sel', 'ls_loc', 'ls_scale') else 1e-5


@pytest.mark.parametrize('key', list(REF['na']))
@pytest.mark.parametrize('action', list(ACTIONS))
def test_na_action(dna: pandas.DataFrame, bna: pandas.DataFrame, key: str, action: str) -> None:
    spec = REF['na'][key][action]
    res, newmods = na_models(dna, bna)[key]()
    na_action = ACTIONS[action]
    rtol = tolerance(key)
    fitted = res.fitted(na_action=na_action)
    if isinstance(fitted, dict):
        for part in ('location', 'scale'):
            close(fitted[part], spec['fitted'][part], f'{key} fitted {part}', rtol)
    else:
        close(fitted, spec['fitted'], f'{key} fitted', rtol)
    if spec['hat'] is not None:
        close(res.hatvalues(na_action=na_action), spec['hat'], f'{key} hat', rtol)
        close(
            res.hatvalues('matrix', na_action=na_action),
            numpy.asarray([[numpy.nan if v is None else v for v in row] for row in spec['hat_matrix']]),
            f'{key} hat matrix',
            rtol,
        )
    if 'error' not in spec['predict']:
        check_prediction(metafopy.predict(res, na_action=na_action), spec['predict'], f'{key} predict', rtol, labels=key != 'sel')
        if spec['predict_vcov'] is not None:
            got = metafopy.predict(res, vcov=True, na_action=na_action).vcov
            close(got, numpy.asarray([[numpy.nan if v is None else v for v in row] for row in spec['predict_vcov']]), f'{key} vcov', rtol)
    if newmods is not None and 'error' not in spec['predict_newmods']:
        check_prediction(metafopy.predict(res, newmods=newmods, na_action=na_action), spec['predict_newmods'], f'{key} newmods', rtol)


def test_na_action_counts(dna: pandas.DataFrame) -> None:
    """bcg with yi[3] and ablat[5] missing: 13 values under exclude and pass, 11 under omit."""
    res = uni(dna, mods='~ablat')
    assert [len(res.fitted(na_action=a)) for a in ('omit', 'exclude', 'pass')] == [11, 13, 13]
    assert [len(metafopy.predict(res, na_action=a).pred) for a in ('omit', 'exclude', 'pass')] == [11, 13, 13]
    assert [len(res.hatvalues(na_action=a)) for a in ('omit', 'exclude', 'pass')] == [11, 13, 13]
    passed = res.fitted(na_action='pass')
    assert numpy.isnan(passed[4])
    assert not numpy.isnan(passed[2])
    assert numpy.isnan(res.fitted(na_action='exclude')[2])


def test_na_action_fail(dna: pandas.DataFrame) -> None:
    res = uni(dna, mods='~ablat')
    mv = metafopy.rma_mv('yi', 'vi', mods='~ablat', random='~1|trial', data=dna)
    ls = metafopy.rma('yi', 'vi', scale='~ablat', data=dna)
    for fit in (res, mv, ls):
        with pytest.raises(metafopy.errors.InputError, match='Missing values in results'):
            fit.fitted(na_action='fail')
        with pytest.raises(metafopy.errors.InputError, match='Missing values in results'):
            metafopy.predict(fit, na_action='fail')
        with pytest.raises(metafopy.errors.InputError, match='Missing values in results'):
            fit.hatvalues(na_action='fail')
        with pytest.raises(metafopy.errors.InputError, match='na_action'):
            fit.fitted(na_action='drop')  # type: ignore[arg-type]
    complete = uni(dna.dropna())
    assert len(complete.fitted(na_action='fail')) == 11


@pytest.mark.parametrize('key', list(REF['ls_na']))
@pytest.mark.parametrize('action', list(ACTIONS))
def test_location_scale_missing_scale_value(dna: pandas.DataFrame, key: str, action: str) -> None:
    """Studies left out because of a missing scale value are rows of the subset: R returns 13 values, with the residual of study 5 under pass."""
    spec = REF['ls_na'][key][action]
    fit = {
        'scale_only': lambda: metafopy.rma('yi', 'vi', scale='~ablat', data=dna),
        'both': lambda: metafopy.rma('yi', 'vi', mods='~ablat', scale='~ablat', data=dna),
        'loc_only': lambda: metafopy.rma('yi', 'vi', mods='~ablat', scale='~1', data=dna),
    }[key]()
    na_action = ACTIONS[action]
    rtol = 1e-4
    close(fit.residuals(na_action=na_action), spec['residuals'], f'{key} residuals', rtol)
    close(fit.weights_(na_action=na_action), spec['weights'], f'{key} weights', rtol)
    close(fit.residuals('pearson', na_action=na_action), spec['pearson'], f'{key} pearson', rtol)
    close(fit.residuals('cholesky', na_action=na_action), spec['cholesky'], f'{key} cholesky', rtol)
    std = metafopy.rstandard(fit, na_action=na_action)
    close(std.z, spec['rstandard'], f'{key} rstandard', rtol)
    close(std.resid, spec['rstandard_resid'], f'{key} rstandard resid', rtol)
    assert list(std.slab) == spec['rstandard_slab']
    close(metafopy.rstandard(fit, type='conditional', na_action=na_action).z, spec['rstandard_cond'], f'{key} conditional', rtol)
    close(metafopy.methods.blup.blup(fit, na_action=na_action).pred, spec['blup'], f'{key} blup', rtol)
    if key == 'scale_only' and action == 'na.pass':
        assert len(fit.residuals(na_action='pass')) == 13
        assert numpy.isnan(fit.residuals(na_action='pass')[2])
        assert not numpy.isnan(fit.residuals(na_action='pass')[4])


def test_glmm_predict(bna: pandas.DataFrame) -> None:
    """rma_glmm fits (no R oracle here: lme4 is not installed) follow the same formulas and refuse newmods without moderators."""
    res = metafopy.rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', measure='OR', data=bna, model='UM.FS')
    with pytest.raises(metafopy.errors.InputError, match='without moderators'):
        metafopy.predict(res, newmods=30)
    default = metafopy.predict(res)
    riley = metafopy.predict(res, predtype='riley', newvi=0.05)
    assert riley.pi_dist == 't'
    numpy.testing.assert_allclose(numpy.asarray(riley.pi_ddf), max(res.k - res.parms, 1))
    numpy.testing.assert_allclose(numpy.asarray(riley.pi_se), numpy.sqrt(default.se**2 + res.tau2 + 0.05))
    assert len(res.fitted(na_action='omit')) == 11
    assert len(res.fitted(na_action='pass')) == 13
    assert numpy.isnan(res.fitted(na_action='exclude')[2])


def test_na_action_club_sandwich(dna: pandas.DataFrame) -> None:
    """R cannot predict from a clubSandwich fit made from data with missing values; metafopy returns one row per study with NaN where unknown."""
    mods = uni(dna, mods='~ablat')
    res = metafopy.robust(mods, dna['trial'].to_numpy(), club_sandwich=True)
    assert [len(metafopy.predict(res, na_action=a).pred) for a in ('omit', 'exclude', 'pass')] == [11, 13, 13]
    passed = metafopy.predict(res, na_action='pass')
    assert numpy.isnan(passed.pred[4])
    assert numpy.isnan(passed.ci_lb[4])
    assert not numpy.isnan(passed.pred[2])
