"""Cluster-robust fits: predict(), forest(), regplot() and addpoly() with clubSandwich, the functions robust() fits refuse and simulate(); checked
against R metafor 5.2-1 and clubSandwich 0.7.0 (tests/r/generate_audit3.R)."""

import collections.abc
import typing

import matplotlib
import matplotlib.pyplot
import numpy
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors

matplotlib.use('Agg')

REF: dict[str, typing.Any] = refutil.load_phase12('audit3')
KON = metafopy.datasets.load('konstantopoulos2011')
MODS = '~ I(year - 1990)'
NEWMODS = [-5, 0, 5]
# auditme: accept[literal-duplication] the fixture names repeat as the R script that generates them repeats them
FIELDS = ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub')  # auditme: accept[non-alphabetical-literal] the order of R's predict()
# auditme: accept[non-alphabetical-literal] the order follows the R fixture generator
COEF_TESTS = ('Satterthwaite', 'z', 'naive-t', 'naive-tp')


@pytest.fixture(scope='module')
def uni1() -> typing.Any:
    return metafopy.rma('yi', 'vi', mods=MODS, data=KON)


@pytest.fixture(scope='module')
def mv1() -> typing.Any:
    return metafopy.rma_mv('yi', 'vi', mods=MODS, random='~ 1 | district/school', data=KON)


@pytest.fixture(scope='module')
def uni0() -> typing.Any:
    return metafopy.rma('yi', 'vi', data=KON)


def check_prediction(pr: typing.Any, ref: dict[str, typing.Any], what: str) -> None:
    for field in FIELDS:
        refutil.close(getattr(pr, field), ref[field], f'{what} {field}', rtol=1e-6)
    refutil.close(numpy.where(numpy.isinf(pr.pi_ddf), numpy.nan, pr.pi_ddf), ref['pi_ddf'], f'{what} pi_ddf', rtol=1e-6)


@pytest.mark.parametrize('coef_test', COEF_TESTS)
def test_predict_uni(uni1: typing.Any, coef_test: str) -> None:
    ref = REF['predict_cs'][f'uni_{coef_test}']
    rb = metafopy.robust(uni1, 'district', club_sandwich=True, coef_test=coef_test)
    check_prediction(rb.predict(newmods=NEWMODS), ref, coef_test)
    check_prediction(rb.predict(), ref['fit'], f'{coef_test} fitted')
    check_prediction(rb.predict(newmods=NEWMODS, level=90), ref['level90'], f'{coef_test} level')
    check_prediction(rb.predict(newmods=NEWMODS, adjust=True), ref['adjust'], f'{coef_test} adjust')
    check_prediction(rb.predict(newmods=NEWMODS, transf=numpy.exp), ref['transf'], f'{coef_test} transf')
    refutil.close(rb.predict(newmods=NEWMODS, vcov=True).vcov.reshape(-1), numpy.asarray(ref['vcov']).reshape(-1).tolist(), f'{coef_test} vcov', rtol=1e-6)


@pytest.mark.parametrize('coef_test', COEF_TESTS)
def test_predict_mv(mv1: typing.Any, coef_test: str) -> None:
    ref = REF['predict_cs'][f'mv_{coef_test}']
    rb = metafopy.robust(mv1, 'district', club_sandwich=True, coef_test=coef_test)
    check_prediction(rb.predict(newmods=NEWMODS), ref, coef_test)
    check_prediction(rb.predict(), ref['fit'], f'{coef_test} fitted')


def test_predict_intercept_only_and_cr1(uni0: typing.Any, uni1: typing.Any) -> None:
    ref = REF['predict_cs']
    check_prediction(metafopy.robust(uni0, 'district', club_sandwich=True).predict(), ref['uni_int'], 'intercept')
    check_prediction(metafopy.robust(uni1, 'district', club_sandwich=True, vcov='CR1').predict(newmods=NEWMODS), ref['uni_cr1'], 'CR1')


def test_predict_saddlepoint_refused(uni1: typing.Any) -> None:
    rb = metafopy.robust(uni1, 'district', club_sandwich=True, coef_test='saddlepoint')
    with pytest.raises(metafopy.errors.InputError, match='saddlepoint'):
        rb.predict(newmods=[0])


def test_plots_accept_clubsandwich(uni1: typing.Any, uni0: typing.Any) -> None:
    rb, rb0 = metafopy.robust(uni1, 'district', club_sandwich=True), metafopy.robust(uni0, 'district', club_sandwich=True)
    fig, ax = matplotlib.pyplot.subplots()
    metafopy.forest(rb, ax=ax)
    metafopy.regplot(rb, mod=1, ax=ax)
    metafopy.forest(rb0, ax=ax)
    metafopy.addpoly(rb0, ax=ax)
    matplotlib.pyplot.close(fig)


def robust_fits(uni1: typing.Any, uni0: typing.Any, mv1: typing.Any) -> dict[str, typing.Any]:
    return {
        'rb_u': metafopy.robust(uni1, 'district'),
        'rc_u': metafopy.robust(uni1, 'district', club_sandwich=True),
        'rb_u0': metafopy.robust(uni0, 'district'),
        'rb_m': metafopy.robust(mv1, 'district'),
    }


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
REFUSING: dict[str, collections.abc.Callable[[typing.Any], typing.Any]] = {
    'rstandard': metafopy.rstandard,
    'rstudent': metafopy.rstudent,
    'dfbetas': metafopy.dfbetas,
    'influence': metafopy.influence,
    'leave1out': metafopy.leave1out,
    'cumul': metafopy.cumul,
    'permutest': lambda x: metafopy.permutest(x, iter=20),
    'funnel': metafopy.funnel,
    'qqnorm': metafopy.qqnorm,
    'plot_permutest': lambda x: metafopy.plot_permutest(typing.cast(typing.Any, metafopy.permutest(x, iter=20))),
    'cooks': metafopy.cooks_distance,
    'hatvalues': lambda x: x.hatvalues(),
    'simulate': lambda x: metafopy.simulate(x, nsim=1, seed=1),
}


@pytest.mark.parametrize('model', ['rb_u', 'rc_u', 'rb_u0', 'rb_m'])
@pytest.mark.parametrize('function', sorted(REFUSING))
def test_refusals(uni1: typing.Any, uni0: typing.Any, mv1: typing.Any, model: str, function: str) -> None:
    x = robust_fits(uni1, uni0, mv1)[model]
    expected = REF['refuse'][model].get(function, REF['refuse'][model].get('plot_' + function))
    try:
        REFUSING[function](x)
        outcome = 'ok'
    except metafopy.errors.MetaforError as e:
        outcome = str(e)
    finally:
        matplotlib.pyplot.close('all')
    if expected == 'ok':
        assert outcome == 'ok'
    else:
        # rma.mv leave1out()/influence()/permutest() have no method in R at all, other fits refuse with metafor's message
        assert outcome != 'ok'
        if 'robust.rma' in expected and model != 'rb_m':
            assert outcome == expected


@pytest.mark.parametrize(
    ('key', 'model', 'kw'),
    [
        ('uni', 'rb_u', {'nsim': 3, 'seed': 42}),
        ('uni_cs', 'rc_u', {'nsim': 2, 'seed': 7}),
        ('mv', 'rb_m', {'nsim': 2, 'seed': 5}),
        ('uni_withvb', 'rb_u', {'nsim': 2, 'seed': 9, 'withvb': True}),
    ],
)
def test_simulate_robust(uni1: typing.Any, uni0: typing.Any, mv1: typing.Any, key: str, model: str, kw: dict[str, typing.Any]) -> None:
    out = metafopy.simulate(robust_fits(uni1, uni0, mv1)[model], **kw)
    refutil.close(out.to_numpy().reshape(-1), numpy.asarray(REF['simulate_robust'][key]).reshape(-1).tolist(), key, rtol=1e-7)
