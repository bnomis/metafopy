"""residuals(type=), weights_(type=) and the R spellings object2 and clubSandwich against R metafor 5.2-1
(fixtures from src/tests/r/generate_residuals_weights.R)."""

import collections.abc
import functools
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors

REF = refutil.load_phase9('residuals_weights')
NA_ACTIONS = {'na.omit': 'omit', 'na.exclude': 'exclude', 'na.pass': 'pass', 'na.fail': 'fail'}
# rma.mv, rma.glmm, location-scale and selection-model estimates come from an optimiser; they agree with R to its tolerance
OPTIM = 1e-4
TOLERANCE = {'ls': OPTIM, 'sel': OPTIM, 'glmm': OPTIM, 'mv_konst': OPTIM, 'mv_na_mods': OPTIM, 'mv_na_int': OPTIM}
CELLS: dict[str, typing.Any] = dict(ai='tpos', bi='tneg', ci='cpos', di='cneg')
# what the error text of R means: the exception class and the words that must be in metafopy's message
ERRORS = [
    ('Missing values in', metafopy.errors.InputError, 'Missing values in'),
    ('Row-sum weights', metafopy.errors.InputError, 'Row-sum weights'),
    ('no applicable method', metafopy.errors.ArgumentError, 'no applicable method'),
    ('Method not available', metafopy.errors.NotSupportedError, 'not available for objects of class'),
    ('Extraction of', metafopy.errors.NotSupportedError, 'not available for objects of class'),
]


def quiet[T](fn: collections.abc.Callable[[], T]) -> T:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn()


def data(vi_na: bool = False, weights: bool = False) -> pandas.DataFrame:
    d = refutil.bcg_na()
    if vi_na:
        d.loc[4, 'vi'] = numpy.nan
    if weights:
        d['n'] = numpy.arange(1.0, d.shape[0] + 1)
    return d


def raw_na() -> pandas.DataFrame:
    d = metafopy.datasets.load('bcg')
    d.loc[2, 'tpos'] = numpy.nan
    return d


@functools.cache
def fits() -> dict[str, typing.Any]:
    return quiet(build)


def build() -> dict[str, typing.Any]:
    d = data()
    complete = refutil.bcg()
    uni = quiet(lambda: metafopy.rma_uni('yi', 'vi', data=d))
    return {
        'uni_mods': quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=d)),
        'uni_nomods_dl': quiet(lambda: metafopy.rma_uni('yi', 'vi', data=d, method='DL')),
        'uni_fe_mods': quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat + year', data=d, method='FE')),
        'uni_ml_subset': quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=d, method='ML', subset=list(range(10)))),
        'uni_vi_na': quiet(lambda: metafopy.rma_uni('yi', 'vi', data=data(vi_na=True))),
        'uni_weights': quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=data(weights=True), weights='n')),
        'uni_unweighted': quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=d, weighted=False)),
        'uni_knha': quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=d, test='knha')),
        'uni_robust': quiet(lambda: metafopy.robust(uni, cluster=d['trial'].to_numpy())),
        'uni_complete': quiet(lambda: metafopy.rma_uni('yi', 'vi', data=complete)),
        'ls': quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ year', data=d)),
        'sel': quiet(lambda: metafopy.selmodel(uni, type='beta')),
        'mh': metafopy.rma_mh(measure='OR', **CELLS, data=raw_na(), subset=list(range(11))),
        'peto': metafopy.rma_peto(**CELLS, data=raw_na()),
        'mh_complete': metafopy.rma_mh(measure='RR', **CELLS, data=metafopy.datasets.load('bcg')),
        'glmm': metafopy.rma_glmm(measure='OR', **CELLS, data=raw_na(), mods='~ ablat'),
        'mv_konst': metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=metafopy.datasets.load('konstantopoulos2011')),
        'mv_na_mods': quiet(lambda: metafopy.rma_mv('yi', 'vi', mods='~ ablat', random='~ 1 | trial', data=d)),
        'mv_na_int': quiet(lambda: metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=d)),
    }


def check(actual: collections.abc.Callable[[], typing.Any], expected: dict[str, typing.Any], key: str, what: str) -> None:
    if 'error' in expected:
        message = expected['error']
        if 'attempt to set an attribute on NULL' in message:
            # rstudent.rma.uni fails under na.fail when no study is missing (a quirk of metafor)
            return
        for r_text, exc, words in ERRORS:
            if r_text in message:
                with pytest.raises(exc, match=words):
                    quiet(actual)
                return
        pytest.fail(f'{what}: unclassified R error {message!r}')
    if expected['value'] is None:
        # R returns NULL for rstandard() and rstudent() under na.fail when no study is missing (a quirk of rstandard.rma.mh and relatives)
        return
    tol = TOLERANCE.get(key, refutil.RTOL)
    refutil.close(quiet(actual), expected['value'], what, rtol=tol, atol=1e-5 if tol > refutil.RTOL else 1e-9)


CASES = [(key, ty, na) for key, spec in REF.items() if isinstance(spec, dict) and 'residuals' in spec for ty in spec['residuals'] for na in NA_ACTIONS]
WEIGHT_CASES = [(key, ty, na) for key, spec in REF.items() if isinstance(spec, dict) and 'weights' in spec for ty in spec['weights'] for na in NA_ACTIONS]


@pytest.mark.parametrize(('key', 'ty', 'na'), CASES)
def test_residuals(key: str, ty: str, na: str) -> None:
    res = fits()[key]
    check(lambda: res.residuals(ty, na_action=NA_ACTIONS[na]), REF[key]['residuals'][ty][na], key, f'{key} residuals {ty} {na}')


@pytest.mark.parametrize(('key', 'ty', 'na'), WEIGHT_CASES)
def test_weights(key: str, ty: str, na: str) -> None:
    res = fits()[key]
    check(lambda: res.weights_(ty, na_action=NA_ACTIONS[na]), REF[key]['weights'][ty][na], key, f'{key} weights {ty} {na}')


def test_function_matches_method() -> None:
    res = fits()['uni_mods']
    numpy.testing.assert_array_equal(metafopy.residuals(res, 'pearson', na_action='exclude'), res.residuals('pearson', na_action='exclude'))


def test_default_is_response_of_used_studies() -> None:
    res = fits()['uni_mods']
    refutil.close(res.residuals(), REF['uni_mods']['residuals']['response']['na.omit']['value'], 'default residuals')


def test_partial_type() -> None:
    res = fits()['uni_mods']
    refutil.close(res.residuals('resp'), REF['partial']['resp'], 'resp')
    refutil.close(res.residuals('pea'), REF['partial']['pear'], 'pea')
    refutil.close(res.residuals('rstan'), REF['partial']['rstan'], 'rstan')
    assert 'should be one of' in REF['partial_error']
    with pytest.raises(metafopy.errors.InputError, match='should be one of'):
        res.residuals('r')
    with pytest.raises(metafopy.errors.InputError, match='should be one of'):
        res.residuals('nonsense')


def test_invalid_na_action() -> None:
    with pytest.raises(metafopy.errors.InputError, match=r'na\.action'):
        fits()['uni_mods'].residuals(na_action='nope')
    with pytest.raises(metafopy.errors.InputError, match=r'na\.action'):
        fits()['uni_mods'].weights_(na_action='nope')


def test_not_a_fit() -> None:
    with pytest.raises(metafopy.errors.ArgumentError):
        metafopy.residuals(object())


def test_object2_alias() -> None:
    d = refutil.bcg()
    full = quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=d, method='ML'))
    reduced = quiet(lambda: metafopy.rma_uni('yi', 'vi', data=d, method='ML'))
    by_name = metafopy.anova(full, model2=reduced)
    assert metafopy.anova(full, object2=reduced).LRT == by_name.LRT
    assert metafopy.anova(full, reduced, object2=reduced).LRT == by_name.LRT
    with pytest.raises(metafopy.errors.InputError, match='object2'):
        metafopy.anova(full, reduced, object2=full)


def test_object2_alias_mv_ls_selmodel() -> None:
    f = fits()
    mv_full = metafopy.rma_mv('yi', 'vi', mods='~ ablat', random='~ 1 | trial', data=refutil.bcg(), method='ML')
    mv_red = metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=refutil.bcg(), method='ML')
    assert metafopy.anova(mv_full, object2=mv_red).LRT == metafopy.anova(mv_full, mv_red).LRT
    ls_red = quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ 1', data=data(), method='ML'))
    ls_full = quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ year', data=data(), method='ML'))
    assert metafopy.anova(ls_full, object2=ls_red).LRT == metafopy.anova(ls_full, ls_red).LRT
    sel = f['sel']
    assert metafopy.anova(sel, object2=f['uni_nomods_dl']).LRT == metafopy.anova(sel, f['uni_nomods_dl']).LRT


def test_club_sandwich_alias() -> None:
    d = refutil.bcg()
    res = quiet(lambda: metafopy.rma_uni('yi', 'vi', data=d))
    spelled = quiet(lambda: metafopy.robust(res, 'trial', clubSandwich=True))
    snake = quiet(lambda: metafopy.robust(res, 'trial', club_sandwich=True))
    assert spelled.robust is not None
    assert spelled.robust.robumethod == 'clubSandwich'
    numpy.testing.assert_array_equal(spelled.se, snake.se)
    both = quiet(lambda: metafopy.robust(res, 'trial', club_sandwich=True, clubSandwich=True))
    numpy.testing.assert_array_equal(both.se, snake.se)
    with pytest.raises(metafopy.errors.InputError, match='clubSandwich'):
        metafopy.robust(res, 'trial', club_sandwich=False, clubSandwich=True)
    default = quiet(lambda: metafopy.robust(res, 'trial'))
    assert default.robust is not None
    assert default.robust.robumethod == 'default'
