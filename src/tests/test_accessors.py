"""se, nobs, df_residual, formula, model_matrix, update and the rma_glmm fitted values/residuals against R
(src/tests/r/generate_accessors.R)."""

import collections.abc
import dataclasses
import functools
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors
import metafopy.models.call

REF = refutil.load_phase7('accessors')
# rma.mv, rma.glmm, location-scale and selection-model estimates come from an optimiser; they agree with R to its tolerance
OPTIM = 1e-4


def raw() -> pandas.DataFrame:
    return metafopy.datasets.load('bcg')


CELLS: dict[str, typing.Any] = dict(ai='tpos', bi='tneg', ci='cpos', di='cneg')


def quiet[T](fn: collections.abc.Callable[[], T]) -> T:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn()


@functools.cache
def fits() -> dict[str, typing.Any]:
    d = refutil.bcg_na_yi()
    return {
        'uni': quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', data=d)),
        'uni_yi': quiet(lambda: metafopy.rma_uni('yi ~ ablat', 'vi', data=d)),
        'uni_subset': quiet(lambda: metafopy.rma_uni('yi', 'vi', data=d, subset=list(range(10)))),
        'sel': quiet(lambda: metafopy.selmodel(metafopy.rma_uni('yi', 'vi', data=d), type='beta')),
        'glmm': metafopy.rma_glmm(measure='OR', **CELLS, data=raw(), mods='~ ablat'),
        'mh': metafopy.rma_mh(measure='OR', **CELLS, data=raw(), subset=list(range(5))),
        'peto': metafopy.rma_peto(**CELLS, data=raw()),
        'ls': quiet(lambda: metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ year', data=d)),
        'mv': metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=metafopy.datasets.load('konstantopoulos2011')),
    }


def nospace(text: str | None) -> str | None:
    return None if text is None else text.replace(' ', '')


@pytest.mark.parametrize('key', ['uni', 'uni_yi', 'uni_subset', 'sel', 'glmm', 'mh', 'peto', 'ls', 'mv'])
def test_accessors(key: str) -> None:
    res = fits()[key]
    exp = REF[key]
    rtol = refutil.RTOL if key in ('uni', 'uni_yi', 'uni_subset', 'mh', 'peto') else OPTIM
    assert res.nobs() == exp['nobs']
    assert metafopy.nobs(res) == exp['nobs']
    got_all = res.nobs(all=True)
    assert got_all.to_series().to_dict() == exp['nobs_all']
    assert dataclasses.asdict(got_all)['df_residual'] == exp['df_residual']
    assert res.df_residual() == exp['df_residual'] == metafopy.df_residual(res)
    assert nospace(res.formula()) == exp['formula']
    se = metafopy.se(res)
    if isinstance(exp['se'], dict):
        for part, values in exp['se'].items():
            refutil.close(se[part], values, f'{key}.se.{part}', rtol=rtol)
            refutil.close(metafopy.se(res, type=part), values, f'{key}.se.{part}', rtol=rtol)
        mm = res.model_matrix() if key == 'ls' else {'X': res.model_matrix()}
        if key == 'ls':
            numpy.testing.assert_allclose(mm['location'], numpy.array(exp['model_matrix']['location'], dtype=float))
            numpy.testing.assert_allclose(mm['scale'], numpy.array(exp['model_matrix']['scale'], dtype=float))
    else:
        refutil.close(se, exp['se'], f'{key}.se', rtol=rtol)
        numpy.testing.assert_allclose(metafopy.model_matrix(res), numpy.array(exp['model_matrix'], dtype=float).reshape(res.k, -1))
        assert list(metafopy.model_matrix(res, asdf=True).columns) == list(res.coef_names)


def test_formula_types() -> None:
    assert nospace(fits()['uni_yi'].formula(type='yi')) == nospace(REF['uni_yi']['formula_yi'])
    assert nospace(fits()['ls'].formula(type='scale')) == nospace(REF['ls']['formula_scale'])
    with pytest.raises(metafopy.errors.InputError, match='scale'):
        fits()['uni'].formula(type='scale')


def test_glmm_fitted_residuals() -> None:
    res = fits()['glmm']
    refutil.close(res.fitted(), REF['glmm']['fitted'], 'fitted', rtol=OPTIM)
    refutil.close(res.residuals(), REF['glmm']['residuals'], 'residuals', rtol=OPTIM, atol=1e-5)


def test_update_rma_uni() -> None:
    res = fits()['uni']
    refutil.close(quiet(lambda: res.update('~ . + year')).beta, REF['update_formula'], 'update_formula')
    refutil.close(quiet(lambda: res.update('~ . + year').update('~ . - ablat')).beta, REF['update_drop'], 'update_drop')
    ml = quiet(lambda: metafopy.update(res, method='ML'))
    refutil.close([*ml.beta, ml.tau2], REF['update_ml'], 'update_ml', rtol=1e-6)
    refutil.close(quiet(lambda: fits()['uni_yi'].update('. ~ . + year')).beta, REF['update_yi'], 'update_yi')
    call = res.update(method='ML', evaluate=False)
    assert isinstance(call, metafopy.models.call.Call)
    assert call.arguments['method'] == 'ML'
    assert call.name == 'rma_uni'


def test_update_other_fits() -> None:
    f = fits()
    mv = f['mv'].update(mods='~ I(year - 1990)')
    refutil.close([*mv.beta, *mv.sigma2], REF['update_mv'], 'update_mv', rtol=OPTIM)
    refutil.close(f['mh'].update(measure='RR').beta, REF['update_mh'], 'update_mh')
    refutil.close(f['peto'].update(subset=list(range(6))).beta, REF['update_peto'], 'update_peto')
    refutil.close(f['glmm'].update('~ . + year').beta, REF['update_glmm'], 'update_glmm', rtol=OPTIM)
    refutil.close(quiet(lambda: f['ls'].update(scale='~ ablat')).alpha, REF['update_ls'], 'update_ls', rtol=OPTIM)
    sel = quiet(lambda: f['sel'].update(type='halfnorm'))
    refutil.close([*sel.beta, *sel.delta], REF['update_sel'], 'update_sel', rtol=OPTIM)


def test_update_errors() -> None:
    res = fits()['uni']
    with pytest.raises(metafopy.errors.InputError, match='call'):
        metafopy.update(dataclasses.replace(res, call=None))
    with pytest.raises(metafopy.errors.InputError, match='formula'):
        # auditme: accept[literal-duplication] the moderator is named as in the R script
        metafopy.rma_uni('yi', 'vi', mods=refutil.bcg_na_yi()[['ablat']].fillna(0).to_numpy(), data=refutil.bcg_na_yi().fillna(0)).update('~ . + year')
    with pytest.raises(metafopy.errors.ArgumentError):
        metafopy.nobs(object())  # type: ignore[arg-type]


def test_update_formula_rules() -> None:
    uf = metafopy.models.call.update_formula
    assert uf('~ ablat', '~ . + year') == '~ ablat + year'
    assert uf(None, '~ . + year') == '~ year'
    assert uf('~ ablat + year', '~ . - year') == '~ ablat'
    assert uf('yi ~ ablat', '. ~ . + x') == 'yi ~ ablat + x'
    assert uf('~ a', '~ . - 1') == '~ a - 1'
    assert uf('~ I(year - 1990) + a:b', '~ . - a:b') == '~ I(year - 1990)'
    with pytest.raises(metafopy.errors.InputError):
        uf('~ a', 'x')
