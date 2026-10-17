"""na_action= on the functions whose R counterpart reads options(na.action=) (regtest, ranktest, fsn, tes, hettest, model.matrix, simulate, funnel, qqnorm)."""

import numpy
import pytest

import metafopy
import metafopy.errors

YI = [0.1, 0.4, numpy.nan, 0.2, -0.3, 0.5, 0.8, 0.0]
VI = [0.05, 0.04, 0.03, 0.06, 0.05, 0.02, 0.07, 0.04]


def test_regtest_fail_stops_and_omit_matches_default() -> None:
    with pytest.raises(metafopy.errors.InputError):
        metafopy.regtest(YI, VI, na_action='fail')
    omitted = metafopy.regtest(YI, VI, na_action='omit')
    default = metafopy.regtest(YI, VI)
    assert omitted.zval == default.zval
    assert metafopy.regtest(YI, VI, na_action='exclude').zval == default.zval


def test_ranktest_and_fsn_fail_stop() -> None:
    with pytest.raises(metafopy.errors.InputError):
        metafopy.ranktest(YI, VI, na_action='fail')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.fsn(YI, VI, na_action='fail')


def test_regtest_rejects_unknown_na_action() -> None:
    with pytest.raises(metafopy.errors.InputError):
        metafopy.regtest(YI, VI, na_action='bogus')  # type: ignore[arg-type]


def test_model_matrix_na_action() -> None:
    fit = metafopy.rma(YI, VI, mods=[1, 2, 3, 4, numpy.nan, 6, 7, 8], method='REML')
    omit = metafopy.model_matrix(fit)
    assert len(omit) == fit.k
    with pytest.raises(metafopy.errors.InputError):
        metafopy.model_matrix(fit, na_action='fail')
    padded = metafopy.model_matrix(fit, na_action='exclude')
    assert len(padded) == len(YI)
