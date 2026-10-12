"""rma_uni features beyond the per-dataset grid, checked against R metafor."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from metafopy import LRTResult, anova, confint, datasets, predict, rma
from metafopy.models.rma_uni import RMAResult

REF: dict[str, Any] = json.loads(
    (Path(__file__).parent / "reference" / "extras" / "rma_uni.json").read_text()
)
RTOL = 1e-7


def _close(actual: Any, expected: Any, what: str = "") -> None:
    np.testing.assert_allclose(
        np.atleast_1d(np.asarray(actual, dtype=float)),
        np.atleast_1d(np.asarray(expected, dtype=float)),
        rtol=RTOL,
        atol=1e-10,
        equal_nan=True,
        err_msg=what,
    )


@pytest.fixture(scope="module")
def dat() -> pd.DataFrame:
    d = datasets.load("bcg")
    a, b, c, e = (d[col].to_numpy(float) for col in ("tpos", "tneg", "cpos", "cneg"))
    d["yi"] = np.log(a / (a + b)) - np.log(c / (c + e))
    d["vi"] = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + e)
    d["n"] = a + b + c + e
    return d


def _check(res: RMAResult, key: str) -> None:
    ref = REF[key]
    for attr in ("beta", "se", "pval", "ci_lb", "ci_ub", "tau2", "QE", "QM", "QMp", "I2", "H2"):
        _close(getattr(res, attr), ref[attr], f"{key}: {attr}")
    if ref["se_tau2"] is not None:
        _close(res.se_tau2, ref["se_tau2"], f"{key}: se_tau2")
    if ref["R2"] is not None:
        _close(res.R2, ref["R2"], f"{key}: R2")
    _close(res.fit_stats_ml.ll, ref["ll_ml"], f"{key}: ll ML")
    _close(res.fit_stats_reml.ll, ref["ll_reml"], f"{key}: ll REML")
    _close(res.weights_(), ref["weights"], f"{key}: weights")
    _close(res.hatvalues(), ref["hat"], f"{key}: hat")


CASES: dict[str, dict[str, Any]] = {
    "weights_REML": {"weights": "n"},
    "weights_knha": {"weights": "n", "test": "knha"},
    "genq": {"weights": "n", "method": "GENQ"},
    "genq_mods": {"weights": "n", "mods": "~ ablat", "method": "GENQ"},
    "tau2_fixed": {"tau2": 0.2},
    "tau2_fixed_mods": {"tau2": 0.2, "mods": "~ ablat"},
    "unweighted": {"weighted": False},
    "unweighted_knha": {"weighted": False, "mods": "~ ablat", "test": "knha"},
    "test_t": {"mods": "~ ablat", "test": "t"},
    "test_adhoc": {"mods": "~ ablat", "test": "adhoc"},
    "btt": {"mods": "~ ablat + alloc", "btt": [2, 3]},
    "dlit": {"method": "DLIT"},
    "sjit": {"method": "SJIT"},
    "no_intercept": {"mods": "~ 0 + alloc"},
    "level90": {"level": 90},
    "subset": {"subset": "alloc == 'random'"},
}


@pytest.mark.parametrize("key", list(CASES))
def test_feature_matches_metafor(dat: pd.DataFrame, key: str) -> None:
    _check(rma(data=dat, **CASES[key]), key)


def test_anova_lrt(dat: pd.DataFrame) -> None:
    full = rma(data=dat, mods="~ ablat + alloc", method="ML")
    red = rma(data=dat, mods="~ ablat", method="ML")
    out = anova(full, red)
    assert isinstance(out, LRTResult)
    ref = REF["anova_lrt"]
    _close(out.LRT, ref["LRT"])
    _close(out.pval, ref["pval"])
    _close(out.R2, ref["R2"])
    assert (out.df_full, out.df_reduced) == (ref["parms_f"], ref["parms_r"])
    # argument order does not matter
    swapped = anova(red, full)
    assert isinstance(swapped, LRTResult)
    assert swapped.LRT == out.LRT


def test_anova_linear_combinations(dat: pd.DataFrame) -> None:
    full = rma(data=dat, mods="~ ablat + alloc", method="ML")
    out = anova(full, X=[[0, 1, 0, 0], [0, 0, 1, -1]])
    ref = REF["anova_X"]
    for attr in ("Xb", "se", "zval", "pval", "QM", "QMp"):
        _close(getattr(out, attr), ref[attr], attr)


def test_anova_btt_knha(dat: pd.DataFrame) -> None:
    res = rma(data=dat, mods="~ ablat + alloc", test="knha")
    out = res.anova(btt=[2, 3])
    _close(out.QM, REF["anova_btt_knha"]["QM"])
    _close(out.QMp, REF["anova_btt_knha"]["QMp"])
    assert "coefficients 3:4" in str(out)


def test_predict_newmods(dat: pd.DataFrame) -> None:
    res = rma(data=dat, mods="~ ablat")
    pr = predict(res, [10, 30, 50])
    for attr in ("pred", "se", "ci_lb", "ci_ub", "pi_lb", "pi_ub"):
        _close(getattr(pr, attr), REF["predict_newmods"][attr], attr)
    fitted = predict(res)
    _close(fitted.pred, REF["predict_fitted"]["pred"])
    _close(fitted.pi_lb, REF["predict_fitted"]["pi_lb"])
    assert fitted.slab == res.slab


def test_predict_transf_drops_se(dat: pd.DataFrame) -> None:
    pr = predict(rma(data=dat), transf=np.exp)
    assert np.isnan(pr.se).all()
    assert pr.ci_lb[0] < pr.pred[0] < pr.ci_ub[0]


def test_confint_fixed_knha(dat: pd.DataFrame) -> None:
    ci = confint(rma(data=dat, mods="~ ablat", test="knha"), fixed=True, random=False).fixed
    assert ci is not None
    ref = REF["confint_fixed_knha"]
    _close(ci["estimate"], ref["estimate"])
    _close(ci["ci.lb"], ref["ci_lb"])
    _close(ci["ci.ub"], ref["ci_ub"])


def test_method_fallback(dat: pd.DataFrame) -> None:
    res = rma(data=dat, method=["GENQ", "REML"])  # GENQ needs weights, so falls back
    assert res.method == "REML"


def test_control_tau2_init_and_names(dat: pd.DataFrame) -> None:
    a = rma(data=dat, control={"tau2.init": 1.0})
    b = rma(data=dat, control={"tau2_init": 1.0})
    assert a.tau2 == pytest.approx(b.tau2)
    with pytest.raises(TypeError, match="unknown control"):
        rma(data=dat, control={"nope": 1})


def test_redundant_predictor_dropped(dat: pd.DataFrame) -> None:
    d = dat.assign(ablat2=dat["ablat"] * 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = rma(data=d, mods="~ ablat + ablat2")
    assert res.coef_names == ("intrcpt", "ablat")
    assert any("Redundant" in str(x.message) for x in w)


def test_print_matches_metafor_layout(dat: pd.DataFrame) -> None:
    text = str(rma(data=dat))
    assert "Random-Effects Model (k = 13; tau^2 estimator: REML)" in text
    assert "Q(df = 12) = 152.2330, p-val < .0001" in text
    assert " -0.7145  0.1798  -3.9744  <.0001  -1.0669  -0.3622  ***" in text


def test_too_many_parameters() -> None:
    with pytest.raises(ValueError, match="larger than the number of observations"):
        rma([0.1, 0.2], [0.01, 0.01], mods=[1.0, 2.0])
