"""The printout of vif() for location-scale models against R metafor 5.2-1 (R 4.6.2, printed 2026-11-04).

R's ``vif.rma`` gives the result of a location-scale model no ``digits``, so ``print.vif.rma`` formats both parts with ``digits = NULL``:
four decimals for every number (``formatC``'s default) and one for the p-values (``fmtp``), whatever ``digits`` ``vif()`` was given.
"""

import typing
import warnings

import numpy
import pytest

import metafopy
import metafopy.methods.ls_vif


@pytest.fixture(scope='module')
def bangert() -> metafopy.RMALSResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma(yi='yi', vi='vi', mods='~ I(ni/100) + meta', scale='~ I(ni/100) + imag', data=metafopy.datasets.load('dat.bangertdrowns2004'))
    assert isinstance(res, metafopy.RMALSResult)
    return res


def lines(out: typing.Any) -> list[str]:
    assert isinstance(out, metafopy.methods.ls_vif.LSVIFResult)
    return [line.rstrip() for line in str(out).splitlines()]


def test_digits_of_vif_are_not_used_by_the_print(bangert: metafopy.RMALSResult) -> None:
    """Audit cases ``options/vif/ls_*/digits=2``: R prints ``1.0094`` where ``digits=2`` would give ``1.01``."""
    text = lines(metafopy.vif(bangert, digits=2))
    assert 'I(ni/100)    meta' in text
    assert '   1.0094  1.0094' in text
    assert '   1.0321  1.0321' in text


def test_table_prints_pvalues_with_one_digit(bangert: metafopy.RMALSResult) -> None:
    """Audit cases ``options/vif/ls_*/table=true``: R's p-values are ``<.1`` and ``0.1``."""
    text = lines(metafopy.vif(bangert, table=True))
    # auditme: accept[duplicated-word] R's printed row: the intercept has no vif and no sif, both NA
    assert 'intrcpt     0.2303 0.0655  3.5166  <.1  0.1019  0.3586     NA     NA' in text
    assert 'I(ni/100)  -0.9995 0.6087 -1.6421  0.1 -2.1925  0.1935 1.0321 1.0159' in text


def test_explicit_print_digits_are_used(bangert: metafopy.RMALSResult) -> None:
    """``print(vif(res), digits = 2)`` in R passes the digits on to both parts."""
    out: typing.Any = metafopy.vif(bangert)  # vif() is typed by its rma_uni overload
    assert isinstance(out, metafopy.methods.ls_vif.LSVIFResult)
    text = [line.rstrip() for line in out.summary(2).splitlines()]
    assert '     1.01  1.01' in text


# R 4.6.2, metafor 5.2-1: vif(res, sim = 50, seed = 42) and set.seed(1234); vif(res, sim = 50), the first draws of each part's first set (printed 2026-11-05)
R_SIM = {
    'seed': {'beta': (0.26, [1.054888452, 1.005777189, 1.00169033]), 'alpha': (0.64, [1, 1, 1.032671213])},
    'stream': {'beta': (0.3, [1.045362421, 1.00908478, 1.001915041, 1.000840407]), 'alpha': (0.76, [1.000029626, 1.000000001, 1.060457984, 1.000000001])},
}


@pytest.mark.parametrize('how', ['seed', 'stream'])
def test_sim_permutes_the_data_variables_as_r(bangert: metafopy.RMALSResult, how: str) -> None:
    """metafor's ``.compvifsim`` permutes the variables of the data a formula uses and rebuilds the model matrix (``I(ni/100)`` keeps its
    transformation), and without ``seed`` continues the global stream; audit cases ``options/vif/ls_*/sim=true``."""
    out: typing.Any  # vif() is typed by its rma_uni overload
    if how == 'seed':
        out = metafopy.vif(bangert, sim=50, seed=42, progbar=False)
    else:
        metafopy.set_seed(1234)
        out = metafopy.vif(bangert, sim=50, progbar=False)
    assert isinstance(out, metafopy.methods.ls_vif.LSVIFResult)
    for part, (prop, first) in R_SIM[how].items():
        res = getattr(out, part)
        sims = numpy.asarray(res.sim)
        assert res.sim_type == 'data'
        assert sims[: len(first), 0] == pytest.approx(first, rel=1e-6)
        if part == 'beta':
            assert res.prop == pytest.approx([prop, prop])
        else:
            # one scale refit (iteration 35 with seed 42, 12 in the stream) runs along a ridge towards alpha = -inf: R's nlminb stops at
            # -17.5 with a finite Hessian and a VIF of 1, metafopy's optimiser goes on to -40.8 where the numerical Hessian is NaN, so the
            # draw is dropped (an optimiser stop on a flat scale likelihood, accepted by decision of the operator, 2026-11-04)
            assert int(numpy.isnan(sims[:, 0]).sum()) == 1
