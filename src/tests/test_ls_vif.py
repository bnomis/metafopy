"""The printout of vif() for location-scale models against R metafor 5.2-1 (R 4.6.2, printed 2026-11-04).

R's ``vif.rma`` gives the result of a location-scale model no ``digits``, so ``print.vif.rma`` formats both parts with ``digits = NULL``:
four decimals for every number (``formatC``'s default) and one for the p-values (``fmtp``), whatever ``digits`` ``vif()`` was given.
"""

import typing
import warnings

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
    out = metafopy.vif(bangert)
    assert isinstance(out, metafopy.methods.ls_vif.LSVIFResult)
    text = [line.rstrip() for line in out.summary(2).splitlines()]
    assert '     1.01  1.01' in text
