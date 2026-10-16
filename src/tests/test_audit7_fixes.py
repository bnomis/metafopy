"""Regression tests for the audit-7 and audit-6 fixes (cross-checked against R metafor 5.2-1 when the audit was run)."""

import numpy
import refutil

import metafopy
import metafopy.core.padjust


def test_ls_anova_adjust_and_digits() -> None:
    res = metafopy.rma_uni('yi', 'vi', mods='~ ablat', scale='~ year', data=refutil.bcg())
    X = numpy.array([[0, 1], [1, 0]], dtype=float)
    plain = metafopy.anova(res, X=X)
    holm = metafopy.anova(res, X=X, adjust='holm', digits=2)
    expected = metafopy.core.padjust.p_adjust(plain.pval, 'holm')
    numpy.testing.assert_allclose(holm.pval, expected)
    assert holm.digits == 2
    assert plain.digits == res.digits
