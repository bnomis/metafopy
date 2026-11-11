"""``property`` cases: identities, invariances and estimating equations that theory guarantees.

A case that is not a comparison of values names a ``check`` function of this module; it is called with the case and the result and raises
``AssertionError`` when the property fails.
"""

import typing

import numpy
import pytest

import correctness.cases


@pytest.mark.parametrize('case', correctness.cases.case_params('property'))
# auditme: accept[paired-sibling-helpers] one test module per kind of evidence, as the spec lays out the suite
def test_property(case: correctness.cases.Case) -> None:
    correctness.cases.check_case(case, correctness.cases.run_case(case))


def i2_follows_h2(case: correctness.cases.Case, result: typing.Any) -> None:
    """``I2`` (a percentage) is ``H2`` as a share of the total variability: ``I2 = 100 (H2 - 1) / H2`` (Higgins and Thompson, 2002)."""
    tolerance = case.tolerance or correctness.cases.PROPERTY_TOLERANCE
    expected = 100 * (result.H2 - 1) / result.H2
    if not numpy.isclose(result.I2, expected, rtol=0, atol=tolerance):
        raise AssertionError(f'{case.id}: I2 is {result.I2!r}, 100 (H2 - 1) / H2 is {expected!r} for H2 = {result.H2!r}')
