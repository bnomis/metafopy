"""``derived`` cases: expected values from ``formulas.py``, coded from the publications' equations."""

import typing

import numpy
import pytest

import correctness.cases
import metafopy


@pytest.mark.parametrize('case', correctness.cases.case_params('derived'))
# auditme: accept[paired-sibling-helpers] one test module per kind of evidence, as the spec lays out the suite
def test_derived(case: correctness.cases.Case) -> None:
    correctness.cases.check_case(case, correctness.cases.run_case(case))


def numbers_agree(case: correctness.cases.Case, name: str, actual: typing.Any, expected: typing.Any) -> None:
    allowed = (case.tolerance or correctness.cases.DERIVED_TOLERANCE) * numpy.maximum(1.0, numpy.abs(expected))
    if not numpy.all(numpy.abs(numpy.asarray(actual) - numpy.asarray(expected)) <= allowed):
        raise AssertionError(f'{case.id}: {name}: metafopy gives {actual!r}, the reference gives {expected!r}')


def confint_bounds(case: correctness.cases.Case, result: typing.Any, kind: str, formula: str) -> None:
    table = metafopy.confint(result, type=kind, tol=1e-12).random
    assert table is not None
    expected = correctness.cases.formula_value(case, f'formulas.{formula}', correctness.cases.case_frame(case))
    numbers_agree(case, f'confint {kind}', [table['ci.lb'].iloc[0], table['ci.ub'].iloc[0]], expected)


def confint_qp(case: correctness.cases.Case, result: typing.Any) -> None:
    """The Q-profile interval of tau2 (Viechtbauer, 2007)."""
    confint_bounds(case, result, 'qp', 'qprofile_bounds')


def confint_genq(case: correctness.cases.Case, result: typing.Any) -> None:
    """The generalised Q interval of tau2 (Jackson, 2013)."""
    confint_bounds(case, result, 'genq', 'general_q_bounds')


def confint_pl(case: correctness.cases.Case, result: typing.Any) -> None:
    """The profile-likelihood interval of tau2 (Hardy and Thompson, 1996)."""
    confint_bounds(case, result, 'pl', 'profile_likelihood_bounds')


def confint_ht(case: correctness.cases.Case, result: typing.Any) -> None:
    """The test-based interval of tau2 (Higgins and Thompson, 2002)."""
    confint_bounds(case, result, 'ht', 'higgins_thompson_bounds')


def confint_wald(case: correctness.cases.Case, result: typing.Any) -> None:
    """The Wald interval of tau2."""
    confint_bounds(case, result, 'wald', 'wald_tau2_bounds')


def confint_wald_log(case: correctness.cases.Case, result: typing.Any) -> None:
    """The Wald interval of tau2 on the log scale."""
    confint_bounds(case, result, 'wald.log', 'wald_log_bounds')


def confint_wald_sqrt(case: correctness.cases.Case, result: typing.Any) -> None:
    """The Wald interval of tau2 on the scale of tau."""
    confint_bounds(case, result, 'wald.sqrt', 'wald_sqrt_bounds')


def predict_pi(case: correctness.cases.Case, result: typing.Any) -> None:
    """The prediction interval at each row of the model matrix (Riley, Higgins and Deeks, 2011)."""
    frame = correctness.cases.case_frame(case)
    predicted = metafopy.predict(result)
    for name, formula, actual in (('pi_lb', 'prediction_lb', predicted.pi_lb), ('pi_ub', 'prediction_ub', predicted.pi_ub)):
        numbers_agree(case, name, actual, correctness.cases.formula_value(case, f'formulas.{formula}', frame))
