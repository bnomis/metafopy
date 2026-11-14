"""``published`` cases: expected values printed in the defining publication or a textbook worked example."""

import collections.abc
import typing

import numpy
import pytest

import correctness.cases
import metafopy


@pytest.mark.parametrize('case', correctness.cases.case_params('published'))
# auditme: accept[paired-sibling-helpers] one test module per kind of evidence, as the spec lays out the suite
def test_published(case: correctness.cases.Case) -> None:
    correctness.cases.check_case(case, correctness.cases.run_case(case))


def scalar(value: typing.Any) -> float:
    return float(numpy.asarray(value, dtype=numpy.float64).reshape(-1)[0])


def check_values(case: correctness.cases.Case, values: collections.abc.Mapping[str, float]) -> None:
    unknown = set(case.printed) - set(values)
    if unknown:
        raise LookupError(f'{case.id}: this check does not know the printed name(s) {sorted(unknown)}; it knows {sorted(values)}')
    for name in case.printed:
        correctness.cases.check_printed(case, name, values[name])


def back_transformed(case: correctness.cases.Case, result: typing.Any) -> None:
    """The pooled ratio and its confidence bounds, exp(beta), exp(ci_lb) and exp(ci_ub), as a source prints them (Mantel and Haenszel, 1959, Table 1)."""
    check_values(
        case,
        {
            'exp_beta': float(numpy.exp(scalar(result.beta))),
            'exp_ci_lb': float(numpy.exp(scalar(result.ci_lb))),
            'exp_ci_ub': float(numpy.exp(scalar(result.ci_ub))),
        },
    )


def prediction_interval(case: correctness.cases.Case, result: typing.Any) -> None:
    """The prediction interval of the pooled effect, ``predict(result)`` (Borenstein et al., 2009, chapter 17, "Prediction intervals")."""
    predicted = metafopy.predict(result)
    values = {'pred': scalar(predicted.pred), 'pi_lb': scalar(predicted.pi_lb), 'pi_ub': scalar(predicted.pi_ub)}
    values.update({f'exp_{name}': float(numpy.exp(value)) for name, value in list(values.items())})
    check_values(case, values)


def confint_ht(case: correctness.cases.Case, result: typing.Any) -> None:
    """The test-based intervals of tau2, I2, H2 and H, ``confint(result, type='ht')`` (Higgins and Thompson, 2002; Borenstein et al., 2009, eqs 16.14-16.19)."""
    table = metafopy.confint(result, type='ht').random
    if table is None:
        raise LookupError(f'{case.id}: confint gives no random-effects table')
    values: dict[str, float] = {}
    for row, name in (('tau^2', 'tau2'), ('I^2(%)', 'I2'), ('H^2', 'H2')):
        values[f'{name}_lb'] = float(table.loc[row, 'ci.lb'])
        values[f'{name}_ub'] = float(table.loc[row, 'ci.ub'])
    values['H_lb'] = float(numpy.sqrt(values['H2_lb']))
    values['H_ub'] = float(numpy.sqrt(values['H2_ub']))
    check_values(case, values)
