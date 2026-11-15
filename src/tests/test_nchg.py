"""Tests of the analytic information of the exact conditional likelihood in ``metafopy.core.nchg``."""

import math
import typing

import numpy
import pandas
import pytest
import scipy.stats

import metafopy
import metafopy.core.nchg
import metafopy.datasets

ARGS: dict[str, typing.Any] = {'ai': 'ai', 'ci': 'ci', 'measure': 'OR', 'method': 'FE', 'model': 'CM.EL', 'n1i': 'n1i', 'n2i': 'n2i'}
PNEUMONIA = {'ai': [1, 0, 1, 13, 2, 4, 6], 'bi': [155, 89, 43, 64, 157, 43, 76], 'ci': [12, 3, 6, 27, 5, 6, 0], 'di': [69, 40, 32, 53, 165, 43, 148]}
TIGHT = metafopy.core.nchg.IntegrationControl(calc='dnoncenhypergeom')


def tables_of(data: pandas.DataFrame) -> metafopy.core.nchg.Tables:
    n1, n2 = data['n1i'].to_numpy(float), data['n2i'].to_numpy(float)
    ai, ci = data['ai'].to_numpy(float), data['ci'].to_numpy(float)
    return metafopy.core.nchg.Tables.from_counts(ai, n1 - ai, ci, n2 - ci)


def pneumonia() -> pandas.DataFrame:
    frame = pandas.DataFrame(PNEUMONIA)
    frame['n1i'], frame['n2i'] = frame['ai'] + frame['bi'], frame['ci'] + frame['di']
    return frame


def collins() -> pandas.DataFrame:
    data = metafopy.datasets.load('collins1985a').rename(columns={'b.xci': 'ai', 'nci': 'n1i', 'b.xti': 'ci', 'nti': 'n2i'})
    return data.dropna(subset=['ai', 'n1i', 'ci', 'n2i']).reset_index(drop=True)


@pytest.mark.parametrize('counts', [(3, 7, 4, 9), (0, 5, 2, 8), (10, 30, 12, 25), (2, 2, 1, 1)])
@pytest.mark.parametrize('log_or', [-1.3, 0.0, 0.4, 2.0])
def test_conditional_moments_match_scipy(counts: tuple[int, int, int, int], log_or: float) -> None:
    a, b, c, d = counts
    tables = metafopy.core.nchg.Tables.from_counts(numpy.array([a]), numpy.array([b]), numpy.array([c]), numpy.array([d]))
    mean, variance = metafopy.core.nchg.conditional_moments(tables, 0, log_or)
    ref = scipy.stats.nchypergeom_fisher(a + b + c + d, a + b, a + c, math.exp(log_or))
    assert mean == pytest.approx(float(ref.mean()), rel=1e-10)
    assert variance == pytest.approx(float(ref.var()), rel=1e-9)


def test_conditional_moments_clamp_gives_zero_variance() -> None:
    tables = metafopy.core.nchg.Tables.from_counts(numpy.array([3.0]), numpy.array([7.0]), numpy.array([4.0]), numpy.array([9.0]))
    mean, variance = metafopy.core.nchg.conditional_moments(tables, 0, 40.0)
    assert variance == 0.0
    assert mean == pytest.approx(7.0)


def second_difference(beta: numpy.ndarray, tables: metafopy.core.nchg.Tables, X: numpy.ndarray, step: float = 3e-4) -> numpy.ndarray:
    p = len(beta)
    hess = numpy.zeros((p, p))

    def f(b: numpy.ndarray) -> float:
        return metafopy.core.nchg.neg_loglik(b, tables, X, False, TIGHT)

    for j in range(p):
        for k in range(p):
            ej, ek = numpy.eye(p)[j] * step, numpy.eye(p)[k] * step
            hess[j, k] = (f(beta + ej + ek) - f(beta + ej - ek) - f(beta - ej + ek) + f(beta - ej - ek)) / (4 * step * step)
    return hess


@pytest.mark.parametrize('moderator', [False, True])
def test_fixed_information_matches_second_difference(moderator: bool) -> None:
    data = metafopy.datasets.load('nielweise2007')
    tables = tables_of(data)
    X = numpy.ones((len(data), 1))
    beta = numpy.array([0.3])
    if moderator:
        X = numpy.column_stack([X, numpy.linspace(-1.0, 1.0, len(data))])
        beta = numpy.array([0.3, -0.2])
    info = metafopy.core.nchg.fixed_information(beta, tables, X)
    numpy.testing.assert_allclose(info, second_difference(beta, tables, X), rtol=1e-7)


@pytest.mark.parametrize(
    ('frame', 'expected'),
    [
        (metafopy.datasets.load('nielweise2007'), 0.22403481656),
        (collins(), 0.09955789399),
        (pneumonia(), 0.25175350287),
    ],
    ids=['nielweise2007', 'collins1985a', 'pneumonia'],
)
def test_fixed_effects_se_matches_spec_measurement(frame: pandas.DataFrame, expected: float) -> None:
    fit = metafopy.rma_glmm(**ARGS, data=frame)
    X = numpy.ones((len(frame), 1))
    info = metafopy.core.nchg.fixed_information(fit.beta, tables_of(frame), X)
    assert math.sqrt(numpy.linalg.inv(info)[0, 0]) == pytest.approx(expected, rel=1e-9)
