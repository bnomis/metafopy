import typing
import warnings

import hypothesis
import pytest
import rstrategies

import metafopy
import metafopy.errors
import metafopy.formula


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('Intercept', 'intrcpt'),
        ('ablat', 'ablat'),
        ('alloc[T.random]', 'allocrandom'),
        ('C(alloc)[T.random]', 'allocrandom'),
        ('ablat:alloc[T.random]', 'ablat:allocrandom'),
    ],
)
def test_r_column_name(name: str, expected: str) -> None:
    assert metafopy.formula.r_column_name(name) == expected


def test_parse_random_simple() -> None:
    assert metafopy.parse_random('~ 1 | study') == metafopy.RandomTerm(None, ('study',), nested=False)


def test_parse_random_nested() -> None:
    term = metafopy.parse_random('~1|district/school')
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    assert term == metafopy.RandomTerm(None, ('district', 'school'), nested=True)
    assert str(term) == '~ 1 | district/school'


def test_parse_random_inner_outer() -> None:
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    assert metafopy.parse_random('~ outcome | trial') == metafopy.RandomTerm('outcome', ('trial',), nested=False)


@pytest.mark.parametrize('bad', ['1 | study', '~ 1', '~ 1 | ', '~ x | a/b', '~ 1 | a//b'])
def test_parse_random_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match=r'random effects formula|cannot also be nested'):
        metafopy.parse_random(bad)


@pytest.fixture(scope='module')
# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
def bcg() -> typing.Any:
    return metafopy.datasets.load('dat.bcg')


# R 4.6.2: colnames(model.matrix(f, dat.bcg)) and the first three rows of the last column (printed 2026-11-04)
@pytest.mark.parametrize(
    ('formula', 'names', 'head'),
    [
        ('~ I(ablat > 30)', ('intrcpt', 'I(ablat > 30)TRUE'), [1.0, 1.0, 1.0]),
        ('~ 0 + I(ablat > 30)', ('I(ablat > 30)FALSE', 'I(ablat > 30)TRUE'), [1.0, 1.0, 1.0]),
        ('~ 0 + I(1|ablat)', ('I(1 | ablat)FALSE', 'I(1 | ablat)TRUE'), [1.0, 1.0, 1.0]),
        ('~ I((ablat > 30) | (year < 1970))', ('intrcpt', 'I((ablat > 30) | (year < 1970))TRUE'), [1.0, 1.0, 1.0]),
        ('~ I(ablat > 30 & year < 1970)', ('intrcpt', 'I(ablat > 30 & year < 1970)TRUE'), [1.0, 1.0, 1.0]),
        ('~ I(!(ablat > 30))', ('intrcpt', 'I(!(ablat > 30))TRUE'), [0.0, 0.0, 0.0]),
        ('~ ablat + I(ablat > 30 | year < 1970)', ('intrcpt', 'ablat', 'I(ablat > 30 | year < 1970)TRUE'), [1.0, 1.0, 1.0]),
    ],
)
def test_logical_terms_are_r_factors(bcg: typing.Any, formula: str, names: tuple[str, ...], head: list[float]) -> None:
    """R's ``|``, ``&`` and ``!`` are logical, and ``model.matrix`` codes a logical term as a factor with levels FALSE and TRUE."""
    X, got, _ = metafopy.formula.design_matrix(formula, bcg)
    assert got == names
    assert X[:3, -1].tolist() == head


def test_logical_terms_keep_r_missing_values(bcg: typing.Any) -> None:
    """R: ``1 | NA`` is TRUE, so the row stays; ``!NA`` is NA, so it goes."""
    dat = bcg.copy()
    dat.loc[1, 'ablat'] = float('nan')
    _, _, missing = metafopy.formula.design_matrix('~ 0 + I(1|ablat)', dat)
    assert not missing.any()
    X, _, missing = metafopy.formula.design_matrix('~ I(!ablat)', dat)
    assert missing.tolist()[:3] == [False, True, False]
    assert X[0, -1] == 0.0


def test_scale_with_a_logical_or_fits_as_r(bcg: typing.Any) -> None:
    """R: ``scale = ~ 0 + I(1|ablat)`` is ``scale = ~ 1`` (the FALSE column is dropped), log-likelihood -8.10687."""
    dat = metafopy.escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma('yi', 'vi', mods='~ ablat + year', scale='~ 0 + I(1|ablat)', data=dat)
    assert res.fit_stats_for(True).ll == pytest.approx(-8.10687, abs=1e-5)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=rstrategies.finite_floats(-1e300, 1e300))
def test_r_number_is_fifteen_digits_and_integers_are_plain(x: float) -> None:
    """Candidate ``src/metafopy/formula.py::r_number``."""
    text = metafopy.formula.r_number(x)
    assert float(text) == float(f'{x:.15g}')
    if x == int(x) and abs(x) < 1e15:
        assert text == str(int(x))


@pytest.mark.parametrize(
    ('rhs', 'expected'),
    [
        ('x + z', 'x + z'),
        ('x*z', 'x*z'),
        ('x:z + z', 'x:z + z'),
        ('z:x + x + z', 'z:x + x + z'),
        ('0 + outcome + I(year - 1985):outcome', '0 + outcome + outcome:I(year - 1985)'),
        ('x + z + z:x - 1', '0 + x + z + x:z'),
        ('x + z + outcome + outcome:z:x', 'x + z + outcome + x:z:outcome'),
    ],
)
def test_r_term_order(rhs: str, expected: str) -> None:
    assert metafopy.formula.r_term_order(rhs) == expected


def test_r_term_order_raises_for_a_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metafopy.formula, 'r_tokens', lambda text: [] if text == 'x:z' else text.split())
    with pytest.raises(metafopy.errors.InternalError):
        metafopy.formula.r_term_order('z:x')
