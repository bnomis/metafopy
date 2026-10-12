from __future__ import annotations

import pytest

from metafopy import RandomTerm, parse_random
from metafopy._formula import r_column_name


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Intercept", "intrcpt"),
        ("ablat", "ablat"),
        ("alloc[T.random]", "allocrandom"),
        ("C(alloc)[T.random]", "allocrandom"),
        ("ablat:alloc[T.random]", "ablat:allocrandom"),
    ],
)
def test_r_column_name(name: str, expected: str) -> None:
    assert r_column_name(name) == expected


def test_parse_random_simple() -> None:
    assert parse_random("~ 1 | study") == RandomTerm(None, ("study",), nested=False)


def test_parse_random_nested() -> None:
    term = parse_random("~1|district/school")
    assert term == RandomTerm(None, ("district", "school"), nested=True)
    assert str(term) == "~ 1 | district/school"


def test_parse_random_inner_outer() -> None:
    assert parse_random("~ outcome | trial") == RandomTerm("outcome", ("trial",), nested=False)


@pytest.mark.parametrize("bad", ["1 | study", "~ 1", "~ 1 | ", "~ x | a/b", "~ 1 | a//b"])
def test_parse_random_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match=r"random effects formula|cannot also be nested"):
        parse_random(bad)
