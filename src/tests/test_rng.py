"""RRandom.sample_int against base R's sample.int (R 4.6.2, set.seed(s); sample.int(...)).

The draws below were printed by Rscript. They reach the private helpers behind sample_int: unif_index, prob_sample_replace and
prob_sample_no_replace.
"""

import numpy
import pytest

import metafopy
import metafopy.errors


def draw(seed: int, *args: object, **kwargs: object) -> list[int]:
    return [int(x) for x in metafopy.set_seed(seed).sample_int(*args, **kwargs)]  # type: ignore[arg-type]


def test_sample_int_uniform_without_replacement() -> None:
    assert draw(1, 10, 3) == [9, 4, 7]


def test_sample_int_uniform_with_replacement() -> None:
    assert draw(3, 100, 5, replace=True) == [5, 58, 12, 36, 58]


def test_sample_int_prob_without_replacement() -> None:
    p = numpy.arange(1, 11) / 55
    assert draw(42, 10, 4, False, prob=p) == [3, 4, 9, 5]
    assert draw(42, 6, 3, prob=[5, 1, 1, 1, 1, 1]) == [2, 6, 1]


def test_sample_int_prob_with_replacement() -> None:
    assert draw(7, 20, 5, True, prob=numpy.arange(1, 21)) == [2, 16, 19, 20, 18]


def test_sample_int_prob_single_draw_without_replacement_uses_replacement_path() -> None:
    assert draw(3, 7, 1, prob=numpy.arange(1, 8)) == [7]


def test_sample_int_errors() -> None:
    rng = metafopy.set_seed(1)
    with pytest.raises(metafopy.errors.InputError, match='incorrect number of probabilities'):
        rng.sample_int(3, 2, prob=[0.5, 0.5])
    with pytest.raises(metafopy.errors.InputError, match='invalid probabilities'):
        rng.sample_int(3, 2, prob=[0.5, -0.1, 0.6])
    with pytest.raises(metafopy.errors.InputError, match='invalid probabilities'):
        rng.sample_int(3, 2, prob=[0.0, 0.0, 0.0])
    with pytest.raises(metafopy.errors.InputError, match='larger than the population'):
        rng.sample_int(3, 4)


def test_sample_int_walker_alias_with_equal_weights() -> None:
    assert draw(42, 300, 5, True, prob=numpy.ones(300)) == [49, 153, 228, 122, 128]


def test_sample_int_walker_alias_with_unequal_weights() -> None:
    assert draw(42, 300, 6, True, prob=numpy.arange(1, 301) / 45150) == [276, 153, 228, 122, 128, 289]
