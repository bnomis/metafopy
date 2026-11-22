"""Property tests for the copies of R's ``pmatch``, ``charmatch``, ``match.arg``, ``make.unique`` and ``make.names``.

Each test names the candidate it covers in its docstring (the id in ``agents/audit-reviews/ledger/T2.jsonl``) and checks the property the
candidate's ledger reason states.
"""

import itertools
import re
import typing
import warnings

import hypothesis
import hypothesis.strategies
import pytest
import rstrategies

import metafopy.core.clubsandwich
import metafopy.core.padjust
import metafopy.effects.datatools
import metafopy.effects.fivenum_estimators
import metafopy.effects.measures
import metafopy.errors
import metafopy.interactive.forest
import metafopy.metadata
import metafopy.methods.anova
import metafopy.methods.common
import metafopy.methods.emm_adjust
import metafopy.methods.emm_grid
import metafopy.methods.emm_nesting
import metafopy.methods.hettest
import metafopy.methods.matreg
import metafopy.methods.predict
import metafopy.methods.robust
import metafopy.models.rma_glmm
import metafopy.models.rma_mv
import metafopy.models.sel_opt
import metafopy.models.sel_spec

SETTINGS = hypothesis.settings(max_examples=100, deadline=None)
st = hypothesis.strategies
COLLIDING_LABELS = st.sampled_from(['a', 'a.1', 'a.2', 'b'])
TEST_NAMES = st.sampled_from(['lrt', 'score', 'wald', 'adn', 'adx2', 'ksn', 'ksx2'])

small_text = st.text(alphabet='abc', max_size=6)
small_words = st.text(alphabet='abcd', min_size=1, max_size=6)
unique_words = st.lists(small_words, min_size=1, max_size=6, unique=True)
ascii_text = st.text(alphabet=st.characters(codec='ascii'), max_size=12)
R_IDENTIFIER = re.compile(r'(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*|\.')


def prefixes(options: list[str]) -> st.SearchStrategy[tuple[str, str]]:
    """A (option, non-empty prefix of it) pair."""
    return st.sampled_from(options).flatmap(
        lambda option: st.tuples(st.just(option), st.integers(min_value=1, max_value=len(option))).map(lambda t: (t[0], t[0][: t[1]]))
    )


# core/padjust.py


@SETTINGS
@hypothesis.given(data=st.data())
def test_match_adjust_prefixes(data: st.DataObject) -> None:
    """src/metafopy/core/padjust.py::match_adjust: booleans map to bonferroni and none, a method returns itself, a unique prefix its method."""
    methods = list(metafopy.core.padjust.P_ADJUST_METHODS)
    assert metafopy.core.padjust.match_adjust(True) == 'bonferroni'
    assert metafopy.core.padjust.match_adjust(False) == 'none'
    option, prefix = data.draw(prefixes(methods))
    assert metafopy.core.padjust.match_adjust(option) == option
    hits = [m for m in methods if m.startswith(prefix)]
    if prefix in methods:
        assert metafopy.core.padjust.match_adjust(prefix) == prefix
    elif len(hits) == 1:
        assert metafopy.core.padjust.match_adjust(prefix) == option
    else:
        with pytest.raises(metafopy.errors.InputError):
            metafopy.core.padjust.match_adjust(prefix)


@SETTINGS
@hypothesis.given(text=ascii_text)
def test_match_adjust_arbitrary_text(text: str) -> None:
    """src/metafopy/core/padjust.py::match_adjust: a string either gives a method it is a prefix of or raises InputError."""
    methods = metafopy.core.padjust.P_ADJUST_METHODS
    try:
        result = metafopy.core.padjust.match_adjust(text)
    except metafopy.errors.InputError:
        assert text not in methods
        assert not text or sum(m.startswith(text) for m in methods) != 1
    else:
        assert result in methods
        assert result.startswith(text)


# effects/datatools.py


@SETTINGS
@hypothesis.given(table=st.lists(small_text, max_size=6), x=small_text)
def test_charmatch_agrees_with_naive_oracle(table: list[str], x: str) -> None:
    """src/metafopy/effects/datatools.py::charmatch: an exact member gives its index, a unique prefix its match, otherwise None."""
    result = metafopy.effects.datatools.charmatch(x, table)
    exact = [i for i, name in enumerate(table) if name == x]
    partial = [i for i, name in enumerate(table) if name.startswith(x)]
    if exact:
        assert result == exact[0]
    elif len(partial) == 1:
        assert result == partial[0]
    else:
        assert result is None


@SETTINGS
@hypothesis.given(names=st.lists(ascii_text, max_size=8), unique=st.booleans())
def test_make_names_gives_valid_names(names: list[str], unique: bool) -> None:
    """src/metafopy/effects/datatools.py::make_names: valid R identifiers, idempotent, and distinct for unique=True.

    ASCII input only: R's ``make.names`` accepts the letters of the current locale (``?make.names``), so a non-ASCII letter is valid in R.
    """
    out = metafopy.effects.datatools.make_names(names, unique=unique)
    assert len(out) == len(names)
    for name in out:
        assert R_IDENTIFIER.fullmatch(name)
        assert name not in metafopy.effects.datatools.R_RESERVED
    if unique:
        assert len(set(out)) == len(out)
    else:
        assert metafopy.effects.datatools.make_names(out) == out


@SETTINGS
@hypothesis.given(
    x=st.floats(min_value=-1e12, max_value=1e12, allow_nan=False) | st.sampled_from([float('nan'), float('inf'), float('-inf')]),
    digits=st.integers(min_value=0, max_value=12),
)
def test_r_round_error_bound(x: float, digits: int) -> None:
    """src/metafopy/effects/datatools.py::r_round: idempotent, NaN and infinities pass through, and the error is at most half a unit."""
    result = metafopy.effects.datatools.r_round(x, digits)
    if x != x:
        assert result != result
    elif x in (float('inf'), float('-inf')):
        assert result == x
    else:
        assert metafopy.effects.datatools.r_round(result, digits) == result
        assert abs(result - x) <= 0.5 * 10.0**-digits * (1 + 1e-9) + 4 * abs(x) * 2.0**-52


@SETTINGS
@hypothesis.given(x=st.lists(st.none() | st.text(alphabet='ab', max_size=5), max_size=8), minlen=st.none() | st.integers(min_value=0, max_value=6))
def test_shorten_names_cuts_to_a_common_length(x: list[str | None], minlen: int | None) -> None:
    """src/metafopy/effects/datatools.py::shorten_names: None stays None, one common cut length, distinct inputs stay distinct."""
    out = metafopy.effects.datatools.shorten_names(x, minlen)
    assert len(out) == len(x)
    assert [v is None for v in out] == [v is None for v in x]
    values = [v for v in x if v is not None]
    shortened = [o for o in out if o is not None]
    assert len(set(shortened)) == len(set(values))
    if values:
        length = max(len(o) for o in shortened)
        assert all(o == v[:length] for o, v in zip(shortened, values, strict=True))


# effects/fivenum_estimators.py


@SETTINGS
@hypothesis.given(n=st.integers(min_value=1, max_value=10**6), kind=st.sampled_from(['S1', 'S2', 'S3']))
def test_sum_stat_index_ranks(n: int, kind: str) -> None:
    """src/metafopy/effects/fivenum_estimators.py::sum_stat_index: ranks are non-decreasing, in [1, n], and three or five in number."""
    ranks = metafopy.effects.fivenum_estimators.sum_stat_index(n, kind)
    assert len(ranks) == (5 if kind == 'S3' else 3)
    assert ranks == sorted(ranks)
    assert all(1 <= r <= n for r in ranks)


# effects/measures.py, metadata.py, interactive/forest.py


# metafor's .make.unique also renames a label that occurs once when it collides with a generated one, so the singleton property holds only for
# labels without a collision: `.make.unique(c("a", "a", "a.1"))` gives "a.1.1" "a.2" "a.1.2" in R (metafor 5.2-1, checked 2026-11-22).


@SETTINGS
@hypothesis.given(labels=st.lists(COLLIDING_LABELS, max_size=8))
def test_make_unique_dot1_is_unique(labels: list[str]) -> None:
    """src/metafopy/effects/measures.py::make_unique_dot1: same length and no duplicates, also when labels collide with generated ones."""
    out = metafopy.effects.measures.make_unique_dot1(labels)
    assert len(out) == len(labels)
    assert len(set(out)) == len(out)


@SETTINGS
@hypothesis.given(labels=st.lists(small_text, max_size=10))
def test_make_unique_dot1_keeps_singletons(labels: list[str]) -> None:
    """src/metafopy/effects/measures.py::make_unique_dot1: a label occurring once is unchanged (labels without dots, so none collide)."""
    out = metafopy.effects.measures.make_unique_dot1(labels)
    for label, result in zip(labels, out, strict=True):
        if labels.count(label) == 1:
            assert result == label


@SETTINGS
@hypothesis.given(labels=st.lists(small_text, max_size=10))
def test_unique_labels_suffix_repeats(labels: list[str]) -> None:
    """src/metafopy/interactive/forest.py::unique_labels: length kept, first occurrence unchanged, the k-th repeat is 'label (k)'."""
    out = metafopy.interactive.forest.unique_labels(labels)
    assert len(out) == len(labels)
    for i, (label, result) in enumerate(zip(labels, out, strict=True)):
        k = labels[: i + 1].count(label)
        assert result == (label if k == 1 else f'{label} ({k})')


@SETTINGS
@hypothesis.given(labels=st.lists(small_text, max_size=10))
def test_make_unique_has_no_duplicates(labels: list[str]) -> None:
    """src/metafopy/metadata.py::make_unique: same length and no duplicates."""
    out = metafopy.metadata.make_unique(labels)
    assert len(out) == len(labels)
    assert len(set(out)) == len(out)


@pytest.mark.parametrize(
    ('labels', 'expected'),
    [
        (['a', 'a', 'a.1'], ['a', 'a.2', 'a.1']),
        (['a', 'a.1', 'a'], ['a', 'a.1', 'a.2']),
        (['a', 'a', 'a', 'a.2'], ['a', 'a.1', 'a.3', 'a.2']),
        (['b', 'b.1', 'b', 'b.1'], ['b', 'b.1', 'b.2', 'b.1.1']),
        (['x', 'x', 'x.1', 'x.1'], ['x', 'x.2', 'x.1', 'x.1.1']),
        (['1', '1'], ['1', '1.1']),
    ],
)
def test_make_unique_equals_r_make_unique(labels: list[str], expected: list[str]) -> None:
    """src/metafopy/metadata.py::make_unique equals R's ``make.unique(labels)`` (``LC_ALL=C.UTF-8 Rscript``, checked 2026-11-22)."""
    assert list(metafopy.metadata.make_unique(labels)) == expected


@SETTINGS
@hypothesis.given(labels=st.lists(COLLIDING_LABELS, max_size=8))
def test_make_unique_keeps_first_occurrences(labels: list[str]) -> None:
    """src/metafopy/metadata.py::make_unique: the first occurrence of every label is unchanged (R's make.unique, ``?make.unique``)."""
    out = metafopy.metadata.make_unique(labels)
    for i, label in enumerate(labels):
        if label not in labels[:i]:
            assert out[i] == label


# R: make.names(c("a b", "a b", "a.b"), unique = TRUE) gives "a.b.1" "a.b.2" "a.b" (checked 2026-11-22): the names it changed are made unique last.
def test_make_names_unique_orders_changed_names_last() -> None:
    """src/metafopy/effects/datatools.py::make_names: with unique=True, names that make.names changed are made unique after the unchanged ones."""
    assert metafopy.effects.datatools.make_names(['a b', 'a b', 'a.b'], unique=True) == ['a.b.1', 'a.b.2', 'a.b']


# methods/anova.py, methods/common.py


@SETTINGS
@hypothesis.given(data=st.data())
def test_check_test_prefixes(data: st.DataObject) -> None:
    """src/metafopy/methods/anova.py::check_test: None gives LRT, a non-empty prefix of LRT or Wald gives it, anything else raises."""
    assert metafopy.methods.anova.check_test(None) == 'LRT'
    choice, prefix = data.draw(prefixes(['LRT', 'Wald']))
    assert metafopy.methods.anova.check_test(prefix) == choice
    other = data.draw(st.text(alphabet='LRTWald', max_size=5).filter(lambda t: not (t and ('LRT'.startswith(t) or 'Wald'.startswith(t)))))
    with pytest.raises(metafopy.errors.ArgumentError):
        metafopy.methods.anova.check_test(other)


@SETTINGS
@hypothesis.given(choices=st.lists(small_words, min_size=1, max_size=6, unique=True), value=small_text)
def test_match_arg(choices: list[str], value: str) -> None:
    """src/metafopy/methods/common.py::match_arg: a member returns itself, a unique non-empty prefix its choice, empty or ambiguous raises."""
    hits = [c for c in choices if value and c.startswith(value)]
    if value in choices:
        assert metafopy.methods.common.match_arg(value, tuple(choices)) == value
    elif len(hits) == 1:
        assert metafopy.methods.common.match_arg(value, tuple(choices)) == hits[0]
    else:
        with pytest.raises(metafopy.errors.InputError):
            metafopy.methods.common.match_arg(value, tuple(choices))


# methods/emm_adjust.py


@SETTINGS
@hypothesis.given(methods=st.lists(small_text, max_size=6, unique=True), adjust=small_text)
def test_match_method_is_pmatch(methods: list[str], adjust: str) -> None:
    """src/metafopy/methods/emm_adjust.py::match_method: an exact member as is, else the only method starting with adjust, else None."""
    result = metafopy.methods.emm_adjust.match_method(adjust, methods)
    hits = [m for m in methods if adjust and m.startswith(adjust)]
    if adjust in methods:
        assert result == adjust
    elif len(hits) == 1:
        assert result == hits[0]
    else:
        assert result is None


@SETTINGS
@hypothesis.given(side=st.integers() | st.floats() | st.text(max_size=8))
def test_parse_side_is_a_tail_code(side: int | float | str) -> None:
    """src/metafopy/methods/emm_adjust.py::parse_side: any int, float or string gives -1, 0 or 1."""
    assert metafopy.methods.emm_adjust.parse_side(side) in {-1, 0, 1}


@SETTINGS
@hypothesis.given(data=st.data())
def test_parse_side_unique_prefix(data: st.DataObject) -> None:
    """src/metafopy/methods/emm_adjust.py::parse_side: a unique prefix of an option maps to the same code as the full option."""
    options = [*metafopy.methods.emm_adjust.SIDE_OPTIONS, *metafopy.methods.emm_adjust.SIDE_SYMBOLS]
    option, prefix = data.draw(prefixes(options))
    if prefix in options or sum(o.startswith(prefix) for o in options) == 1:
        assert metafopy.methods.emm_adjust.parse_side(prefix) == metafopy.methods.emm_adjust.parse_side(option)


# methods/emm_grid.py, methods/emm_nesting.py


@SETTINGS
@hypothesis.given(
    tokens=st.lists(st.sampled_from('abcdefgh'), min_size=1, max_size=8),
    variables=st.lists(st.sampled_from('abcdefghij'), max_size=10, unique=True),
    separator=st.sampled_from([' + ', ' * ', ':']),
)
def test_appearance_orders_by_first_position(tokens: list[str], variables: list[str], separator: str) -> None:
    """src/metafopy/methods/emm_grid.py::appearance: a permutation of variables ordered by first position, absent ones last."""
    formula = 'y ~ ' + separator.join(tokens)
    out = metafopy.methods.emm_grid.appearance(formula, variables)
    assert sorted(out) == sorted(variables)
    positions = [tokens.index(v) if v in tokens else len(tokens) for v in out]
    assert positions == sorted(positions)


@SETTINGS
@hypothesis.given(variables=st.lists(rstrategies.r_identifiers(), min_size=1, max_size=4, unique=True), data=st.data())
def test_factor_id(variables: list[str], data: st.DataObject) -> None:
    """src/metafopy/methods/emm_nesting.py::factor_id: v, C(v) and C( v, x) give v, and any other expression returns itself stripped."""
    var = data.draw(st.sampled_from(variables))
    for expr in (var, f'C({var})', f'C( {var}, x)', f'  C( {var} )  '):
        assert metafopy.methods.emm_nesting.factor_id(expr, variables) == var
    other = data.draw(st.text(alphabet='abxyz()+ ', min_size=1, max_size=8)).replace('C(', 'D(')
    hypothesis.assume(other.strip() not in variables)
    assert metafopy.methods.emm_nesting.factor_id(other, variables) == other.strip()


@SETTINGS
@hypothesis.given(sets=st.lists(st.lists(st.sampled_from('abcde'), max_size=4, unique=True), max_size=5))
def test_strip_supersets(sets: list[list[str]]) -> None:
    """src/metafopy/methods/emm_nesting.py::strip_supersets: inside the union of the sets and containing every smallest set."""
    out = metafopy.methods.emm_nesting.strip_supersets([list(s) for s in sets])
    union = {e for s in sets for e in s}
    assert set(out) <= union
    assert len(set(out)) == len(out)
    if sets:
        smallest = min(len(s) for s in sets)
        for s in sets:
            if len(s) == smallest:
                assert set(s) <= set(out)


@SETTINGS
@hypothesis.given(
    terms=st.lists(st.sampled_from(['a', 'b', 'c', 'a:b', 'a:c', 'b:c', 'a:b:c']), min_size=1, max_size=6, unique=True),
    intercept=st.booleans(),
)
def test_term_info_codes(terms: list[str], intercept: bool) -> None:
    """src/metafopy/methods/emm_nesting.py::term_info: codes are 1 or 2, keys are variables, main is the single-variable terms."""
    variables = ['a', 'b', 'c']  # auditme: accept[literal-duplication] three variable names, the same as in other tests' fixtures
    rhs = ' + '.join(terms) + ('' if intercept else ' - 1')
    info = metafopy.methods.emm_nesting.term_info(rhs, variables)
    for entry in info.codes:
        assert set(entry) <= set(variables)
        assert set(entry.values()) <= {1, 2}
    assert info.main == frozenset(t for t in terms if ':' not in t)
    for entry in info.codes:
        if len(entry) == 1:
            (code,) = entry.values()
            assert code == (1 if intercept else 2)


# methods/hettest.py


@SETTINGS
@hypothesis.given(
    tests=st.lists(TEST_NAMES, min_size=1, max_size=7).map(tuple),
    boot=st.integers(min_value=-(10**6), max_value=10**6) | st.floats(min_value=-1e6, max_value=1e6),
)
def test_expand_boot_numeric(tests: tuple[str, ...], boot: float) -> None:
    """src/metafopy/methods/hettest.py::expand_boot: a numeric boot gives all True with max(1, round(boot)) iterations."""
    flags, iterations = metafopy.methods.hettest.expand_boot(typing.cast(int, boot), tests)
    assert flags == [True] * len(tests)
    assert iterations == max(1, round(boot))


@SETTINGS
@hypothesis.given(
    tests=st.lists(TEST_NAMES, min_size=1, max_size=7).map(tuple),
    data=st.data(),
)
def test_expand_boot_flags(tests: tuple[str, ...], data: st.DataObject) -> None:
    """src/metafopy/methods/hettest.py::expand_boot: one flag per test and goodness-of-fit tests are always True."""
    boot = data.draw(st.booleans() | st.lists(st.booleans(), min_size=1, max_size=1) | st.lists(st.booleans(), min_size=len(tests), max_size=len(tests)))
    flags, _ = metafopy.methods.hettest.expand_boot(boot, tests)
    assert len(flags) == len(tests)
    for i, (test, flag) in enumerate(zip(tests, flags, strict=True)):
        if test in metafopy.methods.hettest.GOODNESS_OF_FIT:
            assert flag is True
        else:
            assert flag == (boot if isinstance(boot, bool) else boot[0 if len(boot) == 1 else i])


# methods/matreg.py and methods/predict.py


@SETTINGS
@hypothesis.given(names=unique_words, data=st.data())
def test_match_columns_returns_permutation_indices(names: list[str], data: st.DataObject) -> None:
    """src/metafopy/methods/matreg.py::match_columns: a permutation of the model names gives exactly that permutation's indices."""
    colnames = data.draw(st.permutations(names))
    assert metafopy.methods.matreg.match_columns(list(colnames), names) == [names.index(c) for c in colnames]


@SETTINGS
@hypothesis.given(names=st.lists(small_words, max_size=5, unique=True), cols=st.lists(small_text, max_size=5))
def test_match_columns_agrees_with_match_names(names: list[str], cols: list[str]) -> None:
    """src/metafopy/methods/matreg.py::match_columns: the same result as predict.match_names, or an InputError from both."""
    try:
        expected = metafopy.methods.predict.match_names(cols, names)
    except metafopy.errors.InputError:
        with pytest.raises(metafopy.errors.InputError):
            metafopy.methods.matreg.match_columns(cols, names)
    else:
        assert metafopy.methods.matreg.match_columns(cols, names) == expected


@SETTINGS
@hypothesis.given(choices=st.lists(small_text, max_size=6), name=small_text)
def test_matreg_pmatch(choices: list[str], name: str) -> None:
    """src/metafopy/methods/matreg.py::pmatch: None or an index whose choice starts with name; a member returns its own index."""
    result = metafopy.methods.matreg.pmatch(name, choices)
    if name in choices:
        assert result == choices.index(name)
    elif result is not None:
        assert choices[result].startswith(name)
        assert sum(c.startswith(name) for c in choices) == 1


@SETTINGS
@hypothesis.given(p=st.integers(min_value=1, max_value=20), xs=st.lists(st.integers(min_value=-5, max_value=25), max_size=6))
def test_resolve_x_indices(p: int, xs: list[int]) -> None:
    """src/metafopy/methods/matreg.py::resolve_x: distinct in-range indices come back unchanged, repeated or out-of-range ones raise."""
    if len(set(xs)) != len(xs) or any(not 0 <= v < p for v in xs):
        with pytest.raises(metafopy.errors.InputError):
            metafopy.methods.matreg.resolve_x(xs, None, p)
    else:
        assert metafopy.methods.matreg.resolve_x(xs, None, p) == xs


@SETTINGS
@hypothesis.given(names=unique_words, data=st.data())
def test_resolve_x_names(names: list[str], data: st.DataObject) -> None:
    """src/metafopy/methods/matreg.py::resolve_x: string inputs give positions whose names start with the given strings."""
    picks = data.draw(st.lists(prefixes(names).map(lambda t: t[1]), min_size=1, max_size=4))
    try:
        out = metafopy.methods.matreg.resolve_x(picks, names, len(names))
    except metafopy.errors.InputError:
        assert any(metafopy.methods.matreg.pmatch(v, names) is None for v in picks)
    else:
        assert len(out) == len(picks)
        for v, pos in zip(picks, out, strict=True):
            assert names[pos].startswith(v)
    assert metafopy.methods.matreg.resolve_x(names, names, len(names)) == list(range(len(names)))


@SETTINGS
@hypothesis.given(p=st.integers(min_value=1, max_value=20), y=st.integers(min_value=-5, max_value=25))
def test_resolve_y_index(p: int, y: int) -> None:
    """src/metafopy/methods/matreg.py::resolve_y: an integer in [0, p-1] is returned unchanged, any other integer raises."""
    if 0 <= y <= p - 1:
        assert metafopy.methods.matreg.resolve_y(y, None, p) == y
    else:
        with pytest.raises(metafopy.errors.InputError):
            metafopy.methods.matreg.resolve_y(y, None, p)


@SETTINGS
@hypothesis.given(names=unique_words, y=small_text)
def test_resolve_y_name(names: list[str], y: str) -> None:
    """src/metafopy/methods/matreg.py::resolve_y: a string gives a position whose name starts with it, or raises InputError."""
    try:
        pos = metafopy.methods.matreg.resolve_y(y, names, len(names))
    except metafopy.errors.InputError:
        assert y not in names
    else:
        assert names[pos].startswith(y)


@SETTINGS
@hypothesis.given(name=small_text, target=small_text)
def test_subsequence_distance(name: str, target: str) -> None:
    """src/metafopy/methods/matreg.py::subsequence_distance: finite exactly for a subsequence, then len(target) - len(name); zero on itself."""
    distance = metafopy.methods.matreg.subsequence_distance(name, target)
    if metafopy.methods.predict.is_subsequence(name, target):
        assert distance == len(target) - len(name) >= 0
    else:
        assert distance == float('inf')
    assert metafopy.methods.matreg.subsequence_distance(target, target) == 0


@SETTINGS
@hypothesis.given(short=st.text(alphabet='abc', max_size=4), long=st.text(alphabet='abc', max_size=8))
def test_is_subsequence_agrees_with_brute_force_deletion(short: str, long: str) -> None:
    """src/metafopy/methods/predict.py::is_subsequence: agrees with deleting characters from long, and a short that deletes down to it."""
    expected = any(''.join(kept) == short for kept in itertools.combinations(long, len(short)))
    assert metafopy.methods.predict.is_subsequence(short, long) is expected
    assert metafopy.methods.predict.is_subsequence(long, long)


@SETTINGS
@hypothesis.given(short=small_text, inserts=st.lists(st.tuples(st.integers(min_value=0, max_value=10), st.sampled_from('abcxyz')), max_size=5))
def test_is_subsequence_after_insertions(short: str, inserts: list[tuple[int, str]]) -> None:
    """src/metafopy/methods/predict.py::is_subsequence: true for any string with characters inserted into short."""
    long = short
    for at, ch in inserts:
        long = long[: at % (len(long) + 1)] + ch + long[at % (len(long) + 1) :]
    assert metafopy.methods.predict.is_subsequence(short, long)


@SETTINGS
@hypothesis.given(labels=unique_words, k_new=st.integers(min_value=0, max_value=6), data=st.data())
def test_level_indices_from_labels(labels: list[str], k_new: int, data: st.DataObject) -> None:
    """src/metafopy/methods/predict.py::level_indices: labels map back to their level, and the length is k_new or InputError."""
    levels = data.draw(st.lists(st.sampled_from(labels), min_size=1, max_size=6))
    if len(levels) in (1, k_new):
        out = metafopy.methods.predict.level_indices(levels, labels, k_new, 'level')
        assert len(out) == k_new
        for i, index in enumerate(out):
            assert labels[index] == levels[0 if len(levels) == 1 else i]
    else:
        with pytest.raises(metafopy.errors.InputError):
            metafopy.methods.predict.level_indices(levels, labels, k_new, 'level')
    assert metafopy.methods.predict.level_indices(levels[0], labels, k_new, 'level') == [labels.index(levels[0])] * k_new
    with pytest.raises(metafopy.errors.InputError):
        metafopy.methods.predict.level_indices(['zzz'], labels, 1, 'level')


@SETTINGS
@hypothesis.given(
    labels=unique_words, k_new=st.integers(min_value=0, max_value=6), levels=st.lists(st.integers(min_value=-2, max_value=8), min_size=1, max_size=6)
)
def test_level_indices_from_integers(labels: list[str], k_new: int, levels: list[int]) -> None:
    """src/metafopy/methods/predict.py::level_indices: 0-based integer levels are returned as they are, out of range ones raise."""
    if any(not 0 <= v < len(labels) for v in levels) or len(levels) not in (1, k_new):
        with pytest.raises(metafopy.errors.InputError):
            metafopy.methods.predict.level_indices(levels, labels, k_new, 'level')
    else:
        assert metafopy.methods.predict.level_indices(levels, labels, k_new, 'level') == levels * (k_new if len(levels) == 1 else 1)


@SETTINGS
@hypothesis.given(coef_names=unique_words, data=st.data())
def test_match_names_permutation(coef_names: list[str], data: st.DataObject) -> None:
    """src/metafopy/methods/predict.py::match_names: for a permutation of coef_names, coef_names[pos[i]] == names[i] and pos permutes."""
    names = list(data.draw(st.permutations(coef_names)))
    pos = metafopy.methods.predict.match_names(names, coef_names)
    assert [coef_names[p] for p in pos] == names
    assert sorted(pos) == list(range(len(coef_names)))


@SETTINGS
@hypothesis.given(
    predtype=st.none() | prefixes(list(metafopy.methods.predict.PREDTYPES)).map(lambda t: t[1]),
    pi_type=st.none() | st.text(alphabet='defaultsimpleriley', max_size=8),
    case=st.sampled_from([str.lower, str.upper, str.title]),
)
def test_resolve_predtype(predtype: str | None, pi_type: str | None, case: type[str]) -> None:
    """src/metafopy/methods/predict.py::resolve_predtype: a PREDTYPES member, predtype wins over pi_type, and the result is a fixed point."""
    predtype = None if predtype is None else case(predtype)
    try:
        result = metafopy.methods.predict.resolve_predtype(predtype, pi_type)
    except metafopy.errors.InputError:
        assert predtype is None
        return
    assert result in metafopy.methods.predict.PREDTYPES
    assert metafopy.methods.predict.resolve_predtype(result, None) == result
    if predtype is not None:
        assert metafopy.methods.predict.resolve_predtype(predtype, None) == result


# methods/robust.py, models/*


@SETTINGS
@hypothesis.given(
    vcov=st.none() | st.sampled_from(metafopy.core.clubsandwich.CR_TYPES),
    coef_test=st.none() | st.sampled_from(metafopy.methods.robust.COEF_TESTS),
    conf_test=st.none() | st.sampled_from(metafopy.methods.robust.CONF_TESTS),
    wald_test=st.none() | st.sampled_from(metafopy.core.clubsandwich.WALD_TESTS),
)
def test_check_options_fixed_point(vcov: str | None, coef_test: str | None, conf_test: str | None, wald_test: str | None) -> None:
    """src/metafopy/methods/robust.py::check_options: allowed values give allowed outputs, and feeding the outputs back changes nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = metafopy.methods.robust.check_options(vcov, coef_test, conf_test, wald_test)
        assert out[0] in metafopy.core.clubsandwich.CR_TYPES
        assert out[1] in metafopy.methods.robust.COEF_TESTS
        assert out[2] in metafopy.methods.robust.CONF_TESTS
        assert out[3] in metafopy.core.clubsandwich.WALD_TESTS
        assert metafopy.methods.robust.check_options(*out) == out


@SETTINGS
@hypothesis.given(
    measure=st.sampled_from([*metafopy.models.rma_glmm.SUPPORTED, *metafopy.models.rma_glmm.EXPERIMENTAL]),
    model=st.none() | st.sampled_from(metafopy.models.rma_glmm.MODELS),
    method=st.sampled_from(metafopy.models.rma_glmm.METHODS),
    coding=st.sampled_from([0.5, 1, 0]),
    test=st.sampled_from(['z', 't']),
)
def test_glmm_check_arguments(measure: str, model: str | None, method: str, coding: float, test: str) -> None:
    """src/metafopy/models/rma_glmm.py::check_arguments: the model is a MODELS member and CM.AL never survives for IRR."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            resolved = metafopy.models.rma_glmm.check_arguments(measure, model, method, coding, test)
        except metafopy.errors.InputError:
            assert measure in metafopy.models.rma_glmm.UNCONDITIONAL_ONLY
            assert model in ('CM.EL', 'CM.AL')
            return
    assert resolved in metafopy.models.rma_glmm.MODELS
    assert not (measure == 'IRR' and resolved == 'CM.AL')


@SETTINGS
@hypothesis.given(value=st.booleans() | st.integers(min_value=-5, max_value=8) | st.sampled_from(metafopy.models.rma_mv.RSCALES) | st.text(max_size=5))
def test_resolve_rscale(value: bool | int | str) -> None:
    """src/metafopy/models/rma_mv.py::resolve_rscale: booleans and integers 0 to 3 give a name, other integers raise, a name maps to itself."""
    if (isinstance(value, int) and not isinstance(value, bool) and not 0 <= value <= 3) or (
        isinstance(value, str) and value not in metafopy.models.rma_mv.RSCALES
    ):
        with pytest.raises(metafopy.errors.InputError):
            metafopy.models.rma_mv.resolve_rscale(value)
        return
    result = metafopy.models.rma_mv.resolve_rscale(value)
    assert result in metafopy.models.rma_mv.RSCALES
    assert metafopy.models.rma_mv.resolve_rscale(result) == result
    if isinstance(value, str):
        assert result == value


@SETTINGS
@hypothesis.given(
    optimizer=st.sampled_from(metafopy.models.sel_opt.OPTIMIZERS),
    optmethod=st.sampled_from(metafopy.models.sel_opt.OPTMETHODS),
    type_=st.sampled_from(metafopy.models.sel_spec.TYPE_OPTIONS),
    ncpus=st.integers(min_value=1, max_value=4),
)
def test_choose_optimizer(optimizer: str, optmethod: str, type_: str, ncpus: int) -> None:
    """src/metafopy/models/sel_opt.py::choose_optimizer: a known optimiser, constrained for stepcon, never constrOptim.nl otherwise."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        choice = metafopy.models.sel_opt.choose_optimizer(optimizer, optmethod, type_, ncpus)
    assert choice.optimizer in metafopy.models.sel_opt.OPTIMIZERS
    assert choice.optmethod in metafopy.models.sel_opt.OPTMETHODS
    if type_ == 'stepcon':
        assert choice.optimizer in metafopy.models.sel_opt.CONSTRAINED
    else:
        assert choice.optimizer != 'constrOptim.nl'


@SETTINGS
@hypothesis.given(options=st.lists(small_words, min_size=1, max_size=6, unique=True), value=small_text)
def test_partial_match(options: list[str], value: str) -> None:
    """src/metafopy/models/sel_opt.py::partial_match: an option matches itself, a unique prefix gives its option, and the result is a fixed point."""
    hits = [o for o in options if o.startswith(value)]
    if value in options:
        result = metafopy.models.sel_opt.partial_match(value, options, 'what')
        assert result == value
    elif len(hits) == 1:
        result = metafopy.models.sel_opt.partial_match(value, options, 'what')
        assert result == hits[0]
    else:
        with pytest.raises(metafopy.errors.InputError):
            metafopy.models.sel_opt.partial_match(value, options, 'what')
        return
    assert metafopy.models.sel_opt.partial_match(result, options, 'what') == result


@SETTINGS
@hypothesis.given(data=st.data())
def test_match_type_first_literal_match(data: st.DataObject) -> None:
    """src/metafopy/models/sel_spec.py::match_type: always a TYPE_OPTIONS member, the first option a literal pattern matches."""
    options = metafopy.models.sel_spec.TYPE_OPTIONS
    pattern = data.draw(
        st.sampled_from(options).flatmap(lambda o: st.tuples(st.integers(0, len(o)), st.integers(0, len(o))).map(lambda ij: o[min(ij) : max(ij)]))
    )
    expected = next((o for o in options if pattern in o), None)
    if expected is None:
        with pytest.raises(metafopy.errors.InputError):
            metafopy.models.sel_spec.match_type(re.escape(pattern))
    else:
        assert metafopy.models.sel_spec.match_type(re.escape(pattern)) == expected
        assert metafopy.models.sel_spec.match_type(expected) == next(o for o in options if expected in o)
    text = data.draw(st.text(alphabet='abcdefghnopqrstuwx12', max_size=6))
    try:
        assert metafopy.models.sel_spec.match_type(re.escape(text)) in options
    except metafopy.errors.InputError:
        assert not any(text in o for o in options)
