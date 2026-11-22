"""Property tests for formula parsing and the rewriting of formula and model-call text.

Formulas are built from generated R names (``rstrategies.r_identifiers``), so the parts a test puts in are the parts it expects back.
"""

import importlib
import keyword

import hypothesis
import hypothesis.strategies
import pytest
import rstrategies

import metafopy.errors
import metafopy.formula

matreg = importlib.import_module('metafopy.methods.matreg')
rcalc = importlib.import_module('metafopy.methods.rcalc')
call = importlib.import_module('metafopy.models.call')
mv_factor = importlib.import_module('metafopy.models.mv_factor')
mv_hessian = importlib.import_module('metafopy.models.mv_hessian')
mv_print = importlib.import_module('metafopy.models.mv_print')
rma_mv = importlib.import_module('metafopy.models.rma_mv')
rma_uni = importlib.import_module('metafopy.models.rma_uni')
infl = importlib.import_module('metafopy.plot.infl')

SETTINGS = hypothesis.settings(max_examples=100, deadline=None)
names = rstrategies.r_identifiers()
python_names = names.filter(lambda s: s.isidentifier() and not keyword.iskeyword(s))
unique_names = hypothesis.strategies.lists(names, min_size=1, max_size=5, unique=True)
signs = hypothesis.strategies.sampled_from('+-')
lower_words = hypothesis.strategies.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=20)


def is_subsequence(small: str, big: str) -> bool:
    rest = iter(big)
    return all(ch in rest for ch in small)


# formula.py


@SETTINGS
@hypothesis.given(before=names, calls=hypothesis.strategies.lists(hypothesis.strategies.tuples(names, names, names), min_size=1, max_size=4))
def test_call_spans_find_each_outermost_call(before: str, calls: list[tuple[str, str, str]]) -> None:
    """formula.py::call_spans: each outermost call is one span, and a call nested in an argument adds none."""
    pieces = [f'{fn}({a}, {inner}({b}))' for fn, a, b in calls for inner in [fn]]
    rhs = f'{before} + ' + ' + '.join(pieces)
    spans = metafopy.formula.call_spans(rhs)
    assert [rhs[start:end] for start, end in spans] == pieces


@SETTINGS
@hypothesis.given(parts=hypothesis.strategies.lists(names, min_size=1, max_size=6), op=hypothesis.strategies.sampled_from(['+', '*', '|', '&', '^', '-']))
def test_r_tokens_keep_every_non_blank_character(parts: list[str], op: str) -> None:
    """formula.py::r_tokens: the tokens, joined, are the text without blanks, and every name is one token."""
    text = f' {op} '.join(parts)
    tokens = metafopy.formula.r_tokens(text)
    assert ''.join(tokens) == text.replace(' ', '')
    assert [t for t in tokens if t != op] == parts


@SETTINGS
@hypothesis.given(left=names, right=names, op=hypothesis.strategies.sampled_from(sorted(metafopy.formula.R_SPACED)))
def test_r_deparse_spaces_binary_operators_and_is_idempotent(left: str, right: str, op: str) -> None:
    """formula.py::r_deparse: a spaced operator gets one blank each side, and deparsing its own output changes nothing."""
    text = metafopy.formula.r_deparse([left, op, right])
    assert text == f'{left} {op} {right}'
    assert metafopy.formula.r_deparse(metafopy.formula.r_tokens(text)) == text


@SETTINGS
@hypothesis.given(outside=names, inside=names, fn=names, power=hypothesis.strategies.integers(2, 9))
def test_r_power_rewrites_only_inside_calls(outside: str, inside: str, fn: str, power: int) -> None:
    """formula.py::r_power: ``^`` inside a call becomes ``**``; outside it stays; a formula with no ``^`` is unchanged."""
    rhs = f'{outside}^{power} + {fn}({inside}^{power})'
    assert metafopy.formula.r_power(rhs) == f'{outside}^{power} + {fn}({inside}**{power})'
    plain = f'{outside} + {fn}({inside})'
    assert metafopy.formula.r_power(plain) == plain


@SETTINGS
@hypothesis.given(before=python_names, fn=python_names, left=python_names, right=python_names, after=python_names)
def test_r_logical_calls_and_formula_rewrite_the_call_that_uses_bar(before: str, fn: str, left: str, right: str, after: str) -> None:
    """formula.py::r_logical_calls and r_logical_formula: the call with ``|`` is found, R's text is kept, and a formula without one is unchanged."""
    rhs = f'{before} + {fn}({left} | {right}) + {after}'
    start = len(before) + 3
    end = start + len(f'{fn}({left} | {right})')
    assert metafopy.formula.r_logical_calls(rhs) == [(start, end, f'{fn}(r_or({left}, {right}))', f'{fn}({left} | {right})')]
    assert metafopy.formula.r_logical_formula(rhs) == f'{before} + {fn}(r_or({left}, {right})) + {after}'
    plain = f'{before} + {fn}({left}) + {after}'
    assert metafopy.formula.r_logical_calls(plain) == []
    assert metafopy.formula.r_logical_formula(plain) == plain


@SETTINGS
@hypothesis.given(
    inner=hypothesis.strategies.none() | names,
    outer=hypothesis.strategies.lists(names, min_size=1, max_size=3, unique=True),
)
def test_parse_random_returns_the_parts_it_was_built_from(inner: str | None, outer: list[str]) -> None:
    """formula.py::parse_random: ``~ inner | a/b`` gives back the inner name and the nesting, and prints back to the same formula."""
    hypothesis.assume(inner is None or len(outer) == 1)
    lhs = '1' if inner is None else inner
    formula = f'~ {lhs} | {"/".join(outer)}'
    term = metafopy.formula.parse_random(formula)
    assert (term.inner, term.outer, term.nested, term.double_bar) == (inner, tuple(outer), len(outer) > 1, False)
    assert str(term) == formula
    assert metafopy.formula.parse_random(str(term)) == term


@SETTINGS
@hypothesis.given(text=hypothesis.strategies.text(alphabet='ab |/', max_size=10))
def test_parse_random_demands_a_leading_tilde(text: str) -> None:
    """formula.py::parse_random: text that does not start with ``~`` is an input error."""
    hypothesis.assume(not text.strip().startswith('~'))
    with pytest.raises(metafopy.errors.InputError):
        metafopy.formula.parse_random(text)


# methods/matreg.py, methods/rcalc.py


@SETTINGS
@hypothesis.given(y=names, xs=hypothesis.strategies.lists(names, min_size=1, max_size=6))
def test_matreg_parse_formula_gives_outcome_and_unique_predictors_in_order(y: str, xs: list[str]) -> None:
    """matreg.py::parse_formula: the outcome, then each predictor once in order of first appearance."""
    assert matreg.parse_formula(f'{y} ~ {" + ".join(xs)}') == (y, list(dict.fromkeys(xs)))


@SETTINGS
@hypothesis.given(rhs=names)
def test_matreg_parse_formula_needs_a_left_hand_side(rhs: str) -> None:
    """matreg.py::parse_formula: no outcome is an input error."""
    with pytest.raises(metafopy.errors.InputError):
        matreg.parse_formula(f'~ {rhs}')


@SETTINGS
@hypothesis.given(y=names, xs=hypothesis.strategies.lists(names, min_size=1, max_size=5), fns=hypothesis.strategies.lists(names, max_size=3))
def test_formula_variables_skip_function_names_and_duplicates(y: str, xs: list[str], fns: list[str]) -> None:
    """rcalc.py::formula_variables: variables in order of appearance, once each, and no function name."""
    terms = [f'{fn}({x})' for fn, x in zip(fns, xs, strict=False)] + xs
    expected = list(dict.fromkeys([y, *[x for x in xs[: len(fns)]], *xs]))
    assert rcalc.formula_variables(f'{y} ~ {" + ".join(terms)}') == expected


@SETTINGS
@hypothesis.given(text=hypothesis.strategies.text(alphabet='ab +', max_size=10))
def test_formula_variables_demand_a_tilde(text: str) -> None:
    """rcalc.py::formula_variables: text without ``~`` is an input error."""
    with pytest.raises(metafopy.errors.InputError):
        rcalc.formula_variables(text)


@SETTINGS
@hypothesis.given(choice=hypothesis.strategies.sampled_from(rcalc.NFUNS), keep=hypothesis.strategies.integers(1, 8))
def test_match_nfun_takes_unambiguous_prefixes_only(choice: str, keep: int) -> None:
    """rcalc.py::match_nfun: a prefix that starts exactly one choice gives it; an ambiguous prefix is an input error."""
    prefix = choice[:keep]
    hits = [c for c in rcalc.NFUNS if c.startswith(prefix)]
    if prefix in rcalc.NFUNS or len(hits) == 1:
        assert rcalc.match_nfun(prefix) == choice
    else:
        with pytest.raises(metafopy.errors.InputError):
            rcalc.match_nfun(prefix)


# models/call.py


@SETTINGS
@hypothesis.given(terms=hypothesis.strategies.lists(hypothesis.strategies.tuples(signs, names), min_size=1, max_size=6), wrap=names)
def test_split_terms_splits_at_top_level_signs_only(terms: list[tuple[str, str]], wrap: str) -> None:
    """call.py::split_terms: the sign and text of each term come back; a sign inside parentheses does not split."""
    first = [('+', terms[0][1]), *terms[1:]]
    text = ''.join(f' {s} {t}' for s, t in first)
    assert call.split_terms(text) == first
    nested = f'{wrap}({terms[0][1]} - {terms[-1][1]})'
    assert call.split_terms(nested) == [('+', nested)]


@SETTINGS
@hypothesis.given(old=unique_names)
def test_update_side_dot_is_idempotent(old: list[str]) -> None:
    """call.py::update_side: ``.`` keeps the terms, and applying it again changes nothing."""
    text = ' + '.join(old)
    once = call.update_side(text, '.')
    assert once == text
    assert call.update_side(once, '.') == once


@SETTINGS
@hypothesis.given(old=unique_names, new=names)
def test_update_side_adds_then_drops_a_term(old: list[str], new: str) -> None:
    """call.py::update_side: ``. + t`` appends a new term, and ``. - t`` then removes it again."""
    hypothesis.assume(new not in old)
    text = ' + '.join(old)
    added = call.update_side(text, f'. + {new}')
    assert added == f'{text} + {new}'
    assert call.update_side(added, f'. - {new}') == text


@SETTINGS
@hypothesis.given(y=names, old=unique_names, new=names)
def test_update_formula_keeps_the_outcome_and_updates_the_terms(y: str, old: list[str], new: str) -> None:
    """call.py::update_formula: ``. ~ . + t`` keeps the left-hand side and the old terms, and is idempotent."""
    hypothesis.assume(new not in old)
    formula = f'{y} ~ {" + ".join(old)}'
    updated = call.update_formula(formula, f'. ~ . + {new}')
    assert updated == f'{formula} + {new}'
    assert call.update_formula(updated, f'. ~ . + {new}') == updated


@SETTINGS
@hypothesis.given(text=hypothesis.strategies.text(alphabet='ab +.', max_size=10))
def test_update_formula_demands_a_tilde_in_the_update(text: str) -> None:
    """call.py::update_formula: an update without ``~`` is an input error."""
    with pytest.raises(metafopy.errors.InputError):
        call.update_formula('y ~ x', text)


# models/mv_factor.py


@SETTINGS
@hypothesis.given(
    pieces=hypothesis.strategies.lists(
        hypothesis.strategies.one_of(
            names,
            hypothesis.strategies.tuples(names, names, names).map(lambda t: f'{t[0]}({t[1]}, {t[2]})'),
            hypothesis.strategies.text(alphabet='ab,', min_size=1, max_size=4).map(lambda s: f'"{s}"'),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_split_top_level_ignores_separators_in_brackets_and_quotes(pieces: list[str]) -> None:
    """mv_factor.py::split_top_level: pieces joined with the separator come back whole, even with commas inside calls or quotes."""
    assert mv_factor.split_top_level(', '.join(pieces), ',') == pieces


# models/mv_hessian.py


@SETTINGS
@hypothesis.given(prefix=names, nlevels=hypothesis.strategies.integers(2, 8))
def test_inner_names_number_each_lower_triangle_pair_once(prefix: str, nlevels: int) -> None:
    """mv_hessian.py::inner_names: one unique name per correlation of an ``n x n`` matrix, in column-major lower-triangle order."""
    count = nlevels * (nlevels - 1) // 2
    out = mv_hessian.inner_names(prefix, count)
    assert len(out) == len(set(out)) == count
    if count == 1:
        assert out == [prefix]
    else:
        assert out == [f'{prefix}.{i}.{j}' for j in range(1, nlevels + 1) for i in range(j + 1, nlevels + 1)]


@SETTINGS
@hypothesis.given(prefix=names, count=hypothesis.strategies.integers(1, 12))
def test_numbered_gives_the_prefix_alone_or_one_unique_name_each(prefix: str, count: int) -> None:
    """mv_hessian.py::numbered: one count gives the bare prefix; more give ``prefix.1`` to ``prefix.count``."""
    out = mv_hessian.numbered(prefix, count)
    assert out == ([prefix] if count == 1 else [f'{prefix}.{i}' for i in range(1, count + 1)])


@SETTINGS
@hypothesis.given(n=hypothesis.strategies.integers(1, 12))
def test_lower_triangle_order_is_a_permutation_in_column_major_order(n: int) -> None:
    """mv_hessian.py::lower_triangle_order: variances sit at ``0..n-1``, covariances after them in column-major order; every position once."""
    expected: list[int] = []
    cov = n
    for j in range(n):
        for i in range(j, n):
            if i == j:
                expected.append(j)
            else:
                expected.append(cov)
                cov += 1
    # covariances are numbered in the order of the strictly lower triangle, which is the order this loop meets them
    assert mv_hessian.lower_triangle_order(n) == expected
    assert sorted(expected) == list(range(n * (n + 1) // 2))


# models/mv_print.py, models/rma_uni.py, plot/infl.py


@SETTINGS
@hypothesis.given(
    text=hypothesis.strategies.text(alphabet='abcdefghijklmnopqrstuvwxyzABC', min_size=1, max_size=20), minlen=hypothesis.strategies.integers(1, 8)
)
def test_abbreviate_once_keeps_the_first_letter_and_a_subsequence(text: str, minlen: int) -> None:
    """mv_print.py::abbreviate_once: a short string is kept; a long one loses characters but never its first, and no longer than before."""
    out = mv_print.abbreviate_once(text, minlen)
    if len(text) <= minlen:
        assert out == text
    assert out[0] == text[0]
    assert is_subsequence(out, text)
    assert len(out) <= len(text)


@SETTINGS
@hypothesis.given(words=hypothesis.strategies.lists(lower_words, min_size=1, max_size=6, unique=True), minlen=hypothesis.strategies.integers(1, 6))
def test_abbreviate_makes_unique_names_unique(words: list[str], minlen: int) -> None:
    """mv_print.py::abbreviate: unique names give unique abbreviations, each a subsequence of its name."""
    out = mv_print.abbreviate(words, minlen)
    assert len(out) == len(words)
    assert len(set(out)) == len(out)
    assert all(is_subsequence(a, w) for a, w in zip(out, words, strict=True))


@SETTINGS
@hypothesis.given(x=names, ref=hypothesis.strategies.text(alphabet='abcXYZ019', max_size=5), degree=hypothesis.strategies.integers(1, 9))
def test_abbreviate_names_strips_the_wrappers_of_coefficient_names(x: str, ref: str, degree: int) -> None:
    """rma_uni.py::abbreviate_names: ``factor()``, ``I()``, ``relevel()`` and ``poly()`` wrappers go; a plain name stays."""
    hypothesis.assume(')' not in x)
    assert rma_uni.abbreviate_names([x]) == [x]
    assert rma_uni.abbreviate_names([f'factor({x})b']) == [f'{x}b']
    assert rma_uni.abbreviate_names([f'I({x})']) == [x]
    assert rma_uni.abbreviate_names([f'relevel(factor({x}), ref = "{ref}")b']) == [f'{x}b']
    assert rma_uni.abbreviate_names([f'poly({x}, degree = {degree})2']) == [f'{x}^2']
    out = rma_uni.abbreviate_names([f'factor({x})b'])
    assert rma_uni.abbreviate_names(out) == out


@SETTINGS
@hypothesis.given(a=names, b=names, fn=hypothesis.strategies.sampled_from(['interaction', 'paste', 'paste0']))
def test_replace_call_notation_writes_interactions_with_a_colon(a: str, b: str, fn: str) -> None:
    """rma_mv.py::replace_call_notation: ``interaction(a, b)``, ``paste(a, b)`` and ``paste0(a, b)`` become ``a:b``; a plain name stays."""
    out = rma_mv.replace_call_notation(f'{fn}({a}, {b})')
    assert out == f'{a}:{b}'
    assert rma_mv.replace_call_notation(out) == out
    assert rma_mv.replace_call_notation(a) == a


@SETTINGS
@hypothesis.given(text=lower_words, minlen=hypothesis.strategies.integers(1, 8))
def test_infl_abbreviate_shortens_a_lower_case_word_to_minlength(text: str, minlen: int) -> None:
    """infl.py::abbreviate: a lower-case word of at most ``minlen`` letters is kept; a longer one is cut to ``minlen`` and keeps its first letter."""
    out = infl.abbreviate(text, minlen)
    assert out[0] == text[0]
    assert is_subsequence(out, text)
    assert out == text if len(text) <= minlen else len(out) == minlen
