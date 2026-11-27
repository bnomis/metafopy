"""The comparison rules of the audit harness (``src/audit``): value trees, printed output, tolerances and the recall matcher."""

import json
import pathlib
import re
import subprocess

import hypothesis
import hypothesis.strategies
import pytest

import audit.cases
import audit.cli
import audit.config
import audit.domains
import audit.errors
import audit.generate
import audit.printdiff
import audit.rdriver
import audit.recall
import audit.report
import audit.rlib
import audit.steps
import audit.universe
import audit.valuetree
import metafopy.core.dbinom


def vector(values: list[object], kind: str = 'num') -> audit.valuetree.Node:
    return {'k': kind, 'v': values, 'd': None}


def diffs(r: audit.valuetree.Node, py: audit.valuetree.Node) -> list[audit.valuetree.Diff]:
    out: list[audit.valuetree.Diff] = []
    audit.valuetree.compare(r, py, '', out)
    return out


def test_equal_vectors_have_no_difference() -> None:
    assert diffs(vector([1.0, 2.0]), vector([1.0, 2.0 + 1e-12])) == []


def test_value_difference_beyond_drift_is_listed() -> None:
    (d,) = diffs(vector([1.0, 2.0]), vector([1.0, 2.001]))
    assert d.kind == 'value-diff'
    assert 'element 1' in d.detail


def test_small_difference_is_drift_not_a_finding() -> None:
    (d,) = diffs(vector([1.0]), vector([1.0 + 5e-6]))
    assert d.kind == 'value-drift'


def test_length_difference_is_a_shape_difference() -> None:
    (d,) = diffs(vector([1.0, 2.0, 3.0]), vector([1.0, 2.0]))
    assert d.kind == 'shape-diff'


def test_one_based_indices_equal_zero_based_ones() -> None:
    assert diffs(vector([1, 2, 3]), vector([0, 1, 2])) == []


def test_not_every_difference_of_one_is_an_index() -> None:
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    assert [d.kind for d in diffs(vector([1.5, 2.5]), vector([0.5, 1.5]))] == ['value-diff']


def test_missing_values_match_each_other_only() -> None:
    assert diffs(vector([None, 1.0]), vector([None, 1.0])) == []
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    assert [d.kind for d in diffs(vector([None]), vector([1.0]))] == ['value-diff']


def test_missing_field_is_reported_by_to_node() -> None:
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    guide: audit.valuetree.Node = {'k': 'list', 'items': [{'name': 'a', 'node': vector([1.0])}, {'name': 'b', 'node': vector([2.0])}], 'cls': ['list']}
    missing: list[str] = []
    audit.valuetree.to_node({'a': 1.0}, guide, '', missing)
    assert missing == ['$b']


def test_python_names_find_r_names() -> None:
    assert audit.valuetree.py_name('ci.lb') == 'ci_lb'
    found, value = audit.valuetree.lookup({'ci_lb': 3}, 'ci.lb')
    assert found
    assert value == 3


def test_printed_output_tolerates_the_last_digit() -> None:
    assert audit.printdiff.first_difference('pred  0.1234  0.5678\n', 'pred 0.1235 0.5678\n') is None
    assert audit.printdiff.first_difference('pred  0.1234\n', 'pred 0.1334\n') is not None


def test_printed_output_notices_extra_columns() -> None:
    assert audit.printdiff.first_difference('pred se\n', 'pred\n') is not None


def test_recall_item_matches_by_globs() -> None:
    item = audit.recall.Item('x', 4, 'text', [{'id': 'options/vcov/*/type=*', 'kind': 'crash'}], [], '', '')
    assert item.matches({'id': 'options/vcov/uni_reml/type="obs"', 'kind': 'crash', 'path': '', 'detail': ''})
    assert not item.matches({'id': 'options/vcov/uni_reml/type="obs"', 'kind': 'value-diff', 'path': '', 'detail': ''})
    assert not item.matches({'id': 'options/predict/uni_reml/type="obs"', 'kind': 'crash', 'path': '', 'detail': ''})


def write_recall(tmp_path: pathlib.Path, hit: bool) -> tuple[str, bool]:
    corpus = audit.recall.Corpus(
        {6: {}},
        [
            audit.recall.Item('a', 6, 'plain', [], [], '', ''),
            audit.recall.Item('b', 6, 'moved text', [], [], '', '', '2026-11-02, R 4.6.2'),
        ],
    )
    path, ok = audit.recall.write(tmp_path, corpus, {6: {'a': True, 'b': hit}}, {6: 'abc1234'})
    return path.read_text(), ok


def test_recall_oracle_moved_item_is_in_neither_count(tmp_path: pathlib.Path) -> None:
    text, ok = write_recall(tmp_path, hit=False)
    assert ok
    assert '1 of 1 items reproduced; 1 not shown by the pinned R.' in text
    assert '## Audit 6 (metafopy `abc1234`): 1 of 1 reproduced' in text
    assert '## Oracle moved (1)' in text
    assert '- `b`: moved text; 2026-11-02, R 4.6.2' in text
    assert 'missed' not in text


def test_recall_oracle_moved_item_that_matches_is_reproduced_again(tmp_path: pathlib.Path) -> None:
    text, ok = write_recall(tmp_path, hit=True)
    assert ok
    assert '- `b`: moved text (reproduced again); 2026-11-02, R 4.6.2' in text


def test_recall_hints_select_script_topics_and_cases() -> None:
    items = [
        audit.recall.Item('a', 5, '', [{'id': 'scripts/rma.mh/*'}], [], '', ''),
        audit.recall.Item('b', 5, '', [{'id': 'options/vcov/*/type=*'}], [], '', ''),
        audit.recall.Item('c', 5, '', [{'id': 'matrix/print/mv_ks'}], ['matrix/print/'], '', ''),
    ]
    assert audit.recall.hints(items) == ['matrix/print/', 'options/vcov/*/type=*', 'rma.mh']
    assert audit.recall.steps_for(items) == ['matrix', 'options', 'scripts']


def test_different_names_are_a_label_difference() -> None:
    r = {**vector([1.0, 2.0]), 'n': ['intrcpt', 'I(1/ablat)']}
    py = {**vector([1.0, 2.0]), 'n': ['intrcpt', 'I(1 / ablat)']}
    (d,) = diffs(r, py)
    assert d.kind == 'label-diff'


def test_matrix_without_dimnames_where_r_has_them_is_a_label_difference() -> None:
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    r = {'k': 'num', 'v': [1.0] * 4, 'd': [2, 2], 'rn': ['a', 'b'], 'cn': ['a', 'b']}
    py = {'k': 'num', 'v': [1.0] * 4, 'd': [2, 2]}
    assert {d.kind for d in diffs(r, py)} == {'label-diff'}


def test_positions_as_row_names_are_not_labels() -> None:
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    r = {'k': 'num', 'v': [1.0] * 4, 'd': [2, 2], 'rn': ['1', '2'], 'cn': None}
    py = {'k': 'num', 'v': [1.0] * 4, 'd': [2, 2]}
    assert diffs(r, py) == []


def test_printed_zero_must_have_the_same_decimals() -> None:
    assert audit.printdiff.first_difference('tau^2: 0 (SE = 0.0061)', 'tau^2: 0.0000 (SE = 0.0061)') is not None
    assert audit.printdiff.first_difference('tau^2: 0.1446', 'tau^2: 0.1447') is None


def test_every_new_finding_is_rendered_in_the_report(tmp_path: pathlib.Path) -> None:
    findings = [
        audit.steps.Finding('matrix', 'matrix/x/y', kind, 'p', 'd')
        for kind in ('default-diff', 'value-diff', 'a-kind-nobody-listed', 'memory', 'process-died', 'r-killed')
    ]
    ctx = audit.steps.Context(
        work=tmp_path, jobs=1, timeout=0.0, quick=False, only=None, no_ledger=True, universe=audit.universe.Universe.load(audit.universe.snapshot_path())
    )
    outcome = audit.steps.Outcome()
    outcome.findings.extend(findings)
    text = audit.report.write(ctx, {'matrix': outcome}, tmp_path).read_text()
    assert 'Different defaults (1)' in text
    assert 'Other (1)' in text
    assert 'Wrong values (1)' in text
    assert 'Crashes (2)' in text
    assert 'R processes killed (1)' in text


def test_digits_number_stands_for_the_vector_of_nine() -> None:
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    guide: audit.valuetree.Node = {'k': 'list', 'items': [{'name': 'digits', 'node': vector([4.0] * 9)}], 'cls': ['list']}
    missing: list[str] = []
    py = audit.valuetree.to_node({'digits': 4}, guide, '', missing)
    out: list[audit.valuetree.Diff] = []
    audit.valuetree.compare(guide, py, '', out)
    assert out == []
    other = audit.valuetree.to_node({'digits': 2}, guide, '', missing)
    audit.valuetree.compare(guide, other, '', out)
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    assert [d.kind for d in out] == ['value-diff']


def test_printed_lines_ignore_curly_quotes_and_trailing_spaces() -> None:
    assert audit.valuetree.cell_equal("Signif. codes:  0 '***' 0.001  ", 'Signif. codes:  0 \u2018***\u2019 0.001')[0] == 'same'
    assert audit.valuetree.cell_equal("a 'b'", "a 'c'")[0] == 'diff'


def test_printed_lines_ignore_blank_lines() -> None:
    r = vector(['', 'QM = 1', ''], 'str')
    py = vector(['QM = 1'], 'str')
    assert diffs(r, py) == []
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
    assert [d.kind for d in diffs(vector(['a', 'b'], 'str'), vector(['a'], 'str'))] == ['shape-diff']


def test_profile_positions_of_mv_fits_are_zero_based() -> None:
    assert audit.rlib.shift_arguments({'tau2': 1}, True) == {'tau2': 0}
    assert audit.rlib.shift_arguments({'tau2': 1}, False) == {'tau2': 1}
    assert audit.rlib.shift_arguments({'include': [1, 3]}) == {'include': [0, 2]}


def test_library_drift_of_equal_libraries_is_empty() -> None:
    assert audit.steps.library_drift({'R': '4.6.2', 'lme4': '2.1-0'}, {'R': '4.6.2', 'lme4': '2.1-0'}) == {}


def test_library_drift_lists_changed_missing_and_extra_packages() -> None:
    lock = {'R': '4.6.2', 'lme4': '2.1-0', 'gone': '1.0'}
    installed = {'R': '4.6.2', 'lme4': '2.0.6', 'extra': '3.0'}
    assert audit.steps.library_drift(lock, installed) == {'extra': (None, '3.0'), 'gone': ('1.0', None), 'lme4': ('2.1-0', '2.0.6')}


def pin_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, pinned: dict[str, str] | None, installed: dict[str, str]) -> audit.steps.Context:
    data = tmp_path / 'data'
    data.mkdir()
    if pinned is not None:
        (data / 'r-versions.json').write_text(json.dumps(pinned))
    monkeypatch.setattr(audit.config, 'DATA_DIR', data)

    def fake_run(script: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
        pathlib.Path(args[0]).write_text(json.dumps(installed))
        return subprocess.CompletedProcess([], 0, '', '')

    monkeypatch.setattr(audit.rdriver, 'run', fake_run)
    work = tmp_path / 'work'
    work.mkdir()
    return audit.steps.Context(work=work, jobs=1, timeout=1.0, quick=False, only=None)


def test_check_r_without_a_pin_raises_even_with_the_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    installed = {'R': '4.6.2', 'metafor': audit.config.METAFOR_VERSION}
    ctx = pin_environment(monkeypatch, tmp_path, None, installed)
    ctx.allow_r_drift = True
    with pytest.raises(audit.errors.HarnessError, match='pin the R library first'):
        audit.steps.check_r(ctx)


def test_check_r_refuses_drift_naming_the_package_and_the_ways_on(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    pinned = {'R': '4.6.2', 'lme4': '2.1-0', 'metafor': audit.config.METAFOR_VERSION}
    ctx = pin_environment(monkeypatch, tmp_path, pinned, {**pinned, 'lme4': '2.0.6'})
    with pytest.raises(audit.errors.HarnessError, match=r'lme4 2\.1-0 -> 2\.0\.6') as info:
        audit.steps.check_r(ctx)
    message = str(info.value)
    assert 'setup.R' in message
    assert 're-pin' in message
    assert '--allow-r-drift' in message


def test_check_r_with_the_flag_turns_drift_into_a_note(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    pinned = {'R': '4.6.2', 'lme4': '2.1-0', 'metafor': audit.config.METAFOR_VERSION}
    ctx = pin_environment(monkeypatch, tmp_path, pinned, {**pinned, 'lme4': '2.0.6'})
    ctx.allow_r_drift = True
    outcome = audit.steps.check_r(ctx)
    assert outcome.notes[0] == 'R library differs from the pin (--allow-r-drift): lme4 2.1-0 -> 2.0.6'


def test_check_r_passes_on_the_pinned_library(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    pinned = {'R': '4.6.2', 'metafor': audit.config.METAFOR_VERSION}
    ctx = pin_environment(monkeypatch, tmp_path, pinned, pinned)
    assert audit.steps.check_r(ctx).notes == [f'R 4.6.2, metafor {audit.config.METAFOR_VERSION}; 2 packages']


def test_library_drift_names_a_different_r_platform_or_arithmetic() -> None:
    lock = {'R': '4.6.2', 'R_fma': 'fused', 'R_platform': 'aarch64-apple-darwin27.0.0'}
    installed = {'R': '4.6.2', 'R_fma': 'unfused', 'R_platform': 'x86_64-pc-linux-gnu'}
    assert audit.steps.library_drift(lock, installed) == {
        'R_fma': ('fused', 'unfused'),
        'R_platform': ('aarch64-apple-darwin27.0.0', 'x86_64-pc-linux-gnu'),
    }


def test_check_r_names_a_different_r_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    pinned = {'R': '4.6.2', 'R_platform': 'aarch64-apple-darwin27.0.0', 'metafor': audit.config.METAFOR_VERSION}
    ctx = pin_environment(monkeypatch, tmp_path, pinned, {**pinned, 'R_platform': 'x86_64-pc-linux-gnu'})
    with pytest.raises(audit.errors.HarnessError, match=r'R_platform aarch64-apple-darwin27\.0\.0 -> x86_64-pc-linux-gnu'):
        audit.steps.check_r(ctx)


@pytest.mark.parametrize('fused', [False, True])
def test_check_r_refuses_a_pinned_arithmetic_that_disagrees_with_python(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, fused: bool) -> None:
    pinned = {'R': '4.6.2', 'R_fma': 'unfused' if fused else 'fused', 'metafor': audit.config.METAFOR_VERSION}
    ctx = pin_environment(monkeypatch, tmp_path, pinned, pinned)
    monkeypatch.setattr(metafopy.core.dbinom, 'FUSED', fused)
    with pytest.raises(audit.errors.HarnessError, match='R_fma') as info:
        audit.steps.check_r(ctx)
    message = str(info.value)
    assert f'FUSED is {"fused" if fused else "unfused"}' in message
    assert 're-pin' in message
    assert '--allow-r-drift' in message


def test_check_r_with_the_flag_turns_an_arithmetic_disagreement_into_a_note(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    pinned = {'R': '4.6.2', 'R_fma': 'fused', 'metafor': audit.config.METAFOR_VERSION}
    ctx = pin_environment(monkeypatch, tmp_path, pinned, pinned)
    monkeypatch.setattr(metafopy.core.dbinom, 'FUSED', False)
    ctx.allow_r_drift = True
    assert 'R_fma is fused' in audit.steps.check_r(ctx).notes[0]


def test_check_r_skips_the_arithmetic_comparison_for_a_source_without_fused(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    pinned = {'R': '4.6.2', 'R_fma': 'fused', 'metafor': audit.config.METAFOR_VERSION}
    ctx = pin_environment(monkeypatch, tmp_path, pinned, pinned)
    monkeypatch.delattr(metafopy.core.dbinom, 'FUSED')
    outcome = audit.steps.check_r(ctx)
    assert 'predates metafopy.core.dbinom.FUSED' in outcome.notes[0]


def finding(step: str, case_id: str) -> audit.steps.Finding:
    return audit.steps.Finding(step, case_id, 'missing-field', '$x', 'detail')


def test_result_class_of_a_matrix_or_options_case_is_the_fit_class() -> None:
    """report.py::result_class: the third id part, without its ``#n`` repeat suffix, names the fit; its class is looked up."""
    classes = {'uni_reml': 'rma.uni'}
    assert audit.report.result_class(finding('matrix', 'matrix/confint/uni_reml#2'), classes) == 'rma.uni'
    assert audit.report.result_class(finding('options', 'options/rma/uni_reml'), classes) == 'rma.uni'


def test_result_class_falls_back_to_the_function_in_the_id() -> None:
    """report.py::result_class: a fit with no known class gives the function of the case, as does an id of two parts."""
    assert audit.report.result_class(finding('matrix', 'matrix/confint/other'), {}) == 'confint'
    assert audit.report.result_class(finding('matrix', 'matrix/confint'), {}) == 'confint'


def test_result_class_of_a_construct_case_is_the_fixture_class() -> None:
    """report.py::result_class: a construct case names the fixture in its second id part."""
    assert audit.report.result_class(finding('construct', 'construct/es_bcg'), {'es_bcg': 'escalc'}) == 'escalc'
    assert audit.report.result_class(finding('construct', 'construct/other'), {}) == 'other'


def test_result_class_of_a_script_finding_names_the_examples() -> None:
    """report.py::result_class: a scripts finding is about the results of the metafor examples and tests, whatever its id."""
    assert audit.report.result_class(finding('scripts', 'scripts/a/b'), {'b': 'x'}) == 'results of the metafor examples and tests'


def test_result_class_of_another_step_is_the_function() -> None:
    """report.py::result_class: any other step gives the function of the case."""
    assert audit.report.result_class(finding('print', 'print/confint/fit'), {}) == 'confint'


def test_generate_cases_generates_once_and_records_the_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """steps.py::generate_cases: the first call generates and stores what is uncovered; the second returns the stored result untouched."""
    calls: list[int] = []
    generated = audit.generate.Generated(cases=[], uncovered=['f/arg'], unaccounted=['g'])

    def fake_generate(*args: object) -> audit.generate.Generated:
        calls.append(1)
        return generated

    monkeypatch.setattr(audit.generate, 'generate', fake_generate)
    monkeypatch.setattr(audit.cases, 'load_fixtures', list)
    monkeypatch.setattr(audit.domains.Domains, 'load', classmethod(lambda cls: object()))
    monkeypatch.setattr(audit.domains.Recipes, 'load', classmethod(lambda cls: object()))
    universe = audit.universe.Universe({'fixtures': {}, 'functions': {}, 'methods': []})
    ctx = audit.steps.Context(work=tmp_path, jobs=1, timeout=1.0, quick=False, only=None, universe=universe)
    assert audit.steps.generate_cases(ctx) is generated
    assert ctx.coverage == {'uncovered_arguments': ['f/arg'], 'unaccounted': ['g']}
    assert audit.steps.generate_cases(ctx) is generated
    assert calls == [1]


@pytest.mark.parametrize('value', ['abc', '999', '1,x'])
def test_recall_value_that_names_no_audit_is_a_usage_error(value: str) -> None:
    with pytest.raises(SystemExit) as raised:
        audit.cli.parse_args(['--recall', value])
    assert raised.value.code == 2


@pytest.mark.parametrize('value', ['0', '-1', '-0.5'])
def test_timeout_that_is_not_positive_is_refused(value: str) -> None:
    with pytest.raises(audit.errors.HarnessError, match=re.escape(value)):
        audit.cli.parse_args(['--timeout', value])


PRINTED_DIGITS = hypothesis.strategies.integers(min_value=0, max_value=6)
PRINTED_NUMBERS = hypothesis.strategies.builds(
    lambda x, n, exponent: f'{x:.{n}f}{exponent}',
    hypothesis.strategies.floats(min_value=-1e6, max_value=1e6),
    PRINTED_DIGITS,
    hypothesis.strategies.sampled_from(['', 'e5', 'E-3', 'e+10']),
)
# a printed number, or a token with numbers inside it (``QM(df = 2) = 15.6105,`` is ``15.6105,`` after the split)
NUMERIC_TOKENS = hypothesis.strategies.one_of(
    PRINTED_NUMBERS,
    hypothesis.strategies.builds(
        lambda prefix, number, suffix: prefix + number + suffix,
        hypothesis.strategies.sampled_from(['', '(', 'tau^2=']),
        PRINTED_NUMBERS,
        hypothesis.strategies.sampled_from(['', ',', ')', '%']),
    ),
)
WORDS = hypothesis.strategies.text(alphabet='abcXYZ', min_size=1, max_size=6)
PRINTED_LINE_TOKENS = hypothesis.strategies.lists(hypothesis.strategies.one_of(WORDS, NUMERIC_TOKENS), min_size=1, max_size=6)
PRINTED_TEXT_LINES = hypothesis.strategies.lists(
    hypothesis.strategies.text(alphabet='abc 0.1234-', min_size=1, max_size=15).filter(lambda line: line.strip() != ''),
    max_size=5,
)
BLANKS = hypothesis.strategies.sampled_from(['', ' ', '\t', '   '])


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(x=hypothesis.strategies.floats(min_value=-1e300, max_value=1e300), n=hypothesis.strategies.integers(min_value=0, max_value=15))
def test_decimals_counts_the_printed_places_and_ignores_an_exponent(x: float, n: int) -> None:
    """``decimals`` (T2): ``decimals(f'{x:.{n}f}') == n``, and an exponent suffix does not change it."""
    printed = f'{x:.{n}f}'
    assert audit.printdiff.decimals(printed) == n
    assert audit.printdiff.decimals(printed + 'e-5') == n
    assert audit.printdiff.decimals(printed + 'E+7') == n


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(a=PRINTED_NUMBERS, b=PRINTED_NUMBERS)
def test_close_numbers_is_reflexive_and_symmetric(a: str, b: str) -> None:
    """``close_numbers`` (T2): a printed number is close to itself, and ``close_numbers(a, b) == close_numbers(b, a)``."""
    assert audit.printdiff.close_numbers(a, a)
    assert audit.printdiff.close_numbers(a, b) == audit.printdiff.close_numbers(b, a)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(k=hypothesis.strategies.integers(min_value=-(10**9), max_value=10**9), n=PRINTED_DIGITS)
def test_close_numbers_accepts_a_difference_of_one_in_the_last_place(k: int, n: int) -> None:
    """``close_numbers`` (T2): the printed ``k / 10**n`` and ``(k + 1) / 10**n`` are close, in either order."""
    a, b = f'{k / 10**n:.{n}f}', f'{(k + 1) / 10**n:.{n}f}'
    assert audit.printdiff.close_numbers(a, b)
    assert audit.printdiff.close_numbers(b, a)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(a=NUMERIC_TOKENS, b=NUMERIC_TOKENS)
def test_same_token_is_reflexive_and_symmetric(a: str, b: str) -> None:
    """``same_token`` (T2): reflexive and symmetric over numeric tokens and tokens with embedded numbers."""
    assert audit.printdiff.same_token(a, a)
    assert audit.printdiff.same_token(a, b) == audit.printdiff.same_token(b, a)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(a=PRINTED_LINE_TOKENS, b=PRINTED_LINE_TOKENS, space=hypothesis.strategies.sampled_from([' ', '  ', '\t', ' \t ']))
def test_same_line_is_reflexive_symmetric_and_ignores_the_spacing(a: list[str], b: list[str], space: str) -> None:
    """``same_line`` (T2): reflexive, symmetric, and re-spacing the whitespace between tokens does not change it."""
    line_a, line_b = ' '.join(a), ' '.join(b)
    assert audit.printdiff.same_line(line_a, line_a)
    assert audit.printdiff.same_line(line_a, line_b) == audit.printdiff.same_line(line_b, line_a)
    assert audit.printdiff.same_line(space.join(a), line_a)
    assert audit.printdiff.same_line(space.join(a), line_b) == audit.printdiff.same_line(line_a, line_b)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    lines=PRINTED_TEXT_LINES,
    others=PRINTED_TEXT_LINES,
    blanks=hypothesis.strategies.lists(BLANKS, max_size=4),
    at=hypothesis.strategies.integers(min_value=0, max_value=5),
)
def test_first_difference_is_none_for_equal_texts_and_ignores_blank_lines(lines: list[str], others: list[str], blanks: list[str], at: int) -> None:
    """``first_difference`` (T2): ``None`` for a text against itself; blank lines put into either text do not change the result."""
    text, other = '\n'.join(lines), '\n'.join(others)
    assert audit.printdiff.first_difference(text, text) is None
    padded = list(lines)
    position = min(at, len(padded))
    padded[position:position] = blanks
    padded_text = '\n'.join(padded)
    assert audit.printdiff.first_difference(padded_text, other) == audit.printdiff.first_difference(text, other)
    assert audit.printdiff.first_difference(other, padded_text) == audit.printdiff.first_difference(other, text)


NAME_PARTS = hypothesis.strategies.text(alphabet='abc 12()[]', max_size=8)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(parts=hypothesis.strategies.lists(NAME_PARTS, min_size=1, max_size=4))
def test_normal_path_is_idempotent_and_leaves_no_index(parts: list[str]) -> None:
    """``normal_path`` (T2): ``normal_path(normal_path(p)) == normal_path(p)`` and the result has no ``[digits]`` index."""
    once = audit.report.normal_path('$'.join(parts))
    assert audit.report.normal_path(once) == once
    assert re.search(r'\[\d+\]', once) is None


CASE_IDS = hypothesis.strategies.text(alphabet='abc.-_01', max_size=12)
PLAIN_PATTERNS = hypothesis.strategies.text(alphabet='abc.-_01', max_size=4)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(case_id=CASE_IDS, pattern=PLAIN_PATTERNS)
def test_selected_matches_a_pattern_without_wildcards_as_a_substring(case_id: str, pattern: str) -> None:
    """``selected`` (T2): a pattern without ``*`` or ``?`` matches exactly when it is a substring of the case id."""
    assert audit.steps.selected(case_id, [pattern]) == (pattern in case_id)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    case_id=CASE_IDS,
    only=hypothesis.strategies.lists(hypothesis.strategies.text(alphabet='abc.-_01*?', max_size=4), max_size=3),
    extra=hypothesis.strategies.text(alphabet='abc.-_01*?', max_size=4),
)
def test_selected_never_loses_a_match_when_a_pattern_is_added(case_id: str, only: list[str], extra: str) -> None:
    """``selected`` (T2): adding a pattern to ``only`` never turns a selected case into an unselected one."""
    if audit.steps.selected(case_id, only):
        assert audit.steps.selected(case_id, [*only, extra])
