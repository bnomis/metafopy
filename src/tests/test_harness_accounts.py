"""The harness accounts for all of metafor: every exported function has a case or a waiver and every argument a value or a reason.

This runs on the committed snapshot of the R universe (``src/audit/data/universe.json``), so it needs neither R nor metafor; a new metafor release
shows up when ``bin/metafor-audit.sh --update-universe`` refreshes the snapshot and these tests fail on what is new.
"""

import hypothesis
import hypothesis.strategies
import pytest

import audit.cases
import audit.domains
import audit.generate
import audit.ledger
import audit.recall
import audit.universe


@pytest.fixture(scope='module')
def generated() -> audit.generate.Generated:
    universe = audit.universe.Universe.load(audit.universe.snapshot_path())
    return audit.generate.generate(universe, audit.domains.Domains.load(), audit.domains.Recipes.load(), audit.cases.load_fixtures())


def test_every_function_has_a_recipe_or_a_waiver(generated: audit.generate.Generated) -> None:
    assert generated.unaccounted == []


def test_every_argument_has_values_or_a_reason(generated: audit.generate.Generated) -> None:
    assert generated.uncovered == []


def test_case_ids_are_unique(generated: audit.generate.Generated) -> None:
    ids = [c.id for c in generated.cases]
    assert len(ids) == len(set(ids))


def test_every_fixture_builds_in_r() -> None:
    universe = audit.universe.Universe.load(audit.universe.snapshot_path())
    assert {name: info['error'] for name, info in universe.fixtures.items() if 'error' in info} == {}


def test_every_fixture_has_a_class_the_matrix_uses(generated: audit.generate.Generated) -> None:
    universe = audit.universe.Universe.load(audit.universe.snapshot_path())
    named = {fx['name'] for fx in audit.cases.load_fixtures()}
    assert named == set(universe.fixtures)
    printed = {c.call['args']['x']['$ref'] for c in generated.cases if c.id.startswith('matrix/print/')}
    assert printed == {name for name in named if universe.classes_of(name)}


def test_cases_refer_to_fixtures_that_exist(generated: audit.generate.Generated) -> None:
    named = {fx['name'] for fx in audit.cases.load_fixtures()}
    assert {need for c in generated.cases for need in c.needs} <= named


def test_the_ledger_entries_say_why() -> None:
    ledger = audit.ledger.Ledger.load()
    assert ledger.entries
    assert all(e.reason and e.match for e in ledger.entries)


def test_recall_corpus_is_well_formed() -> None:
    corpus = audit.recall.Corpus.load()
    audits = {i.audit for i in corpus.items}
    assert audits == set(corpus.audits)
    for item in corpus.items:
        assert item.text
        assert item.find or item.reach == 'none'
        if item.reach == 'none':
            assert item.why
        for detector in item.find:
            assert set(detector) <= {'id', 'kind', 'path', 'detail', 'step'}


def test_a_ledger_entry_takes_a_list_of_paths_and_a_detail_glob() -> None:
    entry = audit.ledger.Entry('options/x/*', 'value-diff', ['$a', '$b$*'], '*(relative 0.00*', 'why', '')
    finding = {'kind': 'value-diff', 'path': '$b$c', 'detail': 'element 1: R 1, Python 1.001 (relative 0.001)'}
    assert entry.matches('options/x/y', finding)
    assert not entry.matches('options/x/y', {**finding, 'path': '$digits'})
    assert not entry.matches('options/x/y', {**finding, 'detail': 'element 1: R 1, Python 2 (relative 0.5)'})
    assert not entry.matches('matrix/x/y', finding)


def test_a_ledger_entry_takes_a_list_of_case_globs() -> None:
    entry = audit.ledger.Entry(['scripts/a/*', 'scripts/b/*'], 'value-diff', '*', '*', 'why', '')
    finding = {'kind': 'value-diff', 'path': '', 'detail': 'x'}
    assert entry.matches('scripts/b/3', finding)
    assert not entry.matches('scripts/c/3', finding)


def test_a_load_dependent_timeout_entry_is_never_stale() -> None:
    slow = audit.ledger.Entry('options/tes/x/*', 'timeout', '*', '*', 'why', '', load_dependent=True)
    other = audit.ledger.Entry('options/vif/x/*', 'timeout', '*', '*', 'why', '')
    assert audit.ledger.Ledger([slow, other]).stale() == [other]


def test_ledger_globs_escape_a_literal_bracket() -> None:
    assert [(e.label(), glob) for e, glob in audit.ledger.Ledger.load().loose_brackets()] == []


def test_a_bracket_set_of_quoted_text_is_found_and_an_escaped_one_matches() -> None:
    loose = audit.ledger.Entry('scripts/*', 'drawn-text-diff', '*', "R only ['FALSE']*", 'why', '')
    escaped = audit.ledger.Entry('scripts/*', 'drawn-text-diff', '*', "R only [[]'FALSE'[]]*", 'why', '')
    plain = audit.ledger.Entry('api/*/code[1-4]', '*', '$[pQ]*', '*', 'why', '')
    finding = {'kind': 'drawn-text-diff', 'path': '', 'detail': "R only ['FALSE'], Python only [] [plot(sav)]"}
    assert not loose.matches('scripts/a/1', finding)
    assert escaped.matches('scripts/a/1', finding)
    assert audit.ledger.Ledger([loose, escaped, plain]).loose_brackets() == [(loose, "R only ['FALSE']*")]
    assert audit.ledger.bracket_sets('[]*') == []
    assert audit.ledger.bracket_sets('matrix/[/*') == []


def small_universe() -> audit.universe.Universe:
    """A universe in the shape of ``universe.json``."""
    return audit.universe.Universe(
        {
            'functions': {'plotfn': {'draws': True}, 'quiet': {}},
            'methods': [
                {'generic': 'g', 'class': 'rma.uni', 'chkclass': [{'notav': ['rma.mv']}], 'draws': False},
                {'generic': 'g', 'class': 'rma', 'chkclass': []},
                {'generic': 'm', 'class': 'rma.mv', 'chkclass': [{'must': ['rma.mv']}]},
                {'generic': 'd', 'class': 'rma', 'draws': True},
            ],
            'datasets': [],
            'fixtures': {},
        }
    )


def test_universe_method_for_is_the_first_class_of_the_chain_with_a_method() -> None:
    """``Universe.method_for`` follows R's dispatch: the first class in the chain that has a method, or none."""
    universe = small_universe()
    assert universe.method_for('g', ['rma.mv', 'rma.uni', 'rma'])['class'] == 'rma.uni'  # type: ignore[index]
    assert universe.method_for('g', ['rma.mv', 'rma'])['class'] == 'rma'  # type: ignore[index]
    assert universe.method_for('g', ['other']) is None
    assert universe.method_for('missing', ['rma']) is None


def test_universe_draws_when_the_generic_or_a_method_draws() -> None:
    """``Universe.draws`` is true when the function entry or any method of the generic draws."""
    universe = small_universe()
    assert universe.draws('plotfn')
    assert universe.draws('d')
    assert not universe.draws('g')
    assert not universe.draws('quiet')


def test_universe_refuses_what_the_class_checks_reject() -> None:
    """``Universe.refuses``: no method, a ``must`` class absent, or a ``notav`` class present is refused; otherwise it is not."""
    universe = small_universe()
    assert universe.refuses('nothing', ['rma'])
    assert universe.refuses('m', ['rma.uni'])
    assert not universe.refuses('m', ['rma.mv'])
    assert universe.refuses('g', ['rma.uni', 'rma.mv'])
    assert not universe.refuses('g', ['rma.uni'])


def test_refs_in_finds_every_fixture_a_value_refers_to() -> None:
    """``refs_in`` collects ``$ref`` names and the fixture of a ``$col``, through nested dicts and lists."""
    value = {'a': {'$ref': 'fit'}, 'b': [{'$col': ['frame', 'col']}, 3, {'c': {'$ref': 'x'}}]}
    assert sorted(audit.domains.refs_in(value)) == ['fit', 'frame', 'x']
    assert audit.domains.refs_in(3) == []


def test_ledger_classify_returns_the_first_matching_entry_and_counts_it() -> None:
    """``Ledger.classify``: the first entry whose globs match the case and finding, counted as used; no match is ``None``."""
    first = audit.ledger.Entry('a/*', 'value', '*', '*', 'why', '')
    second = audit.ledger.Entry('a/*', '*', '*', '*', 'other', '')
    ledger = audit.ledger.Ledger([first, second])
    finding = {'kind': 'value', 'path': 'x', 'detail': 'd'}
    assert ledger.classify('a/1', finding) is first
    assert first.used == 1
    assert second.used == 0
    assert ledger.classify('b/1', finding) is None
    assert ledger.stale() == [second]


QUOTE_FREE = hypothesis.strategies.text(max_size=12).filter(lambda text: '"' not in text)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(text=QUOTE_FREE)
def test_default_string_unquotes_a_quoted_default(text: str) -> None:
    """``default_string`` (T2): for any ``s`` without a double quote, ``default_string('"' + s + '"') == s``; an unquoted default gives ``None``."""
    assert audit.generate.default_string(f'"{text}"') == text
    assert audit.generate.default_string(f'  "{text}"  ') == text
    if not text.strip().startswith('"'):
        assert audit.generate.default_string(text) is None


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(glob=hypothesis.strategies.text(max_size=12).filter(lambda text: '[' not in text))
def test_bracket_sets_is_empty_for_a_glob_without_a_bracket(glob: str) -> None:
    """``bracket_sets`` (T2): a glob without ``[`` has no sets."""
    assert audit.ledger.bracket_sets(glob) == []


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(body=hypothesis.strategies.text(min_size=1, max_size=8).filter(lambda text: ']' not in text and text != '!'))
def test_bracket_sets_finds_a_wrapped_set_body(body: str) -> None:
    """``bracket_sets`` (T2): a set body free of ``]`` (and not the lone ``!``, which ``fnmatch`` reads as an unclosed set), in brackets, gives that body."""
    assert audit.ledger.bracket_sets(f'[{body}]') == [body]
    assert audit.ledger.bracket_sets(f'a*[{body}]?') == [body]
