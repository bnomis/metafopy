"""The harness accounts for all of metafor: every exported function has a case or a waiver and every argument a value or a reason.

This runs on the committed snapshot of the R universe (``src/audit/data/universe.json``), so it needs neither R nor metafor; a new metafor release
shows up when ``bin/metafor-audit.sh --update-universe`` refreshes the snapshot and these tests fail on what is new.
"""

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
