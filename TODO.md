# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## `make_unique` renames a label that occurs once when it collides with a generated one

<!-- doc-counts: accept-start found on 2026-11-22 - a dated record, not a claim about the present -->
Found on 2026-11-22 by the property test of `make_unique` (Task 8 of `agents/tasks/property-tests-for-metafopy-helpers/tasks-1.md`).
`metafopy.metadata.make_unique(['a', 'a', 'a.1'])` returns `('a', 'a.1', 'a.1.1')`: the label `a.1` that the caller supplied is renamed
to `a.1.1`. R's `make.unique(c("a", "a", "a.1"))` returns `"a" "a.2" "a.1"`, the case `?make.unique` documents (`LC_ALL=C.UTF-8 Rscript`,
metafor 5.2-1). The hypothesis shrunk example is `['a', 'a', 'a.1']`.
<!-- doc-counts: accept-end -->

Cause: `make_unique` in `src/metafopy/metadata.py` (lines 31 to 50) checks a candidate only against the labels it has already emitted
(`seen`), so a later input label that equals a generated one is treated as a duplicate. R's `make.unique` keeps every first occurrence and
generates a suffix that avoids every input label. The caller at `src/metafopy/metadata.py:359` makes the study labels (`slab`) shown in
output. `metafopy.effects.datatools.make_names` with `unique=True` calls it too; whether that path shows the same fault is not checked.

Fix direction: seed `seen` with the set of all input labels before generating suffixes; this changes an output, so it needs the operator's
confirmation first. Then remove the strict `xfail` marker of
`src/tests/test_name_matching_properties.py::test_make_unique_keeps_first_occurrences`.

Spec: none (small item); tasks: `agents/tasks/make-unique-first-occurrences/tasks-1.md`.

## Audit-harness and bin helpers with branches have no direct test

<!-- doc-counts: accept-start found on 2026-11-21 - a dated record, not a claim about the present -->
Found on 2026-11-21 by the first auditme audit (`agents/audit-reviews/review-1.md`, question T1, at `31d809f`): 108 functions under
`src/audit` (most in `rlib.py`, `transpile.py`, `pycall.py`, `generate.py` and `apidiff.py`), and 9 in `bin/logo.py`,
`bin/metafor-audit.py` and `bin/platform-probe.py`, branch and have no direct test. The list is in
`agents/audit-reviews/ledger/T1.jsonl`. Decision D4 of that review (answered 2026-11-21) files it here.
<!-- doc-counts: accept-end -->

Cause: the harness is exercised end to end by `bin/metafor-audit.sh --recall` and by a few step tests, not function by function.

Fix direction: a spec with `/specdev`. It decides which harness functions need direct tests (the R-library shims in `rlib.py` and the
transpiler first), and dismisses the rest in the ledger with the reason that the recall check covers them.

Note 2026-11-22: this item also takes the T2 (property test) findings under `src/audit` and `bin`, moved from the item "Numerical cores
have no property-based tests" (decision D2 of `agents/reviews/property-tests-for-metafopy-helpers-spec.md`). The list is in
`agents/audit-reviews/ledger/T2.jsonl`.
