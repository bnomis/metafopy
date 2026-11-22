# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## `make_names(unique=True)` keeps the input order where R puts the changed names last

<!-- doc-counts: accept-start found on 2026-11-22 - a dated record, not a claim about the present -->
Found on 2026-11-22 while fixing `make_unique` (`agents/tasks/make-unique-first-occurrences/tasks-1.md`, Task 1).
`metafopy.effects.datatools.make_names(['a b', 'a b', 'a.b'], unique=True)` returns `['a.b', 'a.b.1', 'a.b.2']`. R's
`make.names(c("a b", "a b", "a.b"), unique = TRUE)` returns `"a.b.1" "a.b.2" "a.b"` (`LC_ALL=C.UTF-8 Rscript`, metafor 5.2-1).
<!-- doc-counts: accept-end -->

Cause: R's `make.names` body is `o <- order(names != names2); names2[o] <- make.unique(names2[o])` (read with `print(body(make.names))`
in `Rscript` on 2026-11-22), so names that `make.names` left unchanged come first and keep their label, and the changed ones are made unique
after them. `make_names` in `src/metafopy/effects/datatools.py` calls `metafopy.metadata.make_unique` on the names in input order.

Fix direction: order the names as R does before calling `make_unique`, then put them back in input order. This changes an output, so it needs
the operator's confirmation first (on 2026-11-22 the operator chose to leave it as it is for the `make_unique` fix). Then remove the strict
`xfail` marker of `src/tests/test_name_matching_properties.py::test_make_names_unique_orders_changed_names_last`.

Spec: none (small item); tasks: none yet.

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
