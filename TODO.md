# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## `qtukey` raises ZeroDivisionError where R returns NaN

<!-- doc-counts: accept-start found on 2026-11-22 - a dated record, not a claim about the present -->
Found on 2026-11-22 by the property test of `qtukey` (Task 3 of `agents/tasks/property-tests-for-metafopy-helpers/tasks-1.md`), at `f1bad68`.
`metafopy.core.tukey.qtukey(0.25, 17, math.inf)` raises `ZeroDivisionError: division by zero` (`src/metafopy/core/tukey.py:194`). So do
`(0.01, 30, 100)`, `(0.1, 17, 100)` and `(0.05, 50, 5)`: a scan of p in {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}, nmeans in
{2, 3, 5, 10, 17, 30, 50} and df in {2, 3, 5, 10, 30, 100, 1000, 1e6, inf} raised for 53 of the 504 combinations, all with 17 or more means.
R 5.2-1's base `qtukey(0.25, 17, Inf)` and `qtukey(0.25, 17, 1e6)` return `NaN` with the warnings "convergence failed in 'qtukey'" and "NaNs
produced" (`LC_ALL=C.UTF-8 Rscript`, R library `/Users/simonb/src/metafopy/.audit/Rlib`).
<!-- doc-counts: accept-end -->

Cause: for a small p and many means `ptukey` is flat at 0 around the start value, so `valx1 - valx0` is 0 in the secant step
(`src/metafopy/core/tukey.py:194`). C and R divide to a non-finite number and carry on; Python raises. The callers
`src/metafopy/methods/emm_adjust.py:125` and `:231` (Tukey adjustment of a family of 17 or more means) have no handler. Whether the port's
`return ans` after 50 iterations also differs from R's NaN on non-convergence is not checked.

Fix direction: return `NaN` with R's warning as R does; this changes an output, so it needs the operator's confirmation first. The test
`src/tests/test_tukey.py` narrows `nmeans` to 2 to 10 with a comment; widen it to 2 to 50 when this is fixed.

Spec: none (small item); tasks: `agents/tasks/qtukey-nan/tasks-1.md`.

## `weibull_shape_start` stops at its start value for large shapes

<!-- doc-counts: accept-start found on 2026-11-22 - a dated record, not a claim about the present -->
Found on 2026-11-22 by the property test of `weibull_shape_start` (Task 3 of `agents/tasks/property-tests-for-metafopy-helpers/tasks-1.md`),
at `f1bad68`. For mean 1 and sd 0.14836888830451764 (the coefficient of variation of a Weibull shape of 8),
`weibull_shape_start(1.0, 0.14836888830451764)` returns 7.94188553535, its start value `(mean/sd)^1.086` = 7.94185 to five digits. R's
`nlminb(start, objective, lower = 1e-3, upper = Inf)` with the same start and objective returns 8 (to ten digits). The coefficient of variation
of the returned shape is off by 0.7% at a shape of 8 and 2.3% at 20; at shapes of 0.5 to 4 it is within 1e-5.
<!-- doc-counts: accept-end -->

Cause: suspected, not confirmed: the L-BFGS-B call `lbfgsb` in `src/metafopy/effects/fivenum_estimators.py` (`ftol` 1e7 times epsilon, `gtol`
0) stops once the squared-error objective, already near 1e-6, changes by less than `ftol`; `nlminb` uses a relative tolerance of 1e-10. The
only caller is `src/metafopy/effects/fivenum_estimators.py:348`, which uses the shape as a start for a later fit, so the effect on a final
estimate is not measured.

Fix direction: make the optimiser reach R's stop for this one-parameter problem (a root find on the coefficient of variation would do);
this changes an output, so it needs the operator's confirmation first. Then remove the `xfail` marker of
`src/tests/test_conv_fivenum_methods.py::test_weibull_shape_start_property_reproduces_the_coefficient_of_variation` (strict, so it fails once
fixed).

Spec: none (small item); tasks: `agents/tasks/weibull-shape-start/tasks-1.md`.

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
