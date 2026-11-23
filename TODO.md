# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Ten `src/metafopy` functions newly flagged as having no direct test

<!-- doc-counts: accept-start found on 2026-11-22 - a dated record, not a claim about the present -->
Found on 2026-11-22 by the T1 re-judge (a branching function has no direct test, brief version 6) in Task 2 of
`agents/tasks/direct-tests-for-the-audit-harness/tasks-1.md` (commit `0425563`), which recorded them in
`agents/specs/direct-tests-for-the-audit-harness/open-after-rejudge.md` under "Outside this list" and did not test them. Still open in
`phaseb.py report . --question T1` on 2026-11-23 at `58c7418`:

- `src/metafopy/bias/fsn.py::FsnResult.summary`, `bias/hc.py::HcResult.summary`, `bias/regtest.py::RegtestResult.summary`,
  `bias/tes.py::TesResult.summary`, `bias/trimfill.py::TrimfillResult.summary` (with `showfit=True`): tests reach each only through `str()`;
- `src/metafopy/core/labelled.py::RFrame.text` and `SparseVCovMat.text`: reached only through `__str__`;
- `src/metafopy/core/numderiv.py::hessian_calculus`: reached only through `hessian_by_package`;
- `src/metafopy/models/mv_result.py::RMAMVResult.R` (with a mapping given and with none) and `formula_text` (through
  `RMAMVResult.formulas`, with and without a random term).
<!-- doc-counts: accept-end -->

Cause (from the verdicts' reasons; the brief versions were not compared): brief version 6 does not count a one-line wrapper's caller (`str()`, `__str__`, a delegating function) as a direct test of a public
candidate, so these verdicts turned open on re-judging.

Fix direction: a direct test for each, asserting the value its R counterpart prints or returns (the existing `str()` tests already compare
with R, so their expected text can be reused); no metafopy output changes. Then re-judge T1 for these files (`phaseb.py plan/check/merge`)
and check `phaseb.py report . --question T1` lists none of them.
