# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## A two-factor interaction with neither main effect gets other columns than in R

<!-- doc-counts: accept-start found on 2026-11-27 - a dated record, not a claim about the present -->
Found on 2026-11-27 at `e9a5f1d` (macOS arm64, R 4.6.2) while writing `agents/specs/coefficient-names-and-print-tests-follow-r/spec.md`, which
leaves it out of scope (its review item S6, approved 2026-11-27): on a small data frame with `deltype` (three levels) and `outcome` (two),
`model.matrix(~ deltype:outcome)` gives seven columns (`(Intercept)`, then `deltypecovert:outcomeAL` to `deltypeovert:outcomePD`, both factors
coded in full), while `metafopy.formula.design_matrix('~ deltype:outcome')` gives six (`intrcpt`, `outcomePD`, then the four `deltypegeneral` and
`deltypeovert` products), coding the second factor by contrasts.
<!-- doc-counts: accept-end -->

Cause: unconfirmed. formulaic and R choose differently which factors of a term without its marginal terms are coded in full; R's rule is in
`stats::model.matrix` (the term's factors that are not contained in an earlier term are given full indicator coding).

Fix direction: check what metafor then does with R's rank-deficient seven columns (it drops redundant columns with a note) and match the columns
and names it ends with; measure before deciding whether it needs a spec. Do after the spec above ships, since both change `design_matrix`.

Note 2026-11-27: the coefficient-names spec it waited for shipped in `e2438a2`; nothing blocks it now.
Spec: `agents/specs/factor-coding-follows-r-model-matrix/spec.md`; tasks: `agents/tasks/factor-coding-follows-r-model-matrix/tasks-1.md`.

## Published `rma_mv` cases for van Houwelingen et al. (2002) and Gleser and Olkin (2009) were never written

Found on 2026-11-26 at `900698c` during a `TODO.md` review: `src/tests/correctness/cases/published.toml:1465` (added in `b0d0bb6`, 2026-11-14) says
both sources "reproduce, but need inline arm-level data or chapter tables; left for Task 13". Task 13 of
`agents/tasks/correctness-mv-robust/tasks-1.md` is done but only files review answers and the predict interval cases; no case id names either source.
Both are seed sources of `agents/specs/correctness-mv-robust/spec.md` (the bivariate UN example on `dat.colditz1994` arm-level log odds, and the FE
example with known V from chapter 19 of the Handbook of Research Synthesis and Meta-Analysis, 2nd ed.).

Cause: the deferral named a task that did not carry the work.

Fix direction: write `published` cases for both, with the arm-level data and the chapter's tables inline as the spec allows; record any value that
does not reproduce as a mismatch with its reason; update the comment at `published.toml:1465`.
