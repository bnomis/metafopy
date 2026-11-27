# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Published `rma_mv` cases for van Houwelingen et al. (2002) and Gleser and Olkin (2009) were never written

Found on 2026-11-26 at `900698c` during a `TODO.md` review: `src/tests/correctness/cases/published.toml:1465` (added in `b0d0bb6`, 2026-11-14) says
both sources "reproduce, but need inline arm-level data or chapter tables; left for Task 13". Task 13 of
`agents/tasks/correctness-mv-robust/tasks-1.md` is done but only files review answers and the predict interval cases; no case id names either source.
Both are seed sources of `agents/specs/correctness-mv-robust/spec.md` (the bivariate UN example on `dat.colditz1994` arm-level log odds, and the FE
example with known V from chapter 19 of the Handbook of Research Synthesis and Meta-Analysis, 2nd ed.).

Cause: the deferral named a task that did not carry the work.

Fix direction: write `published` cases for both, with the arm-level data and the chapter's tables inline as the spec allows; record any value that
does not reproduce as a mismatch with its reason; update the comment at `published.toml:1465`.
