# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## `rma_mv` prints some coefficient labels unlike R, and the print tests skip them for a partly stale reason

<!-- doc-counts: accept-start found on 2026-11-26 - a dated record, not a claim about the present -->
Found on 2026-11-26 at `900698c` (macOS arm64) during a `TODO.md` review, by comparing metafopy's summaries with the R fixtures of the cases the
print tests skip with "coefficient labels follow the formula engine" (`src/tests/test_rma_mv.py:185`, `NAMES_DIFFER`, and
`src/tests/test_robust.py:159`, both since `88dbd52`, 2026-10-13):

- `ass_mods`, `ass_mods_t`: R prints `factor(deltype)general`, `factor(deltype)overt`; metafopy prints `deltypegeneral`, `deltypeovert`;
- `ber_un_mods`: R prints `outcomeAL:I(year - 1985)`; metafopy orders the interaction the other way, `I(year - 1985):outcomeAL`; its `QM` also
  prints `76.7085` against R's `76.7084`;
- `uni_cs_btt`: the label now matches R (`^`); the summary differs only in wrapping, since metafopy breaks the 81-character coefficient table at
  `ci.ub` where R prints it on one line;
- `kon_btt`: both summaries are identical, so its skip is stale.

`docs/differences.md` records none of these.
<!-- doc-counts: accept-end -->

Cause: unconfirmed. `metafopy.formula` names a `factor(x)` term's columns and orders an interaction's factors the formulaic way, not as R's
`model.matrix` does; the wrapping difference is in `metafopy.printing.fmt_wrapped_table` or in the width the coefficient table is measured at.

Fix direction: make the column names and interaction order follow R's `model.matrix` (check the other formula users, such as `rma_uni` and `rma_ls`,
on the same terms), find why R keeps the 81-character table on one line, then drop the skips that pass and record any difference that remains in
`docs/differences.md`. Check the `QM` digit against R before treating it as a tolerance.

## Published `rma_mv` cases for van Houwelingen et al. (2002) and Gleser and Olkin (2009) were never written

Found on 2026-11-26 at `900698c` during a `TODO.md` review: `src/tests/correctness/cases/published.toml:1465` (added in `b0d0bb6`, 2026-11-14) says
both sources "reproduce, but need inline arm-level data or chapter tables; left for Task 13". Task 13 of
`agents/tasks/correctness-mv-robust/tasks-1.md` is done but only files review answers and the predict interval cases; no case id names either source.
Both are seed sources of `agents/specs/correctness-mv-robust/spec.md` (the bivariate UN example on `dat.colditz1994` arm-level log odds, and the FE
example with known V from chapter 19 of the Handbook of Research Synthesis and Meta-Analysis, 2nd ed.).

Cause: the deferral named a task that did not carry the work.

Fix direction: write `published` cases for both, with the arm-level data and the chapter's tables inline as the spec allows; record any value that
does not reproduce as a mismatch with its reason; update the comment at `published.toml:1465`.
