# Review: the factor-coding-follows-r-model-matrix spec

**Review page:** https://claude.ai/artifact/L3xcqoWFqU6HJgDYE3ri2q
**Date:** 2026-11-27

**What is being reviewed and how.** `/todo-do` wrote one new spec, `agents/specs/factor-coding-follows-r-model-matrix/spec.md`, for the `TODO.md`
item "A two-factor interaction with neither main effect gets other columns than in R". Measuring showed that formulaic keeps every model matrix full
rank, while R codes such factors in full and metafor then drops the redundant column, so the two report different coefficients for the same model.
It also showed that `emmprep` has refused fits with an interaction written out of first-appearance order since the coefficient-names work
(`367c13d`), because it builds its matrix without `design_matrix`'s rules. Approve or flag each item below; there are no open decisions.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-27 - a dated record, not a claim about the present -->
- Pass: the item is still open at `cb24530`: `design_matrix('~ deltype:outcome')` gives 6 columns where R gives 7.
- Pass: the models already drop redundant columns with metafor's messages: `rma_uni`, `rma_mv`, `rma_glmm` and the location-scale fit call
  `drop_redundant`.
- Pass: a prototype `RPandasMaterializer` matched R's `model.matrix` names and rank on every measured formula.
- Pass: no `mods=`, `scale=` or test formula in `src/`, `docs/` or `README.md` is one of the affected shapes.
- Pass: the spec has no non-ASCII characters and no line over 160 characters.
<!-- doc-counts: accept-end -->

## The items

### S1. `design_matrix` codes each factor as R's `model.matrix` does

- **What:** a new `RPandasMaterializer` in `metafopy.formula` gives each categorical factor of each term contrasts when the term without it is empty
  (with an intercept) or contained in an earlier term, and full indicator coding otherwise; without an intercept the first main-effect factor is
  coded in full. `~ deltype:outcome` then has R's seven columns, and the models drop `deltypeovert:outcomePD` with metafor's warning.
- **Output change:** only formulas whose R matrix is rank-deficient (`~ f:g`, `~ f:g:h`, `~ f:g + g:h`, `~ 0 + f:g + h`): their coefficients,
  names, per-coefficient statistics and `btt=` selections follow metafor; fitted values, `QM`, `tau2` and the log-likelihood do not change; `emmprep`
  then refuses such a fit, as metafor's does. On 2026-11-27 no formula in the repository's tests and docs is affected. Approving this item is the
  confirmation `claude/rules/metafopy.md` asks for before output changes.
- **Recommendation:** approve.

### S2. `emmprep` builds its grid through the same builder

- **What:** a new `metafopy.formula.r_model_matrix` becomes the only call of `formulaic.model_matrix`; `design_matrix` and
  `emm_grid.reference_grid` both use it, which fixes `emmprep` on `~ g + f:g` (works at `c8fe0fe` and in R, fails since `367c13d`). Chosen in
  writing the spec, reversible.
- **Recommendation:** approve.

### S3. The rule overrides formulaic's private `_get_scoped_terms`

- **What:** one overridden method of formulaic 1.2.2's `PandasMaterializer`. Formula rewriting with `C(x, spans_intercept=False)` fails on
  `~ f:g + g:h`; building the matrix outside formulaic duplicates its encoding. formulaic is pinned in `uv.lock`, and the R oracles fail if the
  override stops holding. Chosen in writing the spec, reversible.
- **Recommendation:** approve.

### S4. `interface-meta` becomes a declared dependency

- **What:** the override needs `interface_meta.override`, so `pyproject.toml` gains `interface-meta>=2.0` (2.0.1 is already installed) and
  `uv lock` is run; `uv.lock` should change only by that entry. Approving this item is the instruction `AGENTS.md` asks for before `uv.lock`
  changes.
- **Recommendation:** approve.

### S5. Two R oracles check the rule

- **What:** `generate_formula_names.R` gains a factor `arm` and nine no-main-effect and no-intercept formulas (existing entries unchanged); a new
  `generate_formula_rank.R` records `rma`, `rma.mv` and a location-scale fit on `~ deltype:outcome`, checked by a new `test_formula_rank.py`;
  `test_emmeans_nesting.py` gains the out-of-order case; `test_formula.py` gets direct materializer tests.
- **Recommendation:** approve.

### S6. `docs/guide/models.md` says how a rank-deficient formula is fitted

- **What:** one sentence where it describes `mods`; `docs/differences.md` needs no change.
- **Recommendation:** approve.

## Answers


Answered on the review page on 2026-11-27 (09:29 UTC), copied here the same day:

| Item | Answer | Comment |
|---|---|---|
| S1. `design_matrix` codes each factor as R's `model.matrix` does | approve | - |
| S2. `emmprep` builds its grid through the same builder | approve | - |
| S3. The rule overrides formulaic's private `_get_scoped_terms` | approve | - |
| S4. `interface-meta` becomes a declared dependency | approve | - |
| S5. Two R oracles check the rule | approve | - |
| S6. `docs/guide/models.md` says how a rank-deficient formula is fitted | approve | - |

Approving S1 is the operator's confirmation of the output change that `claude/rules/metafopy.md` requires; approving S4 is the instruction to change
`pyproject.toml` and run `uv lock`.
