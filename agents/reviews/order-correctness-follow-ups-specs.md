# Review: the specs of the correctness follow-ups order

**Date:** 2026-11-12
**Order:** `agents/tasks/order-correctness-follow-ups.md` (not written yet; it follows this review)
**Review page:** https://claude.ai/artifact/Q3n16EnQj6N19azd41iHRQ

## What is reviewed and how

`/todo-run-order 1-4` takes the four `TODO.md` items of 2026-11-12 to a run order. Items 1 to 3 (the `rma_uni` ZeroDivisionError, the `ht`
interval centre, the harness check functions) are small and become task lists without a spec. Item 4 ("Extend the correctness tests beyond the core
models") is large and becomes four specs, written on 2026-11-12, each extending the correctness harness of `agents/specs/correctness-tests/spec.md` to
one area. Approve or flag each spec, answer each decision, and say whether you will supply the papers before the order starts. The task lists are
written only after every answer is in, because the answers change the specs.

## Automatic checks

- Each spec has the specdev template sections (introduction, Module location, API, Error handling, Implementation, External dependencies): pass.
- Each spec is listed in `agents/specs/index.md` with status proposed: pass.
- Every DOI in the specs' source tables was checked against Crossref on 2026-11-12 by the agent that wrote the spec: pass, as reported. The Knapp and
  Hartung (2003) DOI was checked earlier the same day.
- Size: `correctness-mv-robust` is 216 lines and `correctness-glmm` has 10 top-level sections, both a little over the specdev guide (about 200 lines,
  8 sections); each extra section carries content a task list needs. Noted, not a failure.

## The specs

### `agents/specs/correctness-mv-robust/spec.md` - rma_mv and robust

Adds derived cases from an independent multivariate ML/REML likelihood and GLS fit for every `struct`, test and `dfs` option, and the sandwich family
(CR0 to CR3, Satterthwaite degrees of freedom, the Hotelling-type Wald tests) for `robust`. Small product change: `METHODS`, `TESTS`, `DFS` constants in
`rma_mv.py` and `ADJUST` in `robust.py`, used by the existing validation, so the gate reads the option lists from code. Other software: `mvmeta`,
`nlme`, `robumeta`, `sandwich`, all already in the audit's pinned R library, so nothing is installed. `clubSandwich` is not used because metafopy ports
it. Stops for you: sources to supply; the mismatch review.

### `agents/specs/correctness-glmm/spec.md` - rma_glmm

Adds `glmm_formulas.py`, an independent binomial, Poisson and non-central hypergeometric likelihood with exact, adaptive-quadrature and Laplace
integration. Each ML case checks the likelihood at metafopy's estimates (the strong check) and the estimates against an independent optimum at a
stated looser tolerance. The quadrature and Laplace approximations are treated as documented conventions. Other software: base R
`mantelhaen.test(exact = TRUE)`, and R `glmmML` if installed. Stops for you: sources to supply; the mismatch review.

### `agents/specs/correctness-location-scale/spec.md` - location-scale models

Adds the location-scale likelihood with a hand-coded gradient and observed information, profile bounds, the Knapp-Hartung factor, prediction and exact
permutation p-values. Strong properties: an intercept-only scale model equals `rma_uni`; a categorical moderator in both parts equals separate fits.
Small product change: an `LS_METHODS` constant in `rma_ls_fit.py`. Other software: `glmmTMB`, already in the audit's pinned library. The defining paper
(Viechtbauer and Lopez-Lopez 2022) is by metafor's author, and its numbers were computed with metafor. Stops for you: the paper's PDF (open access, but
the PubMed text lost its tables) and other sources; the mismatch review.

### `agents/specs/correctness-publication-bias/spec.md` - regtest, ranktest, trimfill, fsn, tes, hc and selmodel

Adds re-derivations from the publications: Egger's original OLS form, Begg and Mazumdar, the trim-and-fill estimators L0, R0 and Q0, the fail-safe
N types, TES, Henmi and Copas, and one general selection-model likelihood with each weight function. Harness change: `setup` gains `where`, `drop` and a
fitted `rma_uni` model. Two papers you have already supplied give worked examples here: Borenstein et al. (2009) chapter 30 and the Handbook chapter 23.
Other software: R `weightr` 2.0.2 (independent of metafor) and R `meta`'s `metabias` and `trimfill`. Stops for you: sources to supply; installing
`weightr`; the mismatch review.

## Decisions

### D1. The QE and tau2 defect in models without an intercept

- **What is decided:** whether to file the defect the location-scale spec found as a `TODO.md` item fixed in this order, ahead of the four specs.
- **Background:** metafor 5.2-1 centres `yi` once (`ymci <- scale(yi, center = TRUE, scale = FALSE)` in `rma.uni`), and uses it for QE and for the
  residual sums of squares of the tau2 estimators. That is exact only when the model has an intercept. metafopy ports it. Checked on 2026-11-12 on
  `dat.bangertdrowns2004` with `mods = ni/100` and `intercept=False`: QE is 95.742 in metafopy and in metafor, while Cochran's `y'Py` is 135.998; the
  DerSimonian-Laird tau2 is 0.0384 in metafopy, while the formula with Cochran's Q gives 0.0701. The other estimators were not measured.
- **What it affects:** `rma_uni` (QE, the moment-based tau2 estimators) and possibly `rma_ls`, for models fitted with `intercept=False` only; a new
  difference from metafor in `docs/differences.md` and `known.toml`.
- **Options:**
  - *File it and fix it in this order* (recommended): a `TODO.md` item and a small task list that runs before the four specs. Correct results for
    no-intercept models, with the difference from metafor recorded.
  - *Leave it to the location-scale spec*: that spec writes a case that becomes a mismatch, and you decide at its review, several lists later.
- **Why recommended:** the reference (Cochran's Q, `sum w (y - X b)^2`) is unambiguous, and the error changes estimates, not only a printed statistic.

### D2. Options that only metafor or software documentation defines

- **What is decided:** how the gate treats options with no defining publication. This arises in two specs: `rma_mv` `test="knha"` and `"adhoc"`,
  `dfs="contain"`, the `Rscale` values and the Stata adjustments of `robust` (`correctness-mv-robust`); `fsn` type `Binomial`, the `selmodel` variants
  `halfnorm1` to `power2`, `trunc`, `truncest` and `tes.theta_lim` (`correctness-publication-bias`).
- **What it affects:** how many items have a derived case or an `[[exempt]]` entry; no metafopy code.
- **Options:**
  - *Derive from documented definitions, exempt the code-only ones* (recommended): where metafor's help or other documentation states the formula
    (the `rma_mv` tests, `contain`, `Rscale`, the Stata factors, `trunc`, `truncest`, `theta_lim`), a derived case codes it and its source names the
    documentation; options that exist only in metafor's code (`Binomial`, the numbered `selmodel` variants) get an exemption saying R parity covers them.
  - *Exempt them all*: fewer cases; those options are then checked only against metafor.
  - *Derive them all, from metafor's code where needed*: every item has a case, but the code-only ones only repeat the R parity tests.

### D3. Published values that metafor computed

- **What is decided:** whether printed values that a paper computed with metafor may satisfy the gate alone. This arises for Jackson et al. (2018) in
  `correctness-glmm` and Viechtbauer and Lopez-Lopez (2022) in `correctness-location-scale`.
- **What it affects:** whether every item in those two specs needs a derived case; the glmm spec already plans one for every item.
- **Options:**
  - *Count them as `published`, and require a derived case for every item as well* (recommended): the printed numbers stay checkable facts, and a
    bug metafor shares is still caught by the independent derivation.
  - *Count them as `published` alone*: an item covered only by such a value passes the gate.
  - *Count them as `other`*: they never count for the gate.

### D4. R packages outside the pinned library

- **What is decided:** whether to install R `glmmML` (for `correctness-glmm`) and R `weightr` 2.0.2 (for `correctness-publication-bias`) in the
  library `METAFOPY_CORRECTNESS_RLIB` names, `/Users/simonb/.local/share/metafopy/r-correctness-lib`, where R `meta` 8.5-0 already is.
- **Background:** neither depends on metafor, so each gives an independent check: `glmmML` for the one-group models and `CM.AL`, `weightr` for the
  Vevea-Hedges step-function selection models. The audit's pinned library (`src/audit/data/r-versions.json`) is not changed.
- **Options:**
  - *Install both before the order starts* (recommended): the other-software steps of those lists then run without stopping.
  - *Install neither*: those steps are dropped; `glmm` keeps only `mantelhaen.test` as other software, and `selmodel` stepfun has none.

## Papers to supply

Each spec stops to ask for the sources it cannot read open access. Putting them in `/Users/simonb/src/metafopy papers` before the order starts lets those
stops pass straight through. The full citations with DOIs are in each spec's source table; the review page lists them. Already in the folder and used
by these specs: Knapp and Hartung (2003), Higgins and Thompson (2002), Borenstein et al. (2009) and the Handbook of Research Synthesis and
Meta-Analysis (2nd ed., 2009). The publication-bias spec needs three chapters of Rothstein, Sutton and Borenstein (eds, 2005), Publication Bias in
Meta-Analysis, Wiley, doi:10.1002/0470870168, so the whole book serves.

## Answers

Read from the review page on 2026-11-13 (answers saved 06:14 UTC); no comments were left. Every answer is the recommended option.

| Item | Answer |
|---|---|
| `correctness-mv-robust` spec | approved |
| `correctness-glmm` spec | approved |
| `correctness-location-scale` spec | approved |
| `correctness-publication-bias` spec | approved |
| D1. QE and tau2 defect without an intercept | file it and fix it in this order, before the four specs |
| D2. Options only metafor or software documentation defines | derive from documented definitions, exempt the code-only ones |
| D3. Published values that metafor computed | count them as `published`, and require a derived case for every item as well |
| D4. R packages outside the pinned library | install `glmmML` and `weightr` before the order starts |
| D5. When to supply the papers | before the order starts |

Applied on 2026-11-13: each spec is at version 1.1, with its answered decisions under a "Decisions" section. R `glmmML` 1.1.7 and `weightr` 2.0.2 were
installed in `/Users/simonb/.local/share/metafopy/r-correctness-lib`; `src/audit/data/r-versions.json` is unchanged.

On 2026-11-14, asked in chat after the task list for the QE fix measured its extent, the user widened D1: the fix also covers the ML, REML and EB
estimators, whose Fisher scoring steps use the same centred `yi`. On `dat.bangertdrowns2004` with `mods = ni/100` and no intercept, metafopy's REML
tau2 is 0.0388 and ML 0.0346, while the REML and ML likelihoods of the uncentred data peak at 0.0930 and 0.0874.
