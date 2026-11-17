# Review: specs of the TODO items 1 to 10 order

**Date:** 2026-11-17
**Order:** `agents/tasks/order-todo-items-1-10.md` (written after this review is answered)
**Review page:** https://claude.ai/artifact/5aCWmZDvUUFvHZUmugt1aq

## What is reviewed and how

The order covers the first ten `TODO.md` items as they stood at `194c324` (all re-checked open by a TODO review at 12:44 UTC on 2026-11-17). Two
became specs: `agents/specs/rma-mv-reml-stop/spec.md` (item 2) and `agents/specs/selmodel-fixed-delta0/spec.md` (item 7). The other eight become
five small task lists with no spec. Approve or flag the plan and each spec, and answer each decision. Approving a spec that changes metafopy's output
is the confirmation `claude/rules/metafopy.md` asks for. The task lists are written only after every item and decision has an answer.

## Automatic checks

- Both specs have the specdev sections, quote their `TODO.md` item's dated findings, and pass the pre-commit hooks (including the spec-signature
  audit check): pass.
- The Stijnen et al. (2010) paper item 10 needs is in `/Users/simonb/src/metafopy papers`: pass.
- The `rma-mv-reml-stop` spec's prototype ran every `rma_mv` test file on the development Mac (4098 passed, 1 xfailed, no failures) and the Linux
  container (390 passed), 2026-11-17: pass.

## The items

### 1. The plan: five small task lists, no spec

| Row | Item | List | Waits for | Stops for you |
|---|---|---|---|---|
| 1 | 1. The Docs workflow fails on three docstring links | `docs-strict-links`: rewrite the links in `confint.py`, `selmodel.py`, `fsn.py` as the docs site resolves them; `mkdocs build --strict` passes | - | - |
| 4 | 8. `ls.confint.scale.ML.bcg` stays an expected failure | `ls-confint-scale-branch`: find which branch is the profile's maximum at a = 1.3169; fix the reference or report a metafopy defect | - | only if metafopy's profile misses a higher point |
| 5 | 9. Simmonds and Higgins (2016) Fig 1 | `simmonds2016-fig1`: compare the marginal likelihood at `tau2 = 0` and 0.213; record a source difference or report | - | only if metafopy is the one that is wrong |
| 6 | 10. Stijnen et al. (2010) Table VII PN | `stijnen2010-pn`: check the treated rows of Table VI against the paper, then the likelihood; correct the data, record the source difference, or report | 5 (both edit `glmm_published.toml`) | only if metafopy is the one that is wrong |
| 7 | 3-6. The four audit ledger gaps | `audit-ledger-gaps`: one full audit run, then `known.toml` entries for items 3-5 and item 6 by decision D2 below | 2 and 3 (its audit then covers their output changes) | - |

- **Recommendation:** approve.

### 2. The spec `agents/specs/rma-mv-reml-stop/spec.md` (item 2, row 2)

- **What it changes:** after BFGS stops, `maximise` in `src/metafopy/models/mv_opt.py` takes capped Newton steps on the analytic REML score (default
  optimiser, no lasso). Confirmed cause: loosening `gtol` to 1e-5 on the Mac reproduces CI Linux's figures exactly (gradient 6.0e-07).
- **What output moves:** every `rma_mv` parity fit by at most 7.1e-08 relative (most by 1e-09 or less); no R fixture fails; ridge and boundary fits are
  left as now. `property.mv.reml-score.kon` tightens from 2e-6 to 1e-10 of its scale. R's own estimates sit at 8.0e-09 and 1.6e-09.
- **Choice made in the spec:** the steps are capped at 1e-3 in the transformed parameters; uncapped, `kon_spexp`'s `rho` would move from 0.05596 to
  0.03935 against R's 0.05592 and fail its fixture.
- **Recommendation:** approve.

### 3. The spec `agents/specs/selmodel-fixed-delta0/spec.md` (item 7, row 3)

- **What it changes:** by decision D1 below. Measured 2026-11-17: metafor treats the first weight as 1 in the normalising constant but uses the
  user's in the numerator, so a free fit only shifts ll (and the LRT, p 2.6e-5 with `[2, nan, nan]` against 0.43 with `[1, nan, nan]`), and an all-fixed
  fit of `[0.5, 0.25, 0.125]` is fitted as 1 : 0.25 : 0.125. Nothing in metafor's docs, tests or the audit uses a first weight other than 1.
- **Choice made in the spec:** the check covers every step form (`stepfun` with or without `decreasing`, and `stepcon`), since the other two silently
  replace the user's first weight with 1.
- **Waits for:** row 1 (both edit `src/metafopy/models/selmodel.py`).
- **Recommendation:** approve.

## Decisions

### D1. What `selmodel` does with a fixed first weight other than 1 on a step function

- **Background:** Vevea and Hedges (1995, pp. 424-425) fix the first weight at 1 to identify the model. On 2026-11-15 you chose to reject `stepfun`
  with `prec` with `InputError` for the same hard-coded first weight (D3 in `agents/reviews/order-numerical-fixes-specs.md`).
- **What it affects:** `src/metafopy/models/selmodel.py`, `src/tests/test_selmodel.py`, `docs/differences.md`; no R fixture or correctness case.
- **Options:**
  - *(a) Reject with `InputError`* (recommended): those calls raise "The first value of 'delta' must be 1 or nan ...", where they now return a fit.
  - *(b) Normalise with the user's first weight* (the paper's equation 5): ll, LRT, information criteria, the free weights' estimates and Wald tests and
    all-fixed fits change; R parity breaks for these calls.
  - *(c) Keep metafor's behaviour*, documented only.
- **Recommendation:** (a) - the precedent, no wrong answer left, one line of user code to get the same model, reversible.

### D2. `ls_na` with `mods = ~ ablat * alloc`: refuse as R does, or accept the difference (item 6)

- **Background:** probed on 2026-11-17 (R 4.6.2, metafor 5.2-1). R drops the redundant predictor of the interaction, then stops with "No missing values
  allowed in 'alpha.init'.": the scale model's starting values carry an NA from the dropped column. metafopy fits it (beta
  `[0.282, -0.024, 0.213, -0.134, -0.013]`) but warns "Error when trying to invert the Hessian.", so its standard errors are not available. No
  publication defines the case. The project's precedent: "metafopy does not reproduce a failure of R's optimiser" (`known.toml`, permutest of `ls_na`,
  2026-11-04).
- **What it affects:** `src/audit/data/known.toml` and `docs/differences.md` under (a); `src/metafopy/models/rma_ls_fit.py` (`start_alpha`) and a test
  under (b).
- **Options:**
  - *(a) Accept the difference* (recommended): a `known.toml` entry and a `docs/differences.md` entry; metafopy keeps returning the fit with its warning.
  - *(b) Refuse like R*: metafopy raises R's message when the starting values hold an NA; a test; the audit case then agrees.
- **Recommendation:** (a) - R's refusal comes from its own starting-value step after it has already dropped the column, not from a model that cannot
  be fitted, and refusing would copy that failure.

## Prerequisites

None for you to provide. R metafor's pinned library is in `/Users/simonb/src/metafopy/.audit/Rlib`, the papers are in place, and Docker (for the
`rma-mv-reml-stop` list's container check) was running on 2026-11-17.

## Answers

Answered on the review page on 2026-11-17 (15:52 UTC), read back from its `answers` store; no comments:

| Item or decision | Answer |
|---|---|
| 1. The plan: five small task lists, no spec | approve |
| 2. The spec `rma-mv-reml-stop` (item 2) | approve |
| 3. The spec `selmodel-fixed-delta0` (item 7) | approve |
| D1. A fixed first weight other than 1 on a step function | (a) reject with `InputError` |
| D2. `ls_na` with `mods = ~ ablat * alloc` | (a) accept the difference |

Both specs move to version 1.1, status approved; D1 is recorded in `agents/specs/selmodel-fixed-delta0/spec.md` and D2 in the
`audit-ledger-gaps` task list.
