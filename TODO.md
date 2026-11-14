# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## QE and the moment tau2 estimators centre yi in models without an intercept

Found on 2026-11-12 while writing the location-scale correctness spec (`agents/specs/correctness-location-scale/spec.md`, Implementation step 3); the
user answered *file it and fix it in this order* on 2026-11-13 (`agents/reviews/order-correctness-follow-ups-specs.md`, D1). R metafor 5.2-1 centres
`yi` once in `rma.uni` (`ymci <- scale(yi, center = TRUE, scale = FALSE)`) and uses it for QE and for the residual sums of squares of the tau2
estimators. `y' P y` is unchanged by the centring only when the column space of X holds the constant, so the result is Cochran's Q only for a model
with an intercept. metafopy ports it: `ymc = yi - yi.mean()` in `src/metafopy/models/tau2.py` (`estimate_tau2`, then every `ymc @ P @ ymc`, the QE
root-finding and the likelihoods), in the heterogeneity block of `src/metafopy/models/rma_uni.py` and in `src/metafopy/models/rma_ls_fit.py`. Measured
on 2026-11-12 on `dat.bangertdrowns2004` with `mods = ni/100` and `intercept=False`: QE is 95.742 in metafopy and in metafor, against Cochran's `y' P y`
of 135.998; the DerSimonian-Laird tau2 is 0.0384 in metafopy, against 0.0701 from the same formula with Cochran's Q. The other estimators were not
measured.

Fix direction: use `yi` itself, not `ymc`, in each quadratic form and likelihood (exact with an intercept as well, so only `intercept=False` fits
change); add a derived correctness case for QE and each moment estimator on the model above, and a property case that QE with an intercept is unchanged;
mark the R parity fixtures that hold a no-intercept QE or tau2 as differing; add a `src/audit/data/known.toml` entry, and name the difference in
`docs/differences.md` under "Where metafor departs from its reference" (the section the `ht` item above adds).

Spec: none (small item); tasks: `agents/tasks/qe-no-intercept/tasks-1.md`.

## Extend the correctness tests beyond the core models

Found on 2026-11-12 while reviewing `TODO.md`: the correctness-tests spec (`agents/specs/correctness-tests/spec.md`, introduction) covers `escalc`,
`rma_uni`, `rma_mh` and `rma_peto`, and says "Later specs extend the same harness to `rma_mv`, `rma_glmm`, `rma_ls`, the bias methods, `selmodel` and
`robust`." No such spec exists yet, so those functions are checked only against R metafor, not against their defining references.

Fix direction: one spec per area with `/specdev`, each listing its defining references, the sources to fetch or ask the user for, and its coverage items in
`src/tests/correctness/test_coverage.py`; do the harness item above first, since several of these areas report back-transformed or interval values.

Spec: `agents/specs/correctness-mv-robust/spec.md`; tasks: `agents/tasks/correctness-mv-robust/tasks-1.md`. Spec:
`agents/specs/correctness-glmm/spec.md`; tasks: `agents/tasks/correctness-glmm/tasks-1.md`. Spec: `agents/specs/correctness-location-scale/spec.md`;
tasks: `agents/tasks/correctness-location-scale/tasks-1.md`. Spec: `agents/specs/correctness-publication-bias/spec.md`; tasks:
`agents/tasks/correctness-publication-bias/tasks-1.md`.
