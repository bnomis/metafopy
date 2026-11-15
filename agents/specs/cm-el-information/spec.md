# CM.EL information

**Version:** 1.1
**Status:** implemented (2026-11-15)
**Date:** 2026-11-15

`rma_glmm(measure='OR', model='CM.EL')` with the default optimiser maximises the exact conditional (non-central hypergeometric) likelihood of van
Houwelingen et al. (1993) and takes every standard error from a finite-difference Hessian of that likelihood, as R metafor does (numDeriv's Richardson
method with `r = 16`). This spec replaces that Hessian by the analytic information of the same likelihood: for the fixed-effects (`FE`, `EE`, `CE`)
fit, for the saturated fit behind `QE_Wld`, and for the `ML` fit. Coefficients, `tau2` and log-likelihoods do not change. The `clogit` and `clogistic`
optimisers, `CM.EL` for `IRR` (a binomial model) and every other model are out of scope.

The source is the `TODO.md` item "The CM.EL information is taken by finite differences of the exact likelihood" (consolidated on 2026-11-15) and
decisions D1 and D2 of `agents/specs/correctness-glmm/mismatches.md` (answered 2026-11-14). It follows the
[Correctness GLMM](../correctness-glmm/spec.md) spec, whose cases record the disagreement.

## Motivation

The findings of the item, carried over unchanged:

> Found on 2026-11-14 by the `rma_glmm` correctness cases (`agents/specs/correctness-glmm/mismatches.md`, decision D1); the user answered *follow the
> reference* for the `FE` fits and *investigate* for the `ML` fits on 2026-11-14. [...] only the standard error disagrees.
>
> **FE fits.** Cases `glmm.fit.or.cm.el.fe.{nielweise2007,collins1985a,jackson2018.pneumonia}.se`, tolerance 1e-6, each a strict expected failure.
> metafopy / formulas / R metafor 5.2-1: `dat.nielweise2007` 0.22403812 / 0.22403482 / 0.22403498; `dat.collins1985a` 0.09940439 / 0.09955789 /
> 0.09952917; the pneumonia outcome of Jackson et al. (2018) 0.25175750 / 0.25175350 / 0.25174661. The `FE` information of the exact conditional
> likelihood is analytic (the sum of the conditional variances), and metafopy is furthest from it, 1.5e-3 relative on `dat.collins1985a`.
>
> **ML fits.** Cases `glmm.fit.or.cm.el.ml.{nielweise2007,collins1985a,jackson2018.pneumonia}.se`, tolerance 1e-5, `int_rel_tol` 1e-10, each a strict
> expected failure. metafopy / formulas / R metafor: 0.35126655 / 0.35104033 / 0.35110612; 0.13635527 / 0.13638548 / 0.13648151; 0.88813349 /
> 0.88810230 / 0.88812403. The formulas use a finite-difference Hessian of the exact marginal likelihood, as does R metafor (`numDeriv`), and the three
> values disagree with one another, so which is nearest the true observed information is not known.
>
> **QE_Wld.** Found on 2026-11-14 by the case `glmm.heterogeneity.qe-wld.cm.el.fe.bcg` (decision D2); the user answered *follow the reference* on
> 2026-11-14. For `dat.bcg`, `OR`, `CM.EL`, `FE`, `QE_Wld` is 191.71241948 in metafopy, 163.08947856 in the formulas [...] and 268.5989693 in R metafor
> 5.2-1. The same statistic agrees between the formulas, metafopy and R metafor for `CM.AL` (147.906) and `UM.FS` (163.165), and `QE_LRT` of `CM.EL` is
> 176.8738393 in all three.

### Measured on 2026-11-15 while writing this spec

Main checkout at `c04fae2`, Python with metafopy's own `neg_loglik`, R 4.6.2 with metafor 5.2-1, numDeriv 2016.8.1.1 and BiasedUrn 2.0.12 from
`.audit/Rlib`. Each figure is the standard error of the pooled log odds ratio at metafopy's estimates.

| Data | Fit | metafopy now | analytic | numDeriv `r` 4 / 8 / 12 / 16 on metafopy's objective | R, `hessianCtrl = list(r = 4)` |
|---|---|---|---|---|---|
| nielweise2007 | FE | 0.22403812206 | 0.22403481656 | 0.22403481655 / ...649 / ...83535 / 0.22403812206 | 0.22403481294 |
| collins1985a | FE | 0.09940438923 | 0.09955789399 | 0.09955789399 / ...487 / ...802048 / 0.09940438923 | 0.09955789204 |
| jackson2018 | FE | 0.25175750395 | 0.25175350287 | 0.25175350287 / ...311 / ...50853 / 0.25175750395 | 0.25175350287 |
| nielweise2007 | ML | 0.35126655007 | 0.35104032933 | 0.35104032929 / ...3093 / ...91495 / 0.35126996866 | 0.35104033233 |
| collins1985a | ML | 0.13635526517 | 0.13638547727 | 0.13638547728 / ...7841 / ...33600 / 0.13635526517 | 0.13638546146 |
| jackson2018 | ML | 0.88813348998 | 0.88810229648 | 0.88810229647 / ...29484 / ...34825 / 0.88813348998 | 0.88810228110 |

"Analytic" is the sum of conditional variances (`FE`) and, for `ML`, the observed information from posterior moments (see Implementation) by
100- and 200-node Gauss-Hermite quadrature about each study's posterior mode (identical to 11 digits). A central second difference of the exact marginal
log-likelihood converges to the same values as the step falls from 1e-1 to 3e-4 (nielweise2007: 0.35096580, 0.35103362, 0.35103958, 0.35104026,
0.35104032, 0.35104033) and drifts again below 1e-4. `dat.bcg` `QE_Wld`: analytic `sum v_i (psi_i - psibar)^2` over the per-study conditional
estimates 163.08947856; metafopy with `hessianCtrl` `r` 4 163.08948319, `r` 8 163.09019291; R with `r` 4 and `dnchgcalc = "dnoncenhypergeom"`
163.08946351, R's default 268.59896931.

**Conclusion.** The formulas' values are the true observed information to about 1e-9. metafopy's and R's departures come from the same cause: numDeriv
halves the step 15 times from `0.1 |x|` with `r = 16`, and Richardson extrapolation over those steps amplifies rounding; R adds BiasedUrn's truncation.
The `ML` question of D1 is settled by measurement; whether metafopy follows it is the decision "The `ML` standard error" below.

## Module location

- `src/metafopy/core/nchg.py`: new `conditional_moments`, `fixed_information` and `marginal_information`.
- `src/metafopy/models/glmm_fit.py`: `exact_conditional` uses them; `saturated_wald` takes an information matrix; `GLMMControl.hesspack` docstring.
- `src/metafopy/models/rma_glmm.py`: `make_control` and the docstring paragraph on the `CM.EL` Hessian.
- Tests: new `src/tests/test_nchg.py`; `src/tests/correctness/cases/glmm.toml`; `src/tests/test_rma_glmm.py` and the other parity modules named below.
- Records: `src/audit/data/known.toml`, `docs/differences.md`, `docs/differences-notes.md`, `docs/guide/binary.md`.

## API

```python
def conditional_moments(tables: Tables, i: int, log_or: float) -> tuple[float, float]:
    """Mean and variance of ``ai`` of table ``i`` under the untruncated non-central hypergeometric distribution with log odds ratio ``log_or``
    (clamped to +-log(1e12) as ``log_probability`` is; the variance is 0 outside the clamp, where the clamped likelihood is flat)."""


def fixed_information(beta: FloatArray, tables: Tables, X: FloatArray) -> FloatArray:
    """``sum_i Var_i(ai | margins, x_i' beta) x_i x_i'``: the observed (and expected) information of the fixed-effects exact conditional likelihood
    (van Houwelingen et al. 1993, doi:10.1002/sim.4780122405; Breslow 1981). ``p x p``, the Hessian of ``neg_loglik(random=False)``."""


def marginal_information(parms: FloatArray, tables: Tables, X: FloatArray, control: IntegrationControl) -> FloatArray:
    """Hessian of ``neg_loglik(parms, tables, X, True, control)`` in ``(beta, log(tau2))``, ``(p + 1) x (p + 1)``, from the posterior moments
    ``E[(theta - mu_i)^m]``, ``m = 0..4``, of each study's log odds ratio (derivatives of the normal mixing density taken under the integral).

    Raises:
        metafopy.errors.EstimationError: an integral is not finite or a study's marginal likelihood is zero.
    """
```

`saturated_wald(par, information, p)` keeps its signature; it receives the analytic information instead of a numerical Hessian.

## Implementation

**FE and saturated fits.** `exact_conditional` replaces `h_fe = hess(...)` by `fixed_information(x_fe, tables, spec.X)` and `h_qe` by
`fixed_information(x_qe, tables, X_qe)`. `invert_hessian` and `saturated_wald` stay as they are, including the zero-column rule (a study whose
conditional variance is exactly 0). The `ML` branch with `tau2 < tau2tol` uses the same `FE` information. The analytic information never uses the
truncating `dFNCHypergeo` density: it is the information of the exact likelihood.

**ML fit.** With `s = tau2`, `d = theta - mu_i`, `e_m = E[d^m]` under the posterior `g_i(theta) phi(theta; mu_i, s) / L_i` (`g_i` the smooth
`log_probability` density, clamped as `study_loglik` clamps it):

- scores `S_mu = e1 / s`, `S_s = e2 / (2 s^2) - 1 / (2 s)`;
- `H_mumu = e2 / s^2 - 1 / s - S_mu^2`, `H_mus = e3 / (2 s^3) - 3 e1 / (2 s^2) - S_mu S_s`, `H_ss = e4 / (4 s^4) - 3 e2 / (2 s^3) + 3 / (4 s^2) - S_s^2`;
- in `phi = log(tau2)`: `H_muphi = s H_mus`, `H_phiphi = s^2 H_ss + s S_s` (the score term is kept, so the matrix is exact away from the optimum too);
- summed over studies: `H_bb = sum x_i x_i' H_mumu`, `H_bphi = sum x_i H_muphi`, `H_phiphi = sum H_phiphi`; `marginal_information` returns minus that.

Each `e_m` is one `scipy.integrate.quad` over `(-inf, inf)` with the limits and `subdivisions` of `study_loglik`, after `L_i` is integrated first and
the integrand divided by it, so the absolute tolerance is relative to the posterior mass. The tolerance is `min(int_rel_tol, 1e-10)`, the `tight`
control `exact_conditional` already builds: the information is reported to full accuracy whatever tolerance the search used. `vb`, `se_tau2` and
the `tau2` interval come from `invert_hessian(marginal_information(...))` as they come from `h_ml` now. Five integrals per study replace about a hundred
likelihood evaluations of numDeriv with `r = 16`, so the fit gets faster.

**When the numerical Hessian is kept.** Settled by the decision "The Hessian control entries" below; the recommended rule: `exact_conditional` takes
the numerical path of today (unchanged code) only when the caller's `control` names `hesspack` or `hessianCtrl`, or selects the truncating density
(`dnchgcalc='dFNCHypergeo'` or a `dnchgprec`). `GLMMControl.hesspack` defaults to `None` (analytic) instead of `'numDeriv'`; `make_control` keeps
an explicit value. `res.control` then lacks `hesspack` where R lists `'numDeriv'`: the audit task checks whether a `field-diff` appears and adds a
`known.toml` entry if so.

### Tests

- `src/tests/test_nchg.py`: `fixed_information` and `marginal_information` against a central second difference of `neg_loglik` (tight control,
  step 3e-4; the measurement above agrees to about 1e-9) on `dat.nielweise2007`, an intercept-only and a one-moderator design, to 1e-7 relative;
  `conditional_moments` against `scipy.stats.nchypergeom_fisher` mean and variance; the clamp (variance 0 at `log_or = 40`).
- The seven correctness cases (the six `.se` cases and `glmm.heterogeneity.qe-wld.cm.el.fe.bcg`) lose `status = "mismatch"` and `reason` and pass;
  the comment at the top of `glmm.toml` stops calling the `CM.EL` standard errors mismatches. Each is confirmed failing on `main` first (it is a strict
  expected failure there).
- R parity fixtures are never regenerated: they keep R's values. `src/tests/test_rma_glmm.py` (`el_*`, tolerance `LOOSE` 5e-2, `NOISY_QE_WLD`) and
  `test_glmm_extras.py`, `test_glmm_packages.py`, `test_glmm_links.py`, `test_audit4_vcov.py`, `test_audit4_robust.py`, `test_audit5_numerics.py` are run;
  each comparison of a `CM.EL` `se`, `vb`, `zval`, `ci`, `pval`, `se_tau2` or `QE_Wld` that fails is marked differing with the module's own mechanism
  (its tolerance table or skip set), with a comment naming the `docs/differences.md` entry. The docstring of `test_rma_glmm.py` says metafopy reports
  the analytic information. `test_glmm_hessianctrl_changes_the_hessian` and `test_glmm_hesspack_and_hessianctrl` stay valid under the recommended rule.

### Records

- `src/audit/data/known.toml`: the `*/glmm_cm*` `value-diff` and `print-diff` reasons say metafopy reports the analytic information and R a numDeriv
  `r = 16` Hessian (figures above); the `warning-missing` entry (R cannot invert, metafopy can) keeps its match with the reason reworded; the
  `construct/glmm_cm/control=*` `warning-extra` entry is removed if the audit no longer reports metafopy's warning; the `CM.EL` figure in the `UM.FS`
  print-diff entry (Wld 268.5990 against 191.7124) is updated. Every other `CM.EL` mention in the file (`grep -n 'CM.EL' src/audit/data/known.toml`) is
  checked. Then `bin/metafor-audit.sh --only 'construct/glmm_cm*'` and `bin/metafor-audit.sh --recall` (items `a4-glmm-cmel-optimizers` and
  `a5-glmm-cmel-se` must still be found).
- `docs/differences.md`: the section "rma_glmm with CM.EL: standard errors" moves from "Optimiser stop points" to "Where metafor departs from its
  reference", rewritten as "rma_glmm with CM.EL: standard errors and QE_Wld": metafopy uses the analytic information, the cause of R's value, the
  correctness cases that check it, the `control` entries that restore the numerical Hessian, and a compare block whose Python output is regenerated.
- `docs/differences-notes.md` (the `rma_glmm` `CM.EL` notes on standard errors and `QE_Wld` with the optimisers), `docs/guide/binary.md` (the `hesspack`
  and `dnchgcalc` lines), the `rma_glmm` docstring and the `GLMMControl` docstring say what the default now is.

### Environment for tasks

Worktrees have no `.audit/Rlib`. Every audit run or R call sets `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` (R calls:
`LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript ...`). The audit slice runs in the background.

## Error handling

- `marginal_information` raises `metafopy.errors.EstimationError` when a moment integral is not finite or `L_i` is 0; `exact_conditional` lets it
  propagate as the numerical path's `invert_hessian` failure does today ("Cannot invert the Hessian for the ML model.").
- `invert_hessian` keeps its Cholesky-then-QR fall back and warning. The analytic `FE` and saturated information is positive semi-definite by
  construction, so the "Cannot invert the Hessian for the saturated model." warning should now appear only where a design is singular.

## Settings

`control` entries of `rma_glmm(model='CM.EL')` after the change (recommended rule):

| Entry | Type | Default | Effect |
|---|---|---|---|
| `hesspack` | `str \| None` | `None` | `None`: analytic information. `'numDeriv'`, `'pracma'`, `'calculus'`: today's numerical Hessian. |
| `hessianCtrl` | `dict` | `{}` | Non-empty: numerical Hessian with these settings (numDeriv `r` 16 unless given). |
| `dnchgcalc` | `str \| None` | `None` | `'dFNCHypergeo'`: numerical Hessian of the truncated density, as now. |
| `dnchgprec` | `float \| None` | `None` | Given: as `dnchgcalc='dFNCHypergeo'`. |
| `int_rel_tol` | `float` | `eps^0.25` | Search tolerance; the information integrals use `min(int_rel_tol, 1e-10)`. |

## External dependencies

NumPy and SciPy (`scipy.integrate.quad`, `scipy.special`), already used by `metafopy.core.nchg`. Tests use `scipy.stats.nchypergeom_fisher`. No new
dependency.

## Decisions

Answered on 2026-11-15 in `agents/reviews/order-numerical-fixes-specs.md` (D1, D2): the `ML` standard error follows the reference, so the ML part
(`marginal_information`) is built; the Hessian control entries are opt-in numerical, the recommended rule above. The options as they were put are kept
below as the record.

### The `ML` standard error

**What is decided:** whether the `ML` standard error (and `se_tau2`, its interval) follows the analytic information, as the `FE` one does.
**Background:** the user answered *investigate* on 2026-11-14 because the three values disagreed. The measurement of 2026-11-15 above shows the
formulas' value is the observed information to about 1e-9 and metafopy's is off by numDeriv's `r = 16` (6.4e-4 relative on `dat.nielweise2007`).
**Affects:** the three `ML` cases, `marginal_information`, the `ML` parity comparisons.
**Options:** *follow the reference*: implement the ML part; the three cases pass. *Keep metafor*: drop the ML part; the cases switch to a recorded
difference with this measurement as the reason, and `docs/differences.md` says why.
**Recommendation:** *follow the reference*: the measurement removes the doubt behind *investigate*, and it matches the `FE` answer.

### The Hessian control entries

**What is decided:** what `hesspack`, `hessianCtrl`, `dnchgcalc` and `dnchgprec` do once the information is analytic.
**Background:** metafor documents them as choosing the numerical Hessian; metafopy already departs from R's default density (`dnchgcalc`) and keeps
R's behaviour behind a `control` entry.
**Affects:** the default `hesspack`, two tests of `test_audit5_numerics.py`, the user guide.
**Options:** *opt-in numerical*: analytic by default, today's numerical Hessian when an entry is given (keeps metafor's meaning of each entry; two code
paths). *Always analytic*: the entries are validated and kept in `res.control` but change nothing; the numerical code and the truncating density's
Hessian use go, and the two tests are rewritten to expect equal results.
**Recommendation:** *opt-in numerical*, so a caller who asks for metafor's Hessian gets it, as with `dnchgcalc` today.

## Operator stops

None inside the task list: both decisions were answered at the spec review on 2026-11-15. The ML outcome is no longer unknown, so no task waits on an
investigation. If the review answers *keep metafor* for the `ML` standard error, the task list leaves out `marginal_information` and its tasks.
