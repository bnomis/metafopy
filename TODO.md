# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## rma_uni raises ZeroDivisionError for an equal-effects fit when every effect is identical

Found on 2026-11-12 by the property tests of the correctness-tests task list (Task 10, location and scale invariance).
`metafopy.rma_uni(yi=[0, 0, 0, 0], vi=[0.1, 0.2, 0.3, 0.4], method='EE')` stops with `ZeroDivisionError` at
`src/metafopy/models/rma_uni.py` in the heterogeneity block (`I2 = max(0.0, 100 * (QE - (k - p)) / QE)`), because `QE` is 0. The same holds for
`FE` and `CE`. R metafor evaluates the same line in floating point, where `(0 - (k - p)) / 0` is `-Inf` and `max(0, -Inf)` is 0, so `I2` is 0 there.

Fix direction: give `I2` the value 0 when `QE` is 0 (as R does), keep `H2 = QE / (k - p)`, and add a regression test with identical effects for each
of `EE`, `FE` and `CE`. Check the random-effects branch and `rma_mh`, `rma_peto` for the same division.

## confint type "ht" of a REML fit centres on the fit's H2, not on H from Q

Found on 2026-11-12 by the correctness tests (case `confint.ht.reml.bcg.none` in `src/tests/correctness/cases/inference.toml`, strict expected failure);
the user answered *follow the reference* on 2026-11-12 (`agents/specs/correctness-tests/mismatches.md`, Answers).
Higgins and Thompson (2002, equations 24 and 25), and Borenstein et al. (2009, equations 16.14 to 16.19), build the interval on `ln H` with H from
`Q = (k - 1) H^2`. For `dat.bcg` log risk ratios the tau2 interval is 0.19972 to 0.47038. The `ht` branch of `src/metafopy/methods/confint.py` centres on
`H2` of the fit (12.856 under REML against `Q / (k - 1)` = 12.686), giving 0.20274 to 0.47703. R metafor 5.2-1 gives the same as metafopy
(`confint.rma.uni` centres on `log(x$H2)`), so with DL, where the two agree, there is no difference.

Fix direction: centre on `H = sqrt(Q / (k - 1))` for every method; flip the case to `status = "pass"` and drop its `reason`; mark the R parity
fixture value `ht_ci` in `src/tests/reference/extras/rma_uni_help.json` (read by `test_confint_types` in `src/tests/test_rma_uni_help.py`) as differing
for the reason; add a `src/audit/data/known.toml` entry so `bin/metafor-audit.sh` accepts the difference; add the section "Where metafor departs from its
reference" to `docs/differences.md` (the heading does not exist yet), naming this interval.

## The correctness harness cannot check exp(beta), prediction intervals or test-based tau2 and I2 intervals

Found on 2026-11-12 by Task 14 of the correctness-tests task list (`agents/tasks/correctness-tests/tasks-1.md`), recorded in
`agents/specs/correctness-tests/mismatches.md` under "Harness limits left for later". `expect` in a case compares a value fetched from the result by
key (`fetch` and `check_case` in `src/tests/correctness/cases.py`), so a value a source prints on another scale or that no single result attribute holds
cannot be a case: the printed pooled ratios of Mantel and Haenszel (1959, R = 10.68) and Yusuf et al. (1985, pooled odds ratios) were left out, and so
were prediction intervals and the test-based tau2 and I2 intervals of Higgins and Thompson (2002) and Borenstein et al. (2009).

Fix direction: add `check` functions (the `"<test module>.<function>"` form `parse_case` already accepts) for a back-transformed estimate, for
`predict` intervals and for the `confint` I2 and H2 bounds, then write the published cases the supplied sources in `/Users/simonb/src/metafopy papers`
support.

## Extend the correctness tests beyond the core models

Found on 2026-11-12 while reviewing `TODO.md`: the correctness-tests spec (`agents/specs/correctness-tests/spec.md`, introduction) covers `escalc`,
`rma_uni`, `rma_mh` and `rma_peto`, and says "Later specs extend the same harness to `rma_mv`, `rma_glmm`, `rma_ls`, the bias methods, `selmodel` and
`robust`." No such spec exists yet, so those functions are checked only against R metafor, not against their defining references.

Fix direction: one spec per area with `/specdev`, each listing its defining references, the sources to fetch or ask the user for, and its coverage items in
`src/tests/correctness/test_coverage.py`; do the harness item above first, since several of these areas report back-transformed or interval values.
