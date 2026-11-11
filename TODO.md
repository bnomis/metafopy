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
