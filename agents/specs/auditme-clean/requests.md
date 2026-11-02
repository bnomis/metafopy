# Requests for the shared auditme rules

Sites where a rule gets the code wrong, accepted in the project with a marker.
The operator raises these in the shared auditme skill's `requests.md`.

Updated 2026-11-02 after pinning agents `49dab9d`: the markers the refined
rules no longer need were removed, and their entries with them. Each entry
left here still has a hit without its marker.

## raises-section-drift

- Rule: a callee's `Raises:` entry is attributed to the caller although the
  call never takes the raising path (for example `check_class` called without
  `notav`, or `make_link` called with a fixed name). The checker should
  follow the argument values that decide whether the error can arise, or
  skip callees whose raise depends on an argument.
  Sites: `src/metafopy/methods/emm_grid.py` (`contrast`),
  `src/metafopy/methods/anova.py` (`wald_between`),
  `src/metafopy/methods/profile.py` (`profile_rma_uni`),
  `src/metafopy/methods/rcalc.py` (`rcalc_list`),
  `src/metafopy/methods/selmodel.py` (`anova`),
  `src/metafopy/plot/rma.py` (`plot_rma`, `qqnorm`'s envelope).
- Rule: the caller passes a value of a type the callee's dispatch error does
  not apply to (`rows_of` raises `ArgumentError` for an unsupported class;
  `fitted_values` and the `predict_*` functions only receive supported ones).
  Sites: `src/metafopy/methods/fitted.py`, `ls_predict.py`, `predict.py`.
- Rule: the caller calls the callee only inside an `is_rma(x)` branch, which
  rules out the `ArgumentError` the callee raises for a non-model. The
  checker should not attribute that error across such a guard.
  Sites: `src/metafopy/bias/ranktest.py` (`ranktest`), `src/metafopy/bias/tes.py` (`tes`).
- Rule: a callee's `Raises:` entry is attributed to a function that receives
  an argument (`measure`) already checked against a supported list by every
  caller, so the callee's argument, support and internal errors cannot arise.
  Site: `src/metafopy/models/tables.py` (`prepare_tables`, calling `escalc`).

## raises-section-drift (re-raise)

- Rule: an error caught in a nested function and re-raised for one branch
  (`if m1 == 'blue': raise`) is reported as never reaching the caller. The
  checker should count a bare `raise` in an `except` as a raise of the
  caught types.
  Site: `src/metafopy/effects/fivenum.py` (`conv_fivenum`).

## lockfile-drift

- Done 2026-11-01 in agents `8b43d9b`, not a request: the rule ran
  `uv lock --check` under the user's uv config, so a global relative
  `exclude-newer` the lock was not made with reported drift locally only.
  It now repeats the check without the user config when uv names that
  setting as its reason. This repository pins a later commit that includes it.
