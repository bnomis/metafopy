# Requests for the shared auditme rules

Sites where a rule gets the code wrong, accepted in the project with a marker.
The operator raises these in the shared auditme skill's `requests.md`.

## raises-section-drift

- Rule: a callee's `Raises:` entry is attributed to the caller although the
  call never takes the raising path (for example `check_class` called without
  `notav`, or `make_link` called with a fixed name). The checker should
  follow the argument values that decide whether the error can arise, or
  skip callees whose raise depends on an argument.
  Sites: `src/metafopy/methods/coef_summary.py` (`coef_summary`),
  `src/metafopy/methods/emm_grid.py` (`contrast`),
  `src/metafopy/methods/anova.py` (`wald_between`),
  `src/metafopy/methods/profile.py` (`profile_rma_uni`),
  `src/metafopy/methods/rcalc.py` (`rcalc_list`),
  `src/metafopy/methods/selmodel.py` (`anova`),
  `src/metafopy/plot/llplot.py` (`llplot`, `escalc` with the fixed measure
  `'OR'`), `src/metafopy/plot/rma.py` (`plot_rma`, `qqnorm`'s envelope).
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
- Rule: `non-alphabetical-literal` flags an `__all__` ordered as ruff RUF022
  requires (constants, classes, functions), which the checker's ASCII order
  contradicts. The checker should accept the RUF022 order for `__all__`.
  Sites: `src/metafopy/interactive/data.py`, `src/metafopy/methods/emm_contrasts.py`.

## missing-raises-section

- Rule: a `raise` inside a nested function whose exception the enclosing
  function catches (a `ValueError` subclass caught by the root search) is
  counted as reaching the caller.
  Site: `src/metafopy/methods/permutest.py:501` (`permci_bound`).

## passthrough-shim

- Rule: a wrapper that computes the arguments it passes on is flagged as a
  pass-through because its body is a single `return target(...)`. The
  checker should flag only a wrapper whose arguments are its own parameters
  passed on unchanged.
  Site: `src/metafopy/mfopt.py` (`with_digits`, which computes `digits` and
  `digit_map` for `dataclasses.replace`).

- Rule: `passthrough-shim` also fires on `dash` in `src/metafopy/plot/regplot.py`,
  which passes its one parameter twice (`LTY.get(name, name)`, the name as its
  own default). The checker should treat a wrapper that reuses a parameter as
  the default argument as computing a value, not passing it on.

## missing-raises-section (handled raise)

- Rule: a `raise` inside a nested function whose only caller sits in a
  `try` that catches the error is attributed to the enclosing function. The
  checker should skip a raise caught within the same enclosing function.
  Site: `src/metafopy/effects/fivenum_estimators.py` (`lbfgsb`).

## raises-section-drift (re-raise)

- Rule: an error caught in a nested function and re-raised for one branch
  (`if m1 == 'blue': raise`) is reported as never reaching the caller. The
  checker should count a bare `raise` in an `except` as a raise of the
  caught types.
  Site: `src/metafopy/effects/fivenum.py` (`conv_fivenum`).
