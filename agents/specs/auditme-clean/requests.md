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
  `src/metafopy/methods/selmodel.py` (`anova`).
- Rule: the caller passes a value of a type the callee's dispatch error does
  not apply to (`rows_of` raises `ArgumentError` for an unsupported class;
  `fitted_values` and the `predict_*` functions only receive supported ones).
  Sites: `src/metafopy/methods/fitted.py`, `ls_predict.py`, `predict.py`.

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
