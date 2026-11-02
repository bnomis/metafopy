# Requests for the shared auditme rules

Sites where a rule gets the code wrong, accepted in the project with a marker.
The operator raises an open one in the shared auditme skill's `requests.md`.

Updated 2026-11-02 after pinning agents `49dab9d`: the markers the refined
rules no longer need were removed, and their entries with them. Each entry
left here still has a hit without its marker.

No entry below is an open request. The shared rules' false-positives spec
(agents `agents/specs/auditme-false-positives/spec.md`, decision and
amendment 1.1 of 2026-11-01) narrowed the `raises-section-drift` carve-out
to optional arguments the call leaves unset, and kept fixed names,
pre-checked values, dispatch types and the caller's own guards as hits, for
accept markers or the Phase B review. The entries record why each marker
stays.

## raises-section-drift (kept by decision, 2026-11-01)

- Pattern: a callee's `Raises:` entry is attributed to the caller although the
  call never takes the raising path (for example `check_class` called without
  `notav`, or `make_link` called with a fixed name). The checker does not
  follow argument values, by decision.
  Sites: `src/metafopy/methods/emm_grid.py` (`contrast`),
  `src/metafopy/methods/anova.py` (`wald_between`),
  `src/metafopy/methods/profile.py` (`profile_rma_uni`),
  `src/metafopy/methods/rcalc.py` (`rcalc_list`),
  `src/metafopy/methods/selmodel.py` (`anova`),
  `src/metafopy/plot/rma.py` (`plot_rma`, `qqnorm`'s envelope).
- Pattern: the caller passes a value of a type the callee's dispatch error does
  not apply to (`rows_of` raises `ArgumentError` for an unsupported class;
  `fitted_values` and the `predict_*` functions only receive supported ones).
  Dispatch types stay hits, by decision.
  Sites: `src/metafopy/methods/fitted.py`, `ls_predict.py`, `predict.py`.
- Pattern: the caller calls the callee only inside an `is_rma(x)` branch,
  which rules out the `ArgumentError` the callee raises for a non-model. The
  checker does not read the caller's guards, by decision.
  Sites: `src/metafopy/bias/ranktest.py` (`ranktest`), `src/metafopy/bias/tes.py` (`tes`).
- Pattern: a callee's `Raises:` entry is attributed to a function that
  receives an argument (`measure`) already checked against a supported list
  by every caller, so the callee's argument, support and internal errors
  cannot arise. Pre-checked values stay hits, by decision.
  Site: `src/metafopy/models/tables.py` (`prepare_tables`, calling `escalc`).

## lockfile-drift

- Done 2026-11-01 in agents `8b43d9b`, not a request: the rule ran
  `uv lock --check` under the user's uv config, so a global relative
  `exclude-newer` the lock was not made with reported drift locally only.
  It now repeats the check without the user config when uv names that
  setting as its reason. This repository pins a later commit that includes it.
