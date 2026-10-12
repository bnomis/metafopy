# metafopy

Meta-analysis in Python: a re-implementation of the R package
[metafor](https://www.metafor-project.org/) by Wolfgang Viechtbauer.

metafopy aims for feature and numerical parity with metafor's model-fitting
core (`escalc`, `rma.uni`, `rma.mh`, `rma.peto`, `rma.glmm`, `rma.mv`,
`selmodel`, location-scale models) and its diagnostics and plots. Function and
argument names follow metafor, with `.` replaced by `_` (`rma_uni`, `rma_mv`).

**Status:** pre-alpha. Phase 0 (foundations) is in place: input handling,
formula support, bundled datasets and the R reference harness. Model fitting
arrives in phase 1.

## Install (development)

```bash
uv sync
uv run pytest
```

## Validation against R

R metafor is the numerical reference. `tests/r/` holds the R scripts that
generate JSON fixtures in `tests/reference/` (currently metafor 5.2-1,
metadat 1.6-0). The fixtures are committed, so the normal test run needs no R.
To regenerate them:

```bash
Rscript tests/r/export_datasets.R     # refresh bundled datasets
Rscript tests/r/generate_fixtures.R   # refresh tests/reference/*.json
```

or reproducibly with the pinned image in `tests/r/Dockerfile`:

```bash
docker build -t metafopy-r tests/r
docker run --rm -v "$PWD":/work -w /work metafopy-r Rscript tests/r/generate_fixtures.R
```

## License

GPL-3.0-or-later. metafor is licensed GPL (>= 2) and metafopy ports its
algorithms, so it is distributed under a compatible license. The bundled
datasets come from the R package
[metadat](https://cran.r-project.org/package=metadat) (GPL >= 2).
