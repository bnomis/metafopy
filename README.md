# metafopy

Meta-analysis in Python: a re-implementation of the R package
[metafor](https://www.metafor-project.org/) by Wolfgang Viechtbauer.

metafopy aims for feature and numerical parity with metafor's model-fitting
core (`escalc`, `rma.uni`, `rma.mh`, `rma.peto`, `rma.glmm`, `rma.mv`,
`selmodel`, location-scale models) and its diagnostics and plots. Function and
argument names follow metafor, with `.` replaced by `_` (`rma_uni`, `rma_mv`).

**Status:** pre-alpha. `rma_uni` (alias `rma`) is available with all of
metafor's standard tau^2 estimators, `predict`, `confint` (Q-profile) and
`anova`; it reproduces metafor 5.2-1 to about 1e-9 relative error across the
reference fixtures. `escalc` arrives next.

```python
import numpy as np
from metafopy import datasets, rma, confint, predict

dat = datasets.load('dat.hackshaw1998')  # already has yi / vi columns
res = rma(data=dat, method='REML')
print(res)
print(confint(res))
print(predict(res, transf=np.exp))
```

Requires Python >= 3.14.7, < 3.15.

## Install (development)

```bash
uv sync
source bin/activate.sh
pytest
mypy
ruff check .
```

## Validation against R

R metafor is the numerical reference. `src/tests/r/` holds the R scripts that
generate JSON fixtures in `src/tests/reference/` (currently metafor 5.2-1,
metadat 1.6-0). The fixtures are committed, so the normal test run needs no R.
To regenerate them:

```bash
Rscript src/tests/r/export_datasets.R     # refresh bundled datasets
Rscript src/tests/r/generate_fixtures.R   # refresh src/tests/reference/*.json
```

or reproducibly with the pinned image in `src/tests/r/Dockerfile`:

```bash
docker build -t metafopy-r src/tests/r
docker run --rm -v "$PWD":/work -w /work metafopy-r Rscript src/tests/r/generate_fixtures.R
```

## License

GPL-3.0-or-later. metafor is licensed GPL (>= 2) and metafopy ports its
algorithms, so it is distributed under a compatible license. The bundled
datasets come from the R package
[metadat](https://cran.r-project.org/package=metadat) (GPL >= 2).
