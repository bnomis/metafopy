# metafopy

Meta-analysis in Python: a re-implementation of the R package
[metafor](https://www.metafor-project.org/).

**Status:** work in progress.

[Documentation](https://bnomis.github.io/metafopy/)

## Example

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

## Documentation

The user manual is in `docs/` (installation, tutorial, user guide, worked analyses, a guide for users coming from R and the function reference).

Build and browse it with:

```bash
uv pip install -r docs/requirements.txt
mkdocs serve
```

`pytest src/tests/test_docs_examples.py` runs every Python example in the manual.

## Install (development)

```bash
uv sync
source bin/activate.sh
pytest --mpl   # --mpl also compares the plots with the baseline images (Linux only)
mypy
ruff check .
```

## Validation against R

R metafor is the numerical reference. `src/tests/r/` holds the R scripts that generate JSON fixtures in `src/tests/reference/` (currently metafor 5.2-1, metadat 1.6-0). The fixtures are committed, so the normal test run needs no R.

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

GPL-2.0-or-later, the same as metafor (GPL >= 2). metafopy ports its algorithms, so it is distributed under the same license. The bundled datasets come from the R package [metadat](https://cran.r-project.org/package=metadat) (GPL >= 2).
