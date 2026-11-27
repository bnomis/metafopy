# metafopy

Meta-analysis in Python: a re-implementation of the R package
[metafor](https://www.metafor-project.org/).

**Status:** work in progress.

[Documentation](https://bnomis.github.io/metafopy/)

## Example

```python
import numpy
import metafopy

dat = metafopy.datasets.load('dat.hackshaw1998')  # already has yi / vi columns
res = metafopy.rma(data=dat, method='REML')
print(res)
print(metafopy.confint(res))
print(metafopy.predict(res, transf=numpy.exp))
```

## Building

Requires Python >= 3.14.7, < 3.15.

```bash
git clone https://github.com/bnomis/metafopy
cd metafopy
uv sync
source bin/activate.sh
pytest --mpl   # --mpl also compares the plots with the baseline images (Linux only)
mypy
ruff check .
```

## Building the docs

The user manual is in `docs/` (for example the installation page, tutorial, user guide, worked analyses, a guide for users coming from R, the function reference, the differences from metafor and a showcase notebook).

Build and browse it with:

```bash
uv pip install -r docs/requirements.txt
mkdocs serve
```

`pytest src/tests/test_docs_examples.py` runs every Python example in the manual.

## Validation against R

The publication that defines each method is the first numerical oracle; R metafor is the second. `src/tests/r/` holds the R scripts that generate JSON fixtures in `src/tests/reference/` (`generate_fixtures.R` is one of them; currently metafor 5.2-1, metadat 1.6-0). The fixtures are committed, so the normal test run needs no R.

To regenerate them:

```bash
LC_ALL=C.UTF-8 Rscript src/tests/r/export_datasets.R     # refresh bundled datasets
LC_ALL=C.UTF-8 Rscript src/tests/r/generate_fixtures.R   # refresh src/tests/reference/*.json
```

or reproducibly with the pinned image in `src/tests/r/Dockerfile`:

```bash
docker build -t metafopy-r src/tests/r
docker run --rm -v "$PWD":/work -w /work metafopy-r Rscript src/tests/r/generate_fixtures.R
```

## License

GPL-2.0-or-later, the same as metafor (GPL >= 2).

The bundled datasets come from the R package [metadat](https://cran.r-project.org/package=metadat) (GPL >= 2).
