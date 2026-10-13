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
reference fixtures. `escalc`, the diagnostics (`influence`, `leave1out`,
`cumul`, `permutest`, `profile`, `gosh`), the funnel-plot asymmetry tests
(`regtest`, `ranktest`, `trimfill`, `fsn`, `hc`, `tes`) and the plots in
`metafopy.plot` are available too. For binary and count data, `rma_mh`
(Mantel-Haenszel), `rma_peto` (Peto's method) and `rma_glmm` (generalized linear
mixed-effects models: `UM.FS`, `UM.RS`, `CM.AL`, `CM.EL` for `OR` and `IRR`,
and `PLO`/`IRLN` for proportions and rates) are available. `rma_mh` and
`rma_peto` match metafor to 1e-8. `rma_glmm` matches the fixed-effects and
`IRR` fits to machine precision and the mixed models to about 1e-3 (R's
`lme4::glmer` stops at a loose optimiser tolerance; metafopy fits to the same or a
slightly better optimum). `rma_glmm` does not port the experimental measures `PR`,
`RR`, `RD` and `PLN`.

`rma_mv` fits multivariate and multilevel models (all variance-covariance structures
of metafor, known correlation matrices via `R=`, `sparse=True`, `dfs='contain'`) and
matches metafor 5.2-1 to about 1e-5 (variance components on flat likelihoods to about
1e-3). `predict`, `anova` and profile-likelihood `confint` work for `rma_mv`. `vcalc`,
`rcalc`, `bldiag` and `blsplit` build the sampling variance-covariance matrices, and
`robust` gives cluster-robust inference for `rma_uni` and `rma_mv` fits, either with
metafor's default sandwich estimator or, with `club_sandwich=True`, with the CR0-CR3
estimators, Satterthwaite or saddlepoint tests and Wald tests (chi-sq, Naive-F, HTA, HTB, HTZ, EDF, EDT) of the R package
[clubSandwich](https://jepusto.github.io/clubSandwich/). `rma_mv` also takes `optbeta`
(with `beta`, `lambda1` and `lambda2`), `vccon`, `cvvc` (Hessian and covariance matrix of the
variance components, from numDeriv's Richardson method) and `restart` (from a previous fit);
it does not port `verbose`.

`rma_uni(..., scale=~ ...)` fits location-scale models (`RMALSResult`, Viechtbauer &
Lopez-Lopez, 2022) with log or identity link, and `predict`, `confint`, `profile`,
`anova` (Wald and likelihood-ratio tests), `vif`, `blup`, `ranef`, `rstandard` and
`permutest` (including the exact test) work for them. Fits match metafor to about 1e-6
(scale coefficients about 1e-4 where the likelihood is flat). `sim=` in `vif` is not
ported.

`selmodel` fits the selection models of Vevea & Hedges (1995) and the step-function and
parametric variants of metafor (`beta`, `halfnorm`, `negexp`, `logistic`, `power`, `negexppow`,
`stepfun`, `stepcon`, `trunc`) to an `rma_uni` fit, with Wald and likelihood-ratio tests,
`predict`, `confint`, `profile`, `anova` and `plot_selmodel`. Estimates match metafor to
about 1e-7 and standard errors to 1e-4 (numerical Hessian); `stepcon` is not validated
against R and `confint`/`profile` ignore metafor's parallel options.

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
pytest --mpl   # --mpl also compares the plots with the baseline images (Linux only)
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

GPL-2.0-or-later, the same as metafor (GPL >= 2). metafopy ports its
algorithms, so it is distributed under the same license. The bundled
datasets come from the R package
[metadat](https://cran.r-project.org/package=metadat) (GPL >= 2).
