# Other-software correctness cases for the location-scale model of rma_uni from the R package glmmTMB, written to
# src/tests/correctness/cases/location_scale_other.toml.
#
# Usage (from the repository root), with glmmTMB on R's library path (the audit's pinned library through R_LIBS; nothing is installed into it) and
# METAFOPY_CORRECTNESS_RLIB naming the correctness library, as for the other generators:
#   METAFOPY_CORRECTNESS_RLIB=<library> R_LIBS=<audit library> LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software_ls.R
#
# What is independent of metafor: glmmTMB maximises the likelihood by its own code (TMB, Laplace approximation, exact here because the model is
# Gaussian). The model of Viechtbauer and Lopez-Lopez (2022) is y_i ~ N(x_i beta, v_i + tau2_i) with log(tau2_i) = z_i alpha, which glmmTMB
# expresses as
#   - the known sampling variances v_i as a random effect with a fixed covariance matrix, equalto(0 + obs | g, diag(v)): one level per study, one
#     group, no parameter estimated;
#   - the heterogeneity as the residual variance, through dispformula = ~ z.
# glmmTMB has no identity link for the dispersion, so only the log-link cases exist.
#
# Conventions, each read from glmmTMB's documentation (?glmmTMB, argument dispformula; ?sigma.glmmTMB, section gaussian):
#   - for the Gaussian family the dispersion is sigma, the residual standard deviation, on a log link: log(sigma_i) = z_i delta. So
#     tau2_i = sigma_i^2 = exp(2 z_i delta) and alpha = 2 delta; alpha is compared after that factor of 2.
#   - glmmTMB's REML log-likelihood omits the term (1/2) log|X'X| that metafor's REML log-likelihood keeps (glmmTMB integrates the fixed effects out
#     with the Laplace approximation, which gives -(1/2) log|X'WX|, and no more). The generator adds (1/2) log|X'X| to glmmTMB's value before it is
#     written, so the case compares metafor's convention; the ML log-likelihood needs no change.
#   - the cases state a tolerance looser than the derived cases, because the two programs' optimisers stop at different points of a flat likelihood.

GLMMTMB_VERSION <- "1.1.15.2"
LS_TOLERANCE <- 1e-4

lib <- Sys.getenv("METAFOPY_CORRECTNESS_RLIB")
if (!nzchar(lib)) stop("METAFOPY_CORRECTNESS_RLIB is not set: name the R library for the correctness generators; glmmTMB ", GLMMTMB_VERSION, " comes from R_LIBS")
.libPaths(c(lib, .libPaths()))
if (!requireNamespace("glmmTMB", quietly = TRUE)) stop("glmmTMB is not installed on the library path (set R_LIBS to the audit library)")
if (packageVersion("glmmTMB") != package_version(GLMMTMB_VERSION)) {
  stop("glmmTMB ", packageVersion("glmmTMB"), " found; this script is pinned to ", GLMMTMB_VERSION)
}
suppressPackageStartupMessages(library(glmmTMB))

g17 <- function(x) sprintf("%.17g", unname(as.vector(x)))
r_version <- paste(R.version$major, R.version$minor, sep = ".")
today <- format(Sys.Date())
citation <- sprintf(
  "R package glmmTMB %s (Brooks ME et al. (2017). glmmTMB balances speed and flexibility among packages for zero-inflated generalized linear mixed modeling. R Journal 9:378-400), run on R %s.",
  GLMMTMB_VERSION, r_version
)

toml_value <- function(v) {
  if (is.list(v)) {
    return(paste0("{ ", paste(sprintf("%s = %s", names(v), vapply(v, toml_value, "")), collapse = ", "), " }"))
  }
  if (is.character(v)) return(sprintf("\"%s\"", v))
  g17(v)
}

cases <- character(0)
add_case <- function(id, dataset, args, expect, locator, setup = NULL) {
  lines <- c(
    "[[case]]",
    sprintf("id = \"%s\"", id),
    "kind = \"other\"",
    "function = \"rma_uni\"",
    sprintf("dataset = \"%s\"", dataset)
  )
  if (!is.null(setup)) lines <- c(lines, sprintf("setup = %s", toml_value(list(escalc = setup))))
  lines <- c(
    lines,
    sprintf("args = %s", toml_value(args)),
    sprintf("expect = { %s }", paste(sprintf("\"%s\" = %s", names(expect), vapply(expect, g17, "")), collapse = ", ")),
    sprintf("tolerance = %s", g17(LS_TOLERANCE)),
    "",
    "[case.source]",
    sprintf("citation = \"%s\"", citation),
    sprintf("locator = \"%s\"", locator),
    sprintf("read = %s", today),
    "url = \"https://cran.r-project.org/package=glmmTMB\"",
    ""
  )
  cases <<- c(cases, lines)
}

expect_vector <- function(prefix, x) setNames(as.list(x), sprintf("%s[%d]", prefix, seq_along(x) - 1L))

fit_glmmtmb <- function(d, moderator, reml) {
  d$obs <- factor(seq_len(nrow(d)))
  d$g <- factor(1)
  v <- diag(d$vi)
  dimnames(v) <- list(levels(d$obs), levels(d$obs))
  m <- glmmTMB(
    reformulate(c(moderator, "equalto(0 + obs | g, v)"), response = "yi"), dispformula = reformulate(moderator), data = d, REML = reml,
    control = glmmTMBControl(optCtrl = list(iter.max = 5000, eval.max = 5000))
  )
  if (m$fit$convergence != 0 || !isTRUE(m$sdr$pdHess)) stop("glmmTMB did not converge for ", moderator, ", REML = ", reml)
  X <- model.matrix(reformulate(moderator), d)
  list(
    beta = fixef(m)$cond, alpha = 2 * fixef(m)$disp, tau2 = exp(2 * as.vector(X %*% fixef(m)$disp)),
    ll = as.numeric(logLik(m)) + if (reml) 0.5 * log(det(crossprod(X))) else 0
  )
}

bcg <- metadat::dat.bcg
bcg <- cbind(bcg, metafor::escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg)[, c("yi", "vi")])
datasets <- list(
  bd = list(data = metadat::dat.bangertdrowns2004, name = "dat.bangertdrowns2004", moderator = "imag", setup = NULL),
  bcg = list(
    data = bcg, name = "dat.bcg", moderator = "ablat",
    setup = list(measure = "RR", ai = "tpos", bi = "tneg", ci = "cpos", di = "cneg")
  )
)

for (key in names(datasets)) {
  spec <- datasets[[key]]
  for (method in c("ML", "REML")) {
    f <- fit_glmmtmb(spec$data, spec$moderator, reml = method == "REML")
    mods <- paste("~", spec$moderator)
    add_case(
      sprintf("other.ls.fit.%s.log.%s", method, key), spec$name,
      list(yi = "yi", vi = "vi", mods = mods, scale = mods, method = method, link = "log"),
      c(
        expect_vector("beta", f$beta), expect_vector("alpha", f$alpha), expect_vector("tau2", f$tau2),
        # fit_stats is a frame with rows ll, dev, AIC, BIC, AICc and columns ML, REML; flattened by row, [0] is the ML and [1] the REML log-likelihood
        setNames(list(f$ll), if (method == "REML") "fit_stats[1]" else "fit_stats[0]")
      ),
      sprintf(
        "glmmTMB(yi ~ %s + equalto(0 + obs | g, diag(vi)), dispformula = ~ %s, REML = %s): fixef cond, 2 * fixef disp (sigma is a standard deviation), exp(2 * Z delta), logLik%s",
        spec$moderator, spec$moderator, if (method == "REML") "TRUE" else "FALSE", if (method == "REML") " + log(det(X'X)) / 2" else ""
      ),
      setup = spec$setup
    )
  }
}

header <- c(
  "# Other-software cases for the location-scale model of rma_uni from R glmmTMB, written by src/tests/correctness/r/other_software_ls.R; do not",
  "# edit by hand.",
  "#",
  "# glmmTMB fits the log-link model with the known sampling variances as a fixed-covariance random effect and the heterogeneity as the residual",
  "# dispersion; it has no identity link. alpha is twice glmmTMB's dispersion coefficients (its dispersion is a standard deviation), and its REML",
  "# log-likelihood lacks the term log|X'X| / 2 that metafor keeps, which the generator adds back.",
  ""
)
out <- file.path("src", "tests", "correctness", "cases", "location_scale_other.toml")
writeLines(c(header, cases), out)
cat("wrote", out, "\n")
