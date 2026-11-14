# Other-software correctness cases for rma_glmm from base R's mantelhaen.test and the R package glmmML, written to
# src/tests/correctness/cases/glmm_other.toml.
#
# Usage (from the repository root), with glmmML installed in a library outside this repository (not the audit's pinned library) and the datasets'
# package (metadat) on R's library path, for example the audit's library through R_LIBS:
#   METAFOPY_CORRECTNESS_RLIB=<library> R_LIBS=<audit library> LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software_glmm.R
#
# What is independent of metafor: stats::mantelhaen.test(exact = TRUE) is the conditional maximum likelihood estimate of a common odds ratio
# (Breslow 1981, via the exact conditional distribution of the sum of the treated-arm events), and glmmML fits random-intercept binomial and Poisson
# models by its own Gauss-Hermite quadrature, with no use of lme4. Conventions:
#   - the exact test's interval is an exact (Cornfield-type) interval, metafopy's is a Wald interval, so only the estimate is compared;
#   - glmmML's quadrature is not adaptive; N_POINTS nodes make it accurate enough to stand for the exact marginal likelihood, and the cases state
#     a tolerance looser than the derived cases for that reason (the optimisers of the two programs stop at different points of a flat likelihood).

GLMMML_VERSION <- "1.1.7"
N_POINTS <- 64L
# metafopy fits the glmmML cases with this many adaptive quadrature points, so its integral is as accurate as glmmML's and the two differ only by
# the optimisers' stopping points (its default of 7 is an approximation, tested by the derived cases)
AGQ <- 41L
GLMM_TOLERANCE <- 3e-4

lib <- Sys.getenv("METAFOPY_CORRECTNESS_RLIB")
if (!nzchar(lib)) stop("METAFOPY_CORRECTNESS_RLIB is not set: name the R library that holds glmmML ", GLMMML_VERSION)
.libPaths(c(lib, .libPaths()))
if (!requireNamespace("glmmML", quietly = TRUE)) stop("glmmML is not installed in ", lib)
if (packageVersion("glmmML") != package_version(GLMMML_VERSION)) {
  stop("glmmML ", packageVersion("glmmML"), " found in ", lib, "; this script is pinned to ", GLMMML_VERSION)
}
suppressPackageStartupMessages(library(glmmML))

g17 <- function(x) sprintf("%.17g", unname(as.vector(x)))
r_version <- paste(R.version$major, R.version$minor, sep = ".")
today <- format(Sys.Date())
jackson_citation <- paste(
  "Jackson D, Law M, Stijnen T, Viechtbauer W, White IR (2018). A comparison of seven random-effects models for meta-analyses that estimate the",
  "summary odds ratio. Stat Med 37(7):1059-1085, Table 3, outcome 1 (pneumonia), columns A, B, C, D (data only)"
)

# Jackson et al. (2018), Table 3, outcome 1 (pneumonia): the same inline data as the published and derived cases.
jackson <- data.frame(
  ai = c(1, 0, 1, 13, 2, 4, 6), bi = c(155, 89, 43, 64, 157, 43, 76),
  ci = c(12, 3, 6, 27, 5, 6, 0), di = c(69, 40, 32, 53, 165, 43, 148)
)

toml_value <- function(v) {
  if (is.list(v)) {
    return(paste0("{ ", paste(sprintf("%s = %s", names(v), vapply(v, toml_value, "")), collapse = ", "), " }"))
  }
  if (is.character(v)) return(sprintf("\"%s\"", v))
  if (length(v) > 1L) return(sprintf("[%s]", paste(g17(v), collapse = ", ")))
  g17(v)
}

cases <- character(0)
add_case <- function(id, args, expect, tolerance, tool, locator, url, dataset = NULL, data = NULL) {
  source <- sprintf("%s, run on R %s%s", tool, r_version, if (is.null(data)) "" else paste0(". Data: ", jackson_citation))
  lines <- c("[[case]]", sprintf("id = \"%s\"", id), "kind = \"other\"", "function = \"rma_glmm\"")
  if (!is.null(dataset)) lines <- c(lines, sprintf("dataset = \"%s\"", dataset))
  if (!is.null(data)) lines <- c(lines, sprintf("data = %s", toml_value(as.list(data))))
  lines <- c(
    lines,
    sprintf("args = %s", toml_value(args)),
    sprintf("expect = { %s }", paste(sprintf("\"%s\" = %s", names(expect), vapply(expect, g17, "")), collapse = ", ")),
    sprintf("tolerance = %s", g17(tolerance)),
    "",
    "[case.source]",
    sprintf("citation = \"%s\"", source),
    sprintf("locator = \"%s\"", locator),
    sprintf("read = %s", today),
    sprintf("url = \"%s\"", url),
    ""
  )
  cases <<- c(cases, lines)
}

complete <- function(df, cols) df[stats::complete.cases(df[, cols]), , drop = FALSE]

### CM.EL with FE: the exact conditional estimate of a common odds ratio ---------------------------
exact_or <- function(ai, bi, ci, di) {
  tab <- array(rbind(ai, ci, bi, di), dim = c(2, 2, length(ai)))
  unname(log(stats::mantelhaen.test(tab, exact = TRUE)$estimate))
}
cm_el <- function(suffix, dataset, args, ai, bi, ci, di, data = NULL) {
  add_case(
    sprintf("other.glmm.cm.el.fe.%s", suffix), c(args, list(measure = "OR", model = "CM.EL", method = "FE")),
    list("beta[0]" = exact_or(ai, bi, ci, di)), 1e-5, "stats::mantelhaen.test(exact = TRUE), base R",
    "mantelhaen.test(array of 2 x 2 tables, exact = TRUE)$estimate, the conditional maximum likelihood estimate of the common odds ratio, on the log scale; the exact interval is not compared (metafopy gives a Wald interval)",
    "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mantelhaen.test.html", dataset = dataset, data = data
  )
}

nw <- metadat::dat.nielweise2007
cm_el("nielweise2007", "dat.nielweise2007", list(ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"), nw$ai, nw$n1i - nw$ai, nw$ci, nw$n2i - nw$ci)

col <- complete(metadat::dat.collins1985a, c("b.xti", "nti", "b.xci", "nci"))
cm_el("collins1985a", "dat.collins1985a", list(ai = "b.xti", n1i = "nti", ci = "b.xci", n2i = "nci"), col$b.xti, col$nti - col$b.xti, col$b.xci, col$nci - col$b.xci)

cm_el("jackson2018.pneumonia", NULL, list(ai = "ai", bi = "bi", ci = "ci", di = "di"), jackson$ai, jackson$bi, jackson$ci, jackson$di, data = jackson)

### glmmML: random-intercept binomial and Poisson models --------------------------------------------
glmm_ml <- function(formula, data, family) {
  data$cluster <- seq_len(nrow(data))
  # do.call puts the values in the call: glmmML evaluates its call in its caller's frame, where a local name `formula` is not found
  fit <- do.call(glmmML::glmmML, list(formula = formula, family = family, data = data, cluster = data$cluster, method = "ghq", n.points = N_POINTS))
  list(beta = unname(fit$coefficients[1]), se = unname(fit$coef.sd[1]), tau2 = unname(fit$sigma^2))
}
glmm_locator <- function(call) sprintf("%s: coefficients[1], coef.sd[1], sigma^2 (method = 'ghq', n.points = %d, a plain Gauss-Hermite rule)", call, N_POINTS)
glmm_url <- "https://cran.r-project.org/package=glmmML"
glmm_tool <- sprintf("glmmML %s", GLMMML_VERSION)
expect_fit <- function(f) list("beta[0]" = f$beta, "se[0]" = f$se, tau2 = f$tau2)

# One-group models
pr <- metadat::dat.pritz1997
f <- glmm_ml(cbind(xi, ni - xi) ~ 1, pr, binomial)
add_case(
  "other.glmm.plo.none.ml.pritz1997", list(xi = "xi", ni = "ni", measure = "PLO", method = "ML", nAGQ = AGQ), expect_fit(f), GLMM_TOLERANCE, glmm_tool,
  glmm_locator("glmmML(cbind(xi, ni - xi) ~ 1, family = binomial)"), glmm_url, dataset = "dat.pritz1997"
)

n8 <- metadat::dat.nielweise2008
# glmmML does not converge with the raw person-times as the offset (a log rate near -6 is far from its start). The offset log(t1i / 1000) is the same
# model with the intercept moved by log(1000), so the intercept is shifted back; the standard error and the variance are unchanged.
RATE_SCALE <- 1000
n8$off <- log(n8$t1i / RATE_SCALE)
f <- glmm_ml(x1i ~ 1 + offset(off), n8, poisson)
f$beta <- f$beta - log(RATE_SCALE)
add_case(
  "other.glmm.irln.none.ml.nielweise2008", list(xi = "x1i", ti = "t1i", measure = "IRLN", method = "ML", nAGQ = AGQ), expect_fit(f), GLMM_TOLERANCE, glmm_tool,
  glmm_locator("glmmML(x1i ~ 1 + offset(log(t1i / 1000)), family = poisson), intercept shifted back by log(1000)"), glmm_url, dataset = "dat.nielweise2008"
)

# CM.AL: ai is binomial with ai + ci trials and logit psi + log(n1 / n2), psi = beta + a random study effect
cm_al <- function(suffix, dataset, args, df, data = NULL) {
  f <- glmm_ml(cbind(a, c) ~ 1 + offset(log(n1 / n2)), df, binomial)
  add_case(
    sprintf("other.glmm.cm.al.ml.%s", suffix), c(args, list(measure = "OR", model = "CM.AL", method = "ML", nAGQ = AGQ)), expect_fit(f), GLMM_TOLERANCE, glmm_tool,
    glmm_locator("glmmML(cbind(ai, ci) ~ 1 + offset(log(n1i / n2i)), family = binomial)"), glmm_url, dataset = dataset, data = data
  )
}
cm_al("nielweise2007", "dat.nielweise2007", list(ai = "ai", n1i = "n1i", ci = "ci", n2i = "n2i"),
      data.frame(a = nw$ai, c = nw$ci, n1 = nw$n1i, n2 = nw$n2i))
cm_al("collins1985a", "dat.collins1985a", list(ai = "b.xti", n1i = "nti", ci = "b.xci", n2i = "nci"),
      data.frame(a = col$b.xti, c = col$b.xci, n1 = col$nti, n2 = col$nci))
cm_al("jackson2018.pneumonia", NULL, list(ai = "ai", bi = "bi", ci = "ci", di = "di"),
      data.frame(a = jackson$ai, c = jackson$ci, n1 = jackson$ai + jackson$bi, n2 = jackson$ci + jackson$di), data = jackson)

header <- c(
  "# Other-software cases for rma_glmm from base R's mantelhaen.test and glmmML, written by src/tests/correctness/r/other_software_glmm.R; do not",
  "# edit by hand.",
  "#",
  "# mantelhaen.test(exact = TRUE) and glmmML do not depend on metafor or lme4, so these values are independent of them. glmmML integrates by plain",
  "# Gauss-Hermite quadrature; see the script for the conventions it pins.",
  ""
)
out <- file.path("src", "tests", "correctness", "cases", "glmm_other.toml")
writeLines(c(header, cases), out)
cat("wrote", out, "\n")
