# Reference values for the examples on metafor's rma.uni help page (all but the
# location-scale ones, which belong to a later phase) and for the Phase 1
# additions: measure= passthrough, formulas in yi, CO/VC/GENQM estimators,
# confint(type = "GENQ" / "PL" / "HT"), and anova() with pairmat().
#
# Usage (from the repository root):  Rscript tests/r/generate_rma_uni_help.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
core <- function(res) list(
  beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval),
  ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub), tau2 = res$tau2, se_tau2 = num(res$se.tau2),
  QE = res$QE, QEp = res$QEp, QM = res$QM, QMp = res$QMp, I2 = num(res$I2), H2 = num(res$H2),
  R2 = num(res$R2), k = res$k, coef_names = rownames(res$beta),
  ll = num(logLik(res))
)
ci_random <- function(ci) list(
  estimate = num(ci$random[, "estimate"]), ci_lb = num(ci$random[, "ci.lb"]), ci_ub = num(ci$random[, "ci.ub"])
)

out <- list()

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
out$re_reml <- core(rma(yi, vi, data = dat, method = "REML"))
dat$sei <- sqrt(dat$vi)
out$re_sei <- core(rma(yi, sei = sei, data = dat))
out$measure_rr <- core(rma(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat))
out$mods_cbind <- core(rma(yi, vi, mods = cbind(ablat, year), data = dat))
out$mods_formula <- core(rma(yi, vi, mods = ~ ablat + year, data = dat))
out$yi_formula <- core(rma(yi ~ ablat + year, vi, data = dat))
alloc.random <- ifelse(dat$alloc == "random", 1, 0)
alloc.systematic <- ifelse(dat$alloc == "systematic", 1, 0)
out$alloc_dummies <- core(rma(yi, vi, mods = ~ alloc.random + alloc.systematic + year + ablat, data = dat, btt = 2:3))
out$alloc_factor <- core(rma(yi, vi, mods = ~ factor(alloc) + year + ablat, data = dat, btt = 2:3))
res <- rma(yi, vi, mods = ~ alloc + year + ablat, data = dat, btt = 2:3)
out$alloc_char <- core(res)
pm <- pairmat(res, btt = "alloc")
a <- anova(res, X = pm)
out$anova_pairmat <- list(X = unname(pm), rownames = rownames(pm), Xb = num(a$Xb), se = num(a$se), zval = num(a$zval), pval = num(a$pval), QM = a$QM, QMp = a$QMp)
pm2 <- pairmat(res, btt = 2:3, btt2 = 2:3)
out$pairmat_btt2 <- list(X = unname(pm2), rownames = rownames(pm2))
out$sub_alternate <- core(rma(yi, vi, data = dat, subset = (alloc == "alternate")))
out$sub_random <- core(rma(yi, vi, data = dat, subset = (alloc == "random")))
out$sub_systematic <- core(rma(yi, vi, data = dat, subset = (alloc == "systematic")))
out$alloc_noint <- core(rma(yi, vi, mods = ~ 0 + factor(alloc), data = dat))
out$fe_total <- core(rma(yi, vi, data = dat, method = "FE"))
out$fe_mods <- core(rma(yi, vi, mods = ~ ablat + year, data = dat, method = "FE"))
out$alloc_factor_only <- core(rma(yi, vi, mods = ~ factor(alloc), data = dat))
dat2 <- dat.bangertdrowns2004
out$bangert_re <- core(rma(yi, vi, data = dat2))

# Phase 1 additions
dat$n <- with(dat.bcg, tpos + tneg + cpos + cneg)
out$co <- core(rma(yi, vi, data = dat, method = "CO"))
out$vc_mods <- core(rma(yi, vi, mods = ~ ablat, data = dat, method = "VC"))
genqm <- rma(yi, vi, weights = n, data = dat, method = "GENQM")
out$genqm <- core(genqm)
out$genqm_mods <- core(rma(yi, vi, weights = n, mods = ~ ablat, data = dat, method = "GENQM"))
out$genqm_ci <- ci_random(confint(genqm))
genq <- rma(yi, vi, weights = n, data = dat, method = "GENQ")
out$genq_ci <- ci_random(confint(genq))
re <- rma(yi, vi, data = dat)
out$pl_ci <- ci_random(confint(re, type = "PL"))
out$ht_ci <- ci_random(confint(re, type = "HT"))
out$qp_ci <- ci_random(confint(re))
ml <- rma(yi, vi, mods = ~ ablat, data = dat, method = "ML")
out$pl_ci_ml <- ci_random(confint(ml, type = "PL"))
ci <- confint(re, fixed = TRUE, random = FALSE, transf = exp)
out$confint_transf <- list(estimate = num(ci$fixed[, "estimate"]), ci_lb = num(ci$fixed[, "ci.lb"]), ci_ub = num(ci$fixed[, "ci.ub"]))
pr <- predict(re, transf = transf.exp.int)
out$predict_exp_int <- list(pred = num(pr$pred), ci_lb = num(pr$ci.lb), ci_ub = num(pr$ci.ub), pi_lb = num(pr$pi.lb), pi_ub = num(pr$pi.ub))
ci <- confint(re, fixed = TRUE, random = FALSE, transf = transf.ilogit.int)
out$confint_ilogit_int <- list(estimate = num(ci$fixed[, "estimate"]), ci_lb = num(ci$fixed[, "ci.lb"]), ci_ub = num(ci$fixed[, "ci.ub"]))
out$measure_smd <- core(rma(measure = "SMD", m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i, data = dat.normand1999))
out$measure_or_subset <- core(rma(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, mods = ~ ablat, data = dat.bcg, subset = alloc != "random"))
out$measure_pr_add <- core(rma(measure = "PLO", xi = ci, ni = n1i, data = dat.nielweise2007, add = 0.25, to = "all"))
out$test_hksj <- core(rma(yi, vi, data = dat, test = "hksj"))

out$meta <- list(metafor = as.character(packageVersion("metafor")), R = R.version.string)
dir.create("tests/reference/extras", showWarnings = FALSE, recursive = TRUE)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE), "tests/reference/extras/rma_uni_help.json")
