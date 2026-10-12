# Reference values for rma.uni features not covered by the per-dataset grid:
# user weights (incl. GENQ), fixed tau^2, unweighted fits, t/adhoc tests,
# btt, iterated estimators, no-intercept models, anova(), predict(newmods)
# and confint(fixed=TRUE).
#
# Usage (from the repository root):  Rscript tests/r/generate_extras.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
core <- function(res) list(
  beta = num(res$beta), se = num(res$se), pval = num(res$pval), ci_lb = num(res$ci.lb),
  ci_ub = num(res$ci.ub), tau2 = res$tau2, se_tau2 = num(res$se.tau2), QE = res$QE,
  QM = res$QM, QMp = res$QMp, I2 = num(res$I2), H2 = num(res$H2), R2 = num(res$R2),
  ll_ml = num(fitstats(res, REML = FALSE)[1]), ll_reml = num(fitstats(res, REML = TRUE)[1]),
  weights = num(weights(res)), hat = num(hatvalues(res))
)

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$n <- with(dat.bcg, tpos + tneg + cpos + cneg)
out <- list()

out$weights_REML  <- core(rma(yi, vi, weights = n, data = dat))
out$weights_knha  <- core(rma(yi, vi, weights = n, data = dat, test = "knha"))
out$genq          <- core(rma(yi, vi, weights = n, data = dat, method = "GENQ"))
out$genq_mods     <- core(rma(yi, vi, weights = n, mods = ~ ablat, data = dat, method = "GENQ"))
out$tau2_fixed    <- core(rma(yi, vi, tau2 = 0.2, data = dat))
out$tau2_fixed_mods <- core(rma(yi, vi, tau2 = 0.2, mods = ~ ablat, data = dat))
out$unweighted    <- core(rma(yi, vi, weighted = FALSE, data = dat))
out$unweighted_knha <- core(rma(yi, vi, weighted = FALSE, mods = ~ ablat, data = dat, test = "knha"))
out$test_t        <- core(rma(yi, vi, mods = ~ ablat, data = dat, test = "t"))
out$test_adhoc    <- core(rma(yi, vi, mods = ~ ablat, data = dat, test = "adhoc"))
out$btt           <- core(rma(yi, vi, mods = ~ ablat + alloc, data = dat, btt = 3:4))
out$dlit          <- core(rma(yi, vi, data = dat, method = "DLIT"))
out$sjit          <- core(rma(yi, vi, data = dat, method = "SJIT"))
out$no_intercept  <- core(rma(yi, vi, mods = ~ 0 + alloc, data = dat))
out$level90       <- core(rma(yi, vi, data = dat, level = 90))
out$subset        <- core(rma(yi, vi, data = dat, subset = alloc == "random"))

full <- rma(yi, vi, mods = ~ ablat + alloc, data = dat, method = "ML")
red  <- rma(yi, vi, mods = ~ ablat, data = dat, method = "ML")
a <- anova(full, red)
out$anova_lrt <- list(LRT = a$LRT, pval = a$pval, parms_f = a$parms.f, parms_r = a$parms.r, R2 = a$R2)
w <- anova(full, X = rbind(c(0, 1, 0, 0), c(0, 0, 1, -1)))
out$anova_X <- list(Xb = num(w$Xb), se = num(w$se), zval = num(w$zval), pval = num(w$pval), QM = w$QM, QMp = w$QMp)
wb <- anova(rma(yi, vi, mods = ~ ablat + alloc, data = dat, test = "knha"), btt = 3:4)
out$anova_btt_knha <- list(QM = wb$QM, QMp = wb$QMp)

mreg <- rma(yi, vi, mods = ~ ablat, data = dat)
pr <- predict(mreg, newmods = c(10, 30, 50))
out$predict_newmods <- list(pred = num(pr$pred), se = num(pr$se), ci_lb = num(pr$ci.lb), ci_ub = num(pr$ci.ub), pi_lb = num(pr$pi.lb), pi_ub = num(pr$pi.ub))
pf <- predict(mreg)
out$predict_fitted <- list(pred = num(pf$pred), pi_lb = num(pf$pi.lb))
ci <- confint(rma(yi, vi, mods = ~ ablat, data = dat, test = "knha"), fixed = TRUE, random = FALSE)
out$confint_fixed_knha <- list(estimate = num(ci$fixed[, "estimate"]), ci_lb = num(ci$fixed[, "ci.lb"]), ci_ub = num(ci$fixed[, "ci.ub"]))

out$meta <- list(metafor = as.character(packageVersion("metafor")), R = R.version.string)
write_json(out, "tests/reference/extras/rma_uni.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote tests/reference/extras/rma_uni.json\n")
