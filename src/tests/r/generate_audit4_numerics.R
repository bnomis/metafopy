# Reference values for the numerical-agreement items of the fourth audit: rma.mv() with struct = "GEN" on konstantopoulos2011 (where nlminb stops on a
# flat ridge), permutest() of a location-scale fit whose permuted scale fits run to the boundary, and the scale standard errors of an identity-link
# location-scale fit at two Richardson depths of numDeriv's Hessian.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_audit4_numerics.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
out <- list()

### rma.mv(struct = "GEN"): R's default run stops at logLik -31.47369 (tau2 = 0.2056, 1.3e-8, rho = -0.8045) on a ridge that continues to rho = -1;
### started near the ridge's end R reaches -30.98466 (tau2 = 3801.59, 9.6e-4)
kon <- dat.konstantopoulos2011
gen_default <- rma.mv(yi, vi, mods = ~year, random = ~year | district, struct = "GEN", data = kon)
gen_hot <- rma.mv(yi, vi, mods = ~year, random = ~year | district, struct = "GEN", data = kon, control = list(tau2.init = c(3801.59, 9.63e-4), rho.init = -0.9999))
gen_nomods <- rma.mv(yi, vi, random = ~year | district, struct = "GEN", data = kon)
gen_nomods_hot <- rma.mv(yi, vi, random = ~year | district, struct = "GEN", data = kon, control = list(tau2.init = c(3131.85, 7.94e-4), rho.init = -0.9999))
out$gen <- list(
  kon = lapply(kon[c("district", "school", "study", "yi", "vi", "year")], num),
  default_ll = as.numeric(logLik(gen_default)), default_tau2 = num(gen_default$tau2), default_rho = num(gen_default$rho),
  hot_ll = as.numeric(logLik(gen_hot)), hot_tau2 = num(gen_hot$tau2), hot_rho = num(gen_hot$rho), hot_beta = num(gen_hot$beta),
  nomods_default_ll = as.numeric(logLik(gen_nomods)), nomods_hot_ll = as.numeric(logLik(gen_nomods_hot))
)

### permutest() of a location-scale fit
bcg <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, mods = ~ablat, scale = ~alloc, data = bcg)
pt <- suppressWarnings(permutest(res, iter = 50, progbar = FALSE, seed = 1))
out$permutest <- list(
  pval = num(pt$pval), QMp = pt$QMp, pval_alpha = num(pt$pval.alpha), QSp = pt$QSp,
  alpha_perm = lapply(pt$alpha.perm, num), zval_alpha_perm = lapply(pt$zval.alpha.perm, num), QS_perm = num(pt$QS.perm)
)

### identity link: the scale coefficient is -3.8e-5, so numDeriv's steps (0.1 * |alpha|, halved) are of rounding-error size at depth r = 8
hes <- function(r) {
  f <- suppressWarnings(rma(yi, vi, scale = ~ablat, link = "identity", data = bcg, control = list(hessianCtrl = list(r = r))))
  list(alpha = num(f$alpha), se_alpha = num(f$se.alpha), ll = as.numeric(logLik(f)))
}
out$identity <- list(r4 = hes(4), r8 = hes(8))

write_json(out, "src/tests/reference/phase12/audit4_numerics.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null")
