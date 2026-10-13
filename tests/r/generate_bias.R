# Reference values for regtest(), ranktest(), trimfill(), fsn(), hc() and tes(), plus their printed output.
#
# Usage (from the repository root):  Rscript tests/r/generate_bias.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
g17 <- function(x) if (is.null(x)) NULL else lapply(num(x), function(v) if (is.na(v)) NULL else sprintf("%.17g", v))

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$n <- with(dat.bcg, tpos + tneg + cpos + cneg)
res <- rma(yi, vi, data = dat)
res_mods <- rma(yi, vi, mods = ~ ablat, data = dat)
hack <- dat.hackshaw1998
res_h <- rma(yi, vi, data = hack)
set.seed(1)
sim <- data.frame(vi = runif(60, 0.01, 0.2))
sim$yi <- rnorm(60, 0.3, sqrt(sim$vi + 0.05)) + 0.5 * sqrt(sim$vi)
res_sim <- rma(yi, vi, data = sim)
out <- list(sim = list(yi = num(sim$yi), vi = num(sim$vi)))

### regtest ----------------------------------------------------------------------
rt <- function(x, ...) {
  r <- do.call(regtest, list(x, ...))
  list(zval = num(r$zval), pval = num(r$pval), ddf = num(r$ddf), est = num(r$est), ci.lb = num(r$ci.lb), ci.ub = num(r$ci.ub), method = r$method, print = printed(r))
}
out$reg_sei <- rt(res)
out$reg_sei_fit <- rt(res, ret.fit = TRUE)
out$reg_vi <- rt(res, predictor = "vi")
out$reg_lm <- rt(res, model = "lm")
out$reg_lm_vi <- rt(res, model = "lm", predictor = "vi")
out$reg_knha <- rt(rma(yi, vi, data = dat, test = "knha"))
out$reg_ninv <- rt(res, predictor = "ninv", ni = dat$n)
out$reg_ni <- rt(res, predictor = "ni", ni = dat$n)
out$reg_sqrtni <- rt(res, predictor = "sqrtni", ni = dat$n)
out$reg_sqrtninv <- rt(res, predictor = "sqrtninv", ni = dat$n)
out$reg_mods <- rt(res_mods)
out$reg_ee <- rt(rma(yi, vi, data = dat, method = "EE"))
out$reg_ml <- rt(rma(yi, vi, data = dat, method = "ML"))
out$reg_level <- rt(res, level = 90)
out$reg_hack <- rt(res_h)
out$reg_hack_lm <- rt(res_h, model = "lm")
out$reg_vec <- rt(dat$yi, vi = dat$vi)
out$reg_vec_sei <- rt(dat$yi, sei = sqrt(dat$vi), model = "lm")
out$reg_vec_ni <- rt(dat$yi, vi = dat$vi, ni = dat$n, predictor = "ninv", method = "DL")
yna <- dat$yi; yna[4] <- NA
out$reg_vec_na <- suppressWarnings(rt(yna, vi = dat$vi))
dna <- dat; dna$yi[3] <- NA
out$reg_na_model <- rt(suppressWarnings(rma(yi, vi, data = dna)))

### ranktest ---------------------------------------------------------------------
rk <- function(x, ...) {
  r <- do.call(ranktest, list(x, ...))
  list(tau = num(r$tau), pval = num(r$pval), print = printed(r))
}
out$rank_re <- rk(res)
out$rank_hack <- rk(res_h)
out$rank_sim <- rk(res_sim)
out$rank_vec <- rk(dat$yi, vi = dat$vi)
out$rank_noexact <- rk(res, exact = FALSE)
ties <- data.frame(yi = c(0.1, 0.4, 0.4, 0.8, 0.2, 0.9, 0.5, 0.3), vi = c(0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.5))
out$rank_ties <- rk(ties$yi, vi = ties$vi)

### trimfill ---------------------------------------------------------------------
tf <- function(x, ...) {
  r <- suppressWarnings(do.call(trimfill, list(x, ...)))
  list(k0 = r$k0, se.k0 = num(r$se.k0), side = r$side, p.k0 = num(r$p.k0), fill = as.integer(r$fill), beta = num(r$beta), se = num(r$se), tau2 = num(r$tau2), k = r$k,
       yi = num(r$yi), vi = num(r$vi), slab = as.character(r$slab), QE = num(r$QE), print = printed(r))
}
for (est in c("L0", "R0", "Q0")) {
  out[[paste0("tf_re_", est)]] <- tf(res, estimator = est)
  out[[paste0("tf_hack_", est)]] <- tf(res_h, estimator = est)
  out[[paste0("tf_sim_", est)]] <- tf(res_sim, estimator = est)
}
out$tf_left <- tf(res_h, side = "left")
out$tf_right <- tf(res_h, side = "right")
out$tf_right_sim <- tf(res_sim, side = "right", estimator = "R0")
out$tf_ee <- tf(rma(yi, vi, data = hack, method = "EE"))
out$tf_dl <- tf(rma(yi, vi, data = hack, method = "DL"))
out$tf_knha <- tf(rma(yi, vi, data = hack, test = "knha"))
out$tf_ilim <- tf(res_h, ilim = c(-1, 3))
out$tf_slab <- tf(rma(yi, vi, data = dat, slab = paste(author, year)), side = "right")
out$tf_weights <- tf(rma(yi, vi, data = hack, weights = 1 / sqrt(vi)))
out$tf_level <- tf(res_h, level = 90)

### fsn --------------------------------------------------------------------------
fs <- function(...) {
  r <- tryCatch(do.call(fsn, list(...)), error = function(e) e)
  if (inherits(r, "error")) return(list(error = conditionMessage(r)))
  list(fsnum = num(r$fsnum), est = num(r$est), tau2 = num(r$tau2), pval = num(r$pval), target = num(r$target), est.fsn = num(r$est.fsn), tau2.fsn = num(r$tau2.fsn), pval.fsn = num(r$pval.fsn),
       ub.sign = r$ub.sign, type = r$type, print = printed(r))
}
out$fsn_rosenthal <- fs(dat$yi, vi = dat$vi)
out$fsn_rosenthal_hack <- fs(hack$yi, vi = hack$vi)
out$fsn_fisher <- fs(hack$yi, vi = hack$vi, pool = "fisher")
out$fsn_rosenthal_alpha <- fs(dat$yi, vi = dat$vi, alpha = 0.01)
out$fsn_binomial <- fs(hack$yi, vi = hack$vi, type = "Binomial")
out$fsn_orwin <- fs(dat$yi, vi = dat$vi, type = "Orwin")
out$fsn_orwin_target <- fs(dat$yi, vi = dat$vi, type = "Orwin", target = -0.1)
out$fsn_orwin_unw <- fs(dat$yi, type = "Orwin", weighted = FALSE)
out$fsn_orwin_unw_target <- fs(dat$yi, type = "Orwin", weighted = FALSE, target = -0.2)
out$fsn_rosenberg <- fs(dat$yi, vi = dat$vi, type = "Rosenberg")
out$fsn_general <- fs(res)
out$fsn_general_hack <- fs(res_h)
out$fsn_general_ee <- fs(rma(yi, vi, data = dat, method = "EE"))
out$fsn_general_dl <- fs(rma(yi, vi, data = hack, method = "DL"))
out$fsn_general_target <- fs(res, target = -0.3)
out$fsn_general_target_pos <- fs(rma(yi, vi, data = sim), target = 0.1)
out$fsn_general_vec <- fs(hack$yi, vi = hack$vi, type = "General")
out$fsn_general_vec_method <- fs(hack$yi, vi = hack$vi, type = "General", method = "DL")
out$fsn_general_knha <- fs(rma(yi, vi, data = hack, test = "knha"))
out$fsn_general_weighted <- fs(hack$yi, vi = hack$vi, type = "General", weighted = FALSE)
out$fsn_general_tau2 <- fs(hack$yi, vi = hack$vi, type = "General", tau2 = 0.05)
out$fsn_general_mumiss <- fs(res_h, mumiss = 0.1)
out$fsn_general_interval <- fs(res_h, interval = c(0, 2000))
set.seed(42)
out$fsn_general_exact <- fs(res, exact = TRUE)
set.seed(42)
out$fsn_general_exact_target <- fs(res, exact = TRUE, target = -0.3)
set.seed(7)
out$fsn_general_exact_hack <- fs(res_h, exact = TRUE)
out$fsn_general_maxint <- fs(res_h, target = 0.001, maxint = 1000)

### hc ---------------------------------------------------------------------------
hcf <- function(x, ...) {
  r <- do.call(hc, list(x, ...))
  list(beta = num(r$beta), se = num(r$se), ci.lb = num(r$ci.lb), ci.ub = num(r$ci.ub), beta.rma = num(r$beta.rma), se.rma = num(r$se.rma), ci.lb.rma = num(r$ci.lb.rma), ci.ub.rma = num(r$ci.ub.rma),
       tau2 = num(r$tau2), tau2.rma = num(r$tau2.rma), print = printed(r))
}
out$hc_re <- hcf(res)
out$hc_dl <- hcf(rma(yi, vi, data = dat, method = "DL"))
out$hc_hack <- hcf(res_h)
out$hc_ml <- hcf(rma(yi, vi, data = dat, method = "ML"))
out$hc_level <- hcf(rma(yi, vi, data = dat, level = 90))
out$hc_exp <- hcf(res, transf = exp)
out$hc_ee <- hcf(rma(yi, vi, data = dat, method = "EE"))
out$hc_sim <- hcf(res_sim)
out$hc_intexp <- hcf(res, transf = transf.exp.int)

### tes --------------------------------------------------------------------------
te <- function(x, ...) {
  r <- suppressWarnings(do.call(tes, list(x, ...)))
  list(k = r$k, O = r$O, E = num(r$E), OEratio = num(r$OEratio), test = r$test, X2 = num(r$X2), pval = num(r$pval), power = num(r$power), sig = as.integer(r$sig), theta = num(r$theta),
       theta.lim = if (is.null(r$theta.lim)) "none" else num(r$theta.lim), print = printed(r))
}
out$tes_re <- te(res)
out$tes_re_chi2 <- te(res, test = "chi2")
out$tes_re_binom <- te(res, test = "binom")
out$tes_ee <- te(rma(yi, vi, data = dat, method = "EE"))
out$tes_hack <- te(res_h)
out$tes_sim <- te(res_sim)
out$tes_mods <- te(res_mods)
out$tes_vec <- te(dat$yi, dat$vi)
out$tes_vec_chi2 <- te(dat$yi, dat$vi, test = "chi2")
out$tes_vec_binom <- te(dat$yi, dat$vi, test = "binom")
out$tes_vec_correct <- te(dat$yi, dat$vi, test = "chi2", correct = TRUE)
out$tes_theta <- te(dat$yi, dat$vi, theta = -0.5)
out$tes_theta_vec <- te(dat$yi, dat$vi, theta = seq(-1, 0.1, length.out = 13))
out$tes_tau2 <- te(dat$yi, dat$vi, theta = -0.5, tau2 = 0.3)
out$tes_tau2_chi2 <- te(dat$yi, dat$vi, theta = -0.5, tau2 = 0.3, test = "chi2")
out$tes_greater <- te(dat$yi, dat$vi, alternative = "greater", theta = 0.2)
out$tes_less <- te(dat$yi, dat$vi, alternative = "less")
out$tes_less_tes <- te(dat$yi, dat$vi, tes.alternative = "less")
out$tes_two <- te(dat$yi, dat$vi, tes.alternative = "two.sided")
out$tes_two_chi2 <- te(dat$yi, dat$vi, tes.alternative = "two.sided", test = "chi2")
out$tes_alpha <- te(dat$yi, dat$vi, alpha = 0.1, tes.alpha = 0.05)
out$tes_h0 <- te(dat$yi, dat$vi, H0 = -0.3)
out$tes_small <- te(dat$yi[1:5], dat$vi[1:5])
out$tes_nolim <- te(dat$yi, dat$vi, find.lim = FALSE)
out$tes_sei <- te(dat$yi, sei = sqrt(dat$vi))
out$tes_big_chi2 <- te(hack$yi, hack$vi)
out$tes_two_lim <- te(dat$yi, dat$vi, tes.alternative = "two.sided", theta = -0.7)

out$meta <- list(metafor = as.character(packageVersion("metafor")), R = R.version.string)
write_json(out, "tests/reference/phase2/bias.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote tests/reference/phase2/bias.json\n")
