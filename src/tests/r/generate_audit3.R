# Reference values for the gaps closed after the third audit: sample sizes picked up from escalc() data frames, hatvalues(type = "matrix"),
# summary() (the fit-statistics table), fitstats()/AIC()/BIC() of several models, predict() of cluster-robust (clubSandwich) fits, the functions
# robust() fits refuse, simulate() of robust fits, update() of trimfill results, gosh() with a seed (R's dbinom decides the subset sizes) and the
# standard errors of a location-scale model.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_audit3.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(clubSandwich)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
mat <- function(x) {
  x <- as.matrix(x)
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}
quiet <- function(expr) suppressWarnings(suppressMessages(expr))

out <- list()
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
hack <- dat.hackshaw1998
kon <- dat.konstantopoulos2011

### ni from escalc() -------------------------------------------------------------------------------------------------------------------------
res_ni <- rma(yi, vi, data = dat)
res_ni_sub <- rma(yi, vi, data = dat, subset = ablat > 20)
res_ni_mods <- rma(yi, vi, mods = ~ablat, data = dat)
pdf(NULL)
funnel_xy <- function(...) {
  sav <- funnel(...)
  list(x = num(sav$x), y = num(sav$y))
}
out$ni <- list(
  ni = num(res_ni$ni), ni_sub = num(res_ni_sub$ni), ni_mods = num(res_ni_mods$ni),
  regtest = lapply(c("ni", "sqrtni", "ninv", "sqrtninv"), function(p) {
    r <- regtest(res_ni, predictor = p)
    list(predictor = p, zval = r$zval, pval = r$pval, dfs = num(r$dfs), fit_beta = num(r$fit$beta), print = printed(r))
  }),
  funnel = lapply(c("ni", "sqrtni", "ninv", "sqrtninv", "lni"), function(y) c(list(yaxis = y), funnel_xy(res_ni, yaxis = y))),
  selmodel = lapply(c("sqrtni", "ninv", "sqrtninv"), function(p) {
    r <- selmodel(res_ni, type = "halfnorm", prec = p)
    list(prec = p, beta = num(r$beta), tau2 = r$tau2, delta = num(r$delta), ll = num(r$fit.stats["ll", "ML"]), print = printed(r))
  })
)
dev.off()

### hatvalues(type = "matrix") ---------------------------------------------------------------------------------------------------------------
res_hat <- rma(yi, vi, mods = ~ablat, data = dat)
res_hat_ee <- rma(yi, vi, data = dat, method = "EE")
res_hat_w <- rma(yi, vi, mods = ~ablat, data = dat, weights = 1 / sqrt(vi))
res_hat_ls <- rma(yi, vi, scale = ~ablat, data = dat)
res_hat_tf <- trimfill(rma(yi, vi, data = dat))
res_hat_rb <- robust(res_hat, cluster = trial)
out$hat <- list(
  uni_mods = mat(hatvalues(res_hat, type = "matrix")), uni_mods_diag = num(hatvalues(res_hat)),
  uni_ee = mat(hatvalues(res_hat_ee, type = "matrix")), uni_w = mat(hatvalues(res_hat_w, type = "matrix")),
  ls = mat(hatvalues(res_hat_ls, type = "matrix")), ls_diag = num(hatvalues(res_hat_ls)),
  trimfill = mat(hatvalues(res_hat_tf, type = "matrix")), robust = mat(hatvalues(res_hat_rb, type = "matrix")),
  cooks_robust = num(cooks.distance(res_hat_rb)), cooks_uni = num(cooks.distance(res_hat))
)

### summary() --------------------------------------------------------------------------------------------------------------------------------
sel <- selmodel(rma(yi, vi, data = dat), type = "halfnorm")
out$summary <- list(
  uni_re = printed(summary(rma(yi, vi, data = dat))), uni_mods = printed(summary(rma(yi, vi, mods = ~ablat, data = dat))),
  uni_ml = printed(summary(rma(yi, vi, data = dat, method = "ML"))), uni_ee = printed(summary(rma(yi, vi, data = dat, method = "EE"))),
  uni_knha = printed(summary(rma(yi, vi, data = dat, test = "knha"))), uni_tau2 = printed(summary(rma(yi, vi, data = dat, tau2 = 0.1))),
  selmodel = printed(summary(sel)),
  mh = printed(summary(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg))),
  peto = printed(summary(rma.peto(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg))),
  mv = printed(summary(rma.mv(yi, vi, random = ~1 | trial, data = dat))),
  mv_mods = printed(summary(rma.mv(yi, vi, mods = ~ablat, random = ~1 | trial, data = dat))),
  ls = printed(summary(res_hat_ls)),
  trimfill = printed(summary(res_hat_tf)),
  print_trimfill = printed(res_hat_tf)
)

### fitstats(), AIC(), BIC() of several models ----------------------------------------------------------------------------------------------
r1 <- rma(yi, vi, data = dat)
r2 <- rma(yi, vi, mods = ~ablat, data = dat)
r3 <- rma(yi, vi, mods = ~ablat + year, data = dat)
l1 <- rma(yi, vi, scale = ~1, data = dat)
l2 <- rma(yi, vi, scale = ~ablat, data = dat)
m1 <- rma.mv(yi, vi, random = ~1 | trial, data = dat)
m2 <- rma.mv(yi, vi, mods = ~ablat, random = ~1 | trial, data = dat)
s1 <- selmodel(r1, type = "halfnorm")
s2 <- selmodel(r1, type = "negexp")
frame <- function(x) list(rownames = rownames(x), names = names(x), columns = lapply(x, num))
out$fitstats <- list(
  uni = frame(fitstats(r1, r2, r3)), uni_ml = frame(fitstats(r1, r2, r3, REML = FALSE)), uni_single = num(fitstats(r2)),
  uni_ml_first = frame(fitstats(rma(yi, vi, data = dat, method = "ML"), rma(yi, vi, mods = ~ablat, data = dat, method = "ML"))),
  ls = frame(fitstats(l1, l2)), mv = frame(fitstats(m1, m2)), sel = frame(fitstats(s1, s2)),
  aic_uni = frame(AIC(r1, r2, r3)), aic_uni_correct = frame(AIC(r1, r2, r3, correct = TRUE)), aic_single_correct = AIC(r2, correct = TRUE),
  aic_single_k = AIC(r2, k = 3), aic_ls = frame(AIC(l1, l2)), aic_mv = frame(AIC(m1, m2, correct = TRUE)), aic_sel = frame(AIC(s1, s2)),
  bic_uni = frame(BIC(r1, r2, r3)), bic_ls = frame(BIC(l1, l2)), bic_mv = frame(BIC(m1, m2)), bic_sel = frame(BIC(s1, s2)),
  warning = tryCatch(withCallingHandlers(AIC(r1, rma(yi, vi, data = dat, subset = ablat > 20)), warning = function(w) { conditionMessage(w) }),
                     warning = function(w) conditionMessage(w))
)

### predict() of cluster-robust (clubSandwich) fits ------------------------------------------------------------------------------------------
uni1 <- rma(yi, vi, mods = ~I(year - 1990), data = kon)
mv1 <- rma.mv(yi, vi, mods = ~I(year - 1990), random = ~1 | district / school, data = kon)
uni0 <- rma(yi, vi, data = kon)
pred_fields <- function(p) list(pred = num(p$pred), se = num(p$se), ci_lb = num(p$ci.lb), ci_ub = num(p$ci.ub), pi_lb = num(p$pi.lb), pi_ub = num(p$pi.ub),
                                pi_ddf = num(p$pi.ddf), names = names(p))
out$predict_cs <- list()
for (ct in c("Satterthwaite", "z", "naive-t", "naive-tp")) {
  ru <- robust(uni1, cluster = district, clubSandwich = TRUE, coef_test = ct)
  rm_ <- robust(mv1, cluster = district, clubSandwich = TRUE, coef_test = ct)
  out$predict_cs[[paste0("uni_", ct)]] <- c(
    pred_fields(predict(ru, newmods = c(-5, 0, 5))), list(fit = pred_fields(predict(ru)), level90 = pred_fields(predict(ru, newmods = c(-5, 0, 5), level = 90)),
    adjust = pred_fields(predict(ru, newmods = c(-5, 0, 5), adjust = TRUE)), transf = pred_fields(predict(ru, newmods = c(-5, 0, 5), transf = exp)),
    vcov = mat(predict(ru, newmods = c(-5, 0, 5), vcov = TRUE)$vcov)))
  out$predict_cs[[paste0("mv_", ct)]] <- c(pred_fields(predict(rm_, newmods = c(-5, 0, 5))), list(fit = pred_fields(predict(rm_))))
}
out$predict_cs$uni_int <- pred_fields(predict(robust(uni0, cluster = district, clubSandwich = TRUE)))
out$predict_cs$uni_cr1 <- pred_fields(predict(robust(uni1, cluster = district, clubSandwich = TRUE, vcov = "CR1"), newmods = c(-5, 0, 5)))
out$predict_cs$saddlepoint <- tryCatch({ predict(robust(uni1, cluster = district, clubSandwich = TRUE, coef_test = "saddlepoint"), newmods = 0); "no error" },
                                       error = function(e) conditionMessage(e))
rcs <- robust(uni1, cluster = district, clubSandwich = TRUE)
rcs_int <- robust(uni0, cluster = district, clubSandwich = TRUE)
pdf(NULL)
forest(rcs)
regplot(rcs, mod = 2)
forest(rcs_int)
addpoly(rcs_int)
dev.off()

### functions that refuse robust() fits -----------------------------------------------------------------------------------------------------
rb_u <- robust(uni1, cluster = district)
rc_u <- robust(uni1, cluster = district, clubSandwich = TRUE)
rb_u0 <- robust(uni0, cluster = district)
rb_m <- robust(mv1, cluster = district)
refuse <- list(
  rstandard = function(x) rstandard(x), rstudent = function(x) rstudent(x), dfbetas = function(x) dfbetas(x), influence = function(x) influence(x),
  leave1out = function(x) leave1out(x), cumul = function(x) cumul(x), permutest = function(x) permutest(x, iter = 20),
  funnel = function(x) { pdf(NULL); on.exit(dev.off()); funnel(x) }, qqnorm = function(x) { pdf(NULL); on.exit(dev.off()); qqnorm(x) },
  plot_influence = function(x) { pdf(NULL); on.exit(dev.off()); plot(influence(x)) },
  plot_permutest = function(x) { pdf(NULL); on.exit(dev.off()); plot(permutest(x, iter = 20)) },
  cooks = function(x) cooks.distance(x), hatvalues = function(x) hatvalues(x), simulate = function(x) simulate(x, nsim = 1, seed = 1)
)
out$refuse <- lapply(list(rb_u = rb_u, rc_u = rc_u, rb_u0 = rb_u0, rb_m = rb_m), function(x) {
  lapply(refuse, function(f) tryCatch({ quiet(f(x)); "ok" }, error = function(e) conditionMessage(e)))
})
out$simulate_robust <- list(
  uni = mat(simulate(rb_u, nsim = 3, seed = 42)), uni_cs = mat(simulate(rc_u, nsim = 2, seed = 7)), mv = mat(simulate(rb_m, nsim = 2, seed = 5)),
  uni_withvb = mat(simulate(rb_u, nsim = 2, seed = 9, withvb = TRUE))
)

### update() of trimfill results -------------------------------------------------------------------------------------------------------------
tf <- trimfill(rma(yi, vi, data = dat))
tf_nofill <- trimfill(rma(yi, vi, data = dat, subset = ablat > 40))
upd <- function(x) list(beta = num(x$beta), se = num(x$se), tau2 = x$tau2, k = x$k, method = x$method, print = printed(x), class = class(x))
out$update_trimfill <- list(
  ml = upd(update(tf, method = "ML")), same = upd(update(tf)), knha = upd(update(tf, test = "knha")), nofill = upd(update(tf_nofill, method = "ML")),
  k0 = tf$k0
)

### gosh() with a seed -----------------------------------------------------------------------------------------------------------------------
gsh <- function(x, ...) {
  r <- quiet(gosh(x, progbar = FALSE, ...))
  list(res = lapply(r$res, num), incl = lapply(seq_len(nrow(r$incl)), function(i) as.integer(r$incl[i, ])), print = printed(r),
       dbinom = sprintf("%.17g", dbinom(x$p:x$k, x$k, 0.5)))
}
out$gosh <- list(
  bcg13 = gsh(rma(yi, vi, data = dat), subsets = 60, seed = 1),
  bcg13_seed2 = gsh(rma(yi, vi, data = dat), subsets = 40, seed = 2),
  bcg13_mods = gsh(rma(yi, vi, mods = ~ablat, data = dat), subsets = 50, seed = 3),
  bcg13_ml = gsh(rma(yi, vi, data = dat, method = "ML"), subsets = 30, seed = 4),
  hackshaw = gsh(rma(yi, vi, data = hack), subsets = 80, seed = 5),
  hackshaw_mods = gsh(rma(yi, vi, mods = ~year, data = hack), subsets = 80, seed = 6)
)
out$gosh$dbinom_k <- lapply(c(2, 5, 8, 13, 16, 24, 28, 37, 90, 210, 400), function(k) list(k = k, x = 0:k, hex = sprintf("%a", dbinom(0:k, k, 0.5))))

### location-scale standard errors ---------------------------------------------------------------------------------------------------------
ls_fit <- function(...) {
  r <- quiet(rma(yi, vi, ...))
  list(alpha = num(r$alpha), se_alpha = num(r$se.alpha), beta = num(r$beta), se = num(r$se), ll = num(r$fit.stats["ll", "REML"]), vb_alpha = mat(r$va))
}
out$ls_se <- list(
  bcg_ablat = ls_fit(scale = ~ablat, data = dat), bcg_ablat_loc = ls_fit(mods = ~ablat, scale = ~ablat, data = dat),
  bcg_year = ls_fit(scale = ~I(year - 1950), data = dat), hack_year = ls_fit(scale = ~year, data = hack),
  bcg_ml = ls_fit(scale = ~ablat, data = dat, method = "ML"), bcg_ablat_r8 = {
    r <- quiet(rma(yi, vi, scale = ~ablat, data = dat, control = list(hessianCtrl = list(r = 8))))
    list(se_alpha = num(r$se.alpha))
  }
)

### printed confint() and anova() tables, result fields, predict() aliases, confint(control=) --------------------------------------------------
c0 <- rma(yi, vi, data = dat)
c1 <- rma(yi, vi, mods = ~ablat, data = dat)
c0ml <- rma(yi, vi, data = dat, method = "ML")
c1ml <- rma(yi, vi, mods = ~ablat, data = dat, method = "ML")
rob <- robust(uni1, cluster = district)
ctrl <- confint(c0, control = list(tau2.max = 50, tol = 1e-8))
printed80 <- function(x) {
  old <- options(width = 80)
  on.exit(options(old))
  printed(x)
}
out$small <- list(
  confint_random = printed(confint(c0)), confint_fixed = printed(confint(c1, fixed = TRUE)), confint_digits = printed(confint(c1, fixed = TRUE, digits = 2)),
  anova_lrt = printed80(quiet(anova(c0, c1))), anova_lrt_ml = printed80(anova(c0ml, c1ml)),
  confint_control = mat(ctrl$random),
  fields = list(k_eff = c0$k.eff, p_eff = c1$p.eff, k_all = c0$k.all, ni = num(c0$ni), formula_yi = if (is.null(c0$formula.yi)) NULL else "formula",
                formula_mods = deparse(c1$formula.mods)),
  meat = mat(rob$meat), cr = list(lb = num(predict(c1, newmods = 30)$cr.lb), ub = num(predict(c1, newmods = 30)$cr.ub), pi_lb = num(predict(c1, newmods = 30)$pi.lb))
)

write_json(out, "src/tests/reference/phase12/audit3.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null")
