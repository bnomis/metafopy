# Reference values for the gaps closed after the fourth audit that concern cluster-robust fits and rma.glmm: anova() of robust(..., clubSandwich = TRUE)
# fits (Wald test with the HTZ and the other small-sample approximations of clubSandwich::Wald_test), the old 'L' spelling and 'fixed' of anova(),
# anova(test = "Wald") and anova(df = ) for two models, blup()/ranef() of such fits (t quantiles with the Satterthwaite degrees of freedom of
# the coefficients, recycled over the studies by R) and rma.glmm(model = "CM.EL", method = "FE") with the optimisers of optim() and minqa
# (control$optimizer; method = "ML" needs lme4 in R, which is not used here).
#
# Usage (from the repository root; BiasedUrn and minqa must be installed for the last part):
#   LC_ALL=C.UTF-8 Rscript src/tests/r/generate_audit4_robust.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(clubSandwich)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))

# evaluate expr, collecting its warnings
caught <- function(expr) {
  warns <- character()
  res <- tryCatch(withCallingHandlers(expr, warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }),
                  error = function(e) structure(list(error = conditionMessage(e)), class = "caught.error"))
  list(value = res, warnings = I(warns))
}

out <- list()
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
kon <- dat.konstantopoulos2011

wald <- function(a) {
  r <- a$value
  if (inherits(r, "caught.error")) return(list(error = r$error, warnings = a$warnings))
  list(QM = r$QM, QMdf = num(r$QMdf), QMp = r$QMp, btt = num(r$btt), m = r$m, Xb = num(r$Xb), se = num(r$se), zval = num(r$zval), pval = num(r$pval),
       ddf = num(r$ddf), print = printed(r), warnings = a$warnings)
}

### anova() of clubSandwich fits ---------------------------------------------------------------------------------------------------------------
res1 <- rma(yi, vi, mods = ~ablat, data = dat)
res0 <- rma(yi, vi, data = dat)
res2 <- rma(yi, vi, mods = ~ablat + year, data = dat)
rb1 <- robust(res1, cluster = alloc, clubSandwich = TRUE)
rb0 <- robust(res0, cluster = alloc, clubSandwich = TRUE)
rb2 <- robust(res2, cluster = alloc, clubSandwich = TRUE)
out$anova_uni <- list(
  default = wald(caught(anova(rb1))),
  btt1 = wald(caught(anova(rb1, btt = 1))),
  btt12 = wald(caught(anova(rb1, btt = 1:2))),
  btt_name = wald(caught(anova(rb1, btt = "abl"))),
  btt_fixed = wald(caught(anova(rb1, btt = "ablat", fixed = TRUE))),
  intercept_only = wald(caught(anova(rb0))),
  x_vec = wald(caught(anova(rb1, X = c(1, 30)))),
  x_mat = wald(caught(anova(rb1, X = rbind(c(1, 30), c(1, 40))))),
  x_noint = wald(caught(anova(rb1, X = 30))),
  x_rhs = wald(caught(anova(rb1, X = c(1, 30), rhs = 1))),
  x_rank = wald(caught(anova(rb1, X = rbind(c(1, 30), c(2, 60))))),
  l_vec = wald(caught(anova(rb1, L = c(1, 30)))),
  btt_rhs = wald(caught(anova(rb1, btt = 2, rhs = 0.1))),
  btt_list = lapply(unclass(caught(anova(rb1, btt = list(1, 2)))$value), function(r) list(QM = r$QM, QMdf = num(r$QMdf), QMp = r$QMp)),
  btt_list_adjust = lapply(unclass(caught(anova(rb1, btt = list(1, 2), adjust = "bonferroni"))$value), function(r) list(QM = r$QM, QMdf = num(r$QMdf), QMp = r$QMp)),
  adjust = wald(caught(anova(rb1, btt = 2, adjust = TRUE))),
  adjust_x = wald(caught(anova(rb1, X = rbind(c(1, 30), c(1, 40)), adjust = "holm"))),
  p3_btt = wald(caught(anova(rb2))),
  p3_x = wald(caught(anova(rb2, X = rbind(c(1, 30, 70), c(1, 10, 70))))),
  p3_x_short = wald(caught(anova(rb2, X = rbind(c(30, 70), c(10, 70))))),
  wald_chisq = wald(caught(anova(robust(res1, cluster = alloc, clubSandwich = TRUE, wald_test = "chi-sq")))),
  wald_naivef = wald(caught(anova(robust(res1, cluster = alloc, clubSandwich = TRUE, wald_test = "Naive-F")))),
  wald_hta = wald(caught(anova(robust(res2, cluster = alloc, clubSandwich = TRUE, wald_test = "HTA")))),
  wald_edf = wald(caught(anova(robust(res2, cluster = alloc, clubSandwich = TRUE, wald_test = "EDF")))),
  coef_naive = wald(caught(anova(robust(res1, cluster = alloc, clubSandwich = TRUE, coef_test = "naive-t"), X = c(1, 30)))),
  coef_z = wald(caught(anova(robust(res1, cluster = alloc, clubSandwich = TRUE, coef_test = "z"), X = c(1, 30)))),
  cr1 = wald(caught(anova(robust(res1, cluster = alloc, clubSandwich = TRUE, vcov = "CR1"), btt = 2))),
  plain_robust = wald(caught(anova(robust(res1, cluster = alloc)))),
  plain_robust_x = wald(caught(anova(robust(res1, cluster = alloc), X = c(1, 30))))
)

# rma.mv fits made robust
mv1 <- rma.mv(yi, vi, mods = ~I(year - 1990), random = ~1 | district/school, data = kon)
rbm <- robust(mv1, cluster = district, clubSandwich = TRUE)
out$anova_mv <- list(
  default = wald(caught(anova(rbm))),
  x = wald(caught(anova(rbm, X = c(1, 5)))),
  x_mat = wald(caught(anova(rbm, X = rbind(c(1, -5), c(1, 5))))),
  btt1 = wald(caught(anova(rbm, btt = 1))),
  plain = wald(caught(anova(robust(mv1, cluster = district)))),
  l_vec = wald(caught(anova(rbm, L = c(1, 5))))
)

### the other ... arguments of anova() --------------------------------------------------------------------------------------------------------
fit_full <- rma(yi, vi, mods = ~ablat + year, data = dat, method = "ML")
fit_red <- rma(yi, vi, mods = ~ablat, data = dat, method = "ML")
fit_full_reml <- rma(yi, vi, mods = ~ablat + year, data = dat)
fit_red_reml <- rma(yi, vi, mods = ~ablat, data = dat)
two <- function(a) {
  r <- a$value
  if (inherits(r, "caught.error")) return(list(error = r$error, warnings = a$warnings))
  if (r$type == "LRT") list(type = "LRT", LRT = r$LRT, pval = r$pval, parms_f = r$parms.f, parms_r = r$parms.r, print = printed(r), warnings = a$warnings)
  else list(type = r$type, QM = r$QM, QMdf = num(r$QMdf), QMp = r$QMp, btt = num(r$btt), print = printed(r), warnings = a$warnings)
}
out$anova_two <- list(
  wald_reml = two(caught(anova(fit_full_reml, fit_red_reml, test = "Wald"))),
  wald_reml_swapped = two(caught(anova(fit_red_reml, fit_full_reml, test = "Wald"))),
  wald_ml = two(caught(anova(fit_full, fit_red, test = "Wald"))),
  wald_prefix = two(caught(anova(fit_full, fit_red, test = "W"))),
  wald_same = two(caught(anova(fit_red, fit_red, test = "Wald", df = 1))),
  wald_methods = two(caught(anova(fit_full_reml, fit_red, test = "Wald"))),
  lrt_df = two(caught(anova(fit_full, fit_red, df = 3))),
  lrt_df_same = two(caught(anova(fit_red, fit_red, df = 2))),
  lrt_default = two(caught(anova(fit_full, fit_red))),
  lrt_bad_test = two(caught(anova(fit_full, fit_red, test = "x")))
)

### blup() and ranef() of clubSandwich fits ------------------------------------------------------------------------------------------------------
blup_out <- function(a) {
  r <- a$value
  list(pred = num(r$pred), se = num(r$se), pi_lb = num(r$pi.lb), pi_ub = num(r$pi.ub), slab = r$slab, warnings = a$warnings)
}
dat12 <- dat[1:12, ]
res12 <- rma(yi, vi, mods = ~ablat, data = dat12)
rb12 <- robust(res12, cluster = alloc, clubSandwich = TRUE)
dat_na <- dat
dat_na$yi[3] <- NA
res_na <- rma(yi, vi, mods = ~ablat, data = dat_na)
rb_na <- robust(res_na, cluster = alloc, clubSandwich = TRUE)
res_t <- rma(yi, vi, mods = ~ablat, data = dat, test = "knha")
rb_t <- robust(res_t, cluster = alloc, clubSandwich = TRUE)
out$blup <- list(
  mods = list(blup = blup_out(caught(blup(rb1))), ranef = blup_out(caught(ranef(rb1))), level90 = blup_out(caught(blup(rb1, level = 90)))),
  mods_k12 = list(blup = blup_out(caught(blup(rb12))), ranef = blup_out(caught(ranef(rb12)))),
  intercept = list(blup = blup_out(caught(blup(rb0))), ranef = blup_out(caught(ranef(rb0)))),
  p3 = list(blup = blup_out(caught(blup(rb2)))),
  transf = blup_out(caught(blup(rb1, transf = exp))),
  cr1 = blup_out(caught(blup(robust(res1, cluster = alloc, clubSandwich = TRUE, vcov = "CR1")))),
  coef_z = blup_out(caught(blup(robust(res1, cluster = alloc, clubSandwich = TRUE, coef_test = "z")))),
  na_omit = blup_out(caught(blup(rb_na))),
  plain = blup_out(caught(blup(robust(res1, cluster = alloc)))),
  knha_club = blup_out(caught(blup(rb_t)))
)
op <- options(na.action = "na.exclude")
out$blup$na_exclude <- blup_out(caught(blup(rb_na)))
options(op)

### rma.glmm(model = "CM.EL") with other optimisers ----------------------------------------------------------------------------------------------
glmm_out <- function(a) {
  r <- a$value
  if (inherits(r, "caught.error")) return(list(error = r$error, warnings = a$warnings))
  list(beta = num(r$beta), se = num(r$se), vb = num(r$vb), tau2 = r$tau2, ll = num(r$fit.stats["ll", "ML"]), QE_Wld = r$QE.Wld, QE_LRT = r$QE.LRT,
       QEp_LRT = r$QEp.LRT, parms = r$parms, warnings = a$warnings)
}
bcg <- dat.bcg
fit_el <- function(optimizer, method = "FE", ..., ctrl = list()) {
  glmm_out(caught(rma.glmm(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = bcg, model = "CM.EL", method = method, ...,
                           control = c(list(optimizer = optimizer), ctrl))))
}
out$glmm <- list()
for (o in c("nlminb", "optim", "BFGS", "Nelder-Mead", "CG", "L-BFGS-B", "bobyqa", "uobyqa", "newuoa", "nlm")) out$glmm[[paste0("fe_", o)]] <- fit_el(o)
out$glmm$fe_optim_nm <- fit_el("optim", ctrl = list(optmethod = "Nelder-Mead"))
out$glmm$fe_optim_maxit <- fit_el("optim", ctrl = list(optCtrl = list(maxit = 5)))
out$glmm$fe_mods_bfgs <- fit_el("BFGS", mods = ~ablat)
out$glmm$fe_mods_nm <- fit_el("Nelder-Mead", mods = ~ablat)
out$glmm$bad_optimizer <- fit_el("nope")

write_json(out, "src/tests/reference/phase12/audit4_robust.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null")
