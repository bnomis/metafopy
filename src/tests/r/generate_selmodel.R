# Reference values for selmodel() on dat.hackshaw1998, dat.bangertdrowns2004 and dat.raudenbush1985:
# every continuous selection function (beta, halfnorm, negexp, logistic, power, negexppow and the halfnorm1/2, negexp1/2,
# logistic1/2, power1/2 variants), step function models (alternative greater/less/two.sided, fixed and partly fixed deltas,
# decreasing=TRUE, a precision measure, subsets), truncated models, equal-effects and mixed-effects inputs (also fitted with
# REML or at level 90), precision measures, fixed tau^2 and beta values, optimizer and Hessian control settings, with printed
# output, the Wald/LRT tests, fit statistics, predict(), confint(), profile(), anova() and the data behind plot().
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_selmodel.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
mat <- function(x) {
  if (is.null(x) || length(x) == 1L && is.na(x)) {
    return(NULL)
  }
  x <- as.matrix(x)
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x))
}
frame_cols <- function(d) lapply(as.list(as.data.frame(d)), function(col) if (is.numeric(col)) num(col) else as.character(col))
warns <- function(expr) {
  w <- character()
  value <- withCallingHandlers(expr, warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = w)
}
table_cols <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  out <- lapply(as.list(as.data.frame(x)), num)
  out$.rownames <- rownames(x)
  out
}

hack <- dat.hackshaw1998
bang <- dat.bangertdrowns2004
rau <- dat.raudenbush1985
datasets <- list(hack = hack, bang = bang, rau = rau)

core <- function(res, w = character()) {
  list(
    beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval),
    ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub), vb = mat(res$vb),
    tau2 = res$tau2, se_tau2 = res$se.tau2, ci_lb_tau2 = res$ci.lb.tau2, ci_ub_tau2 = res$ci.ub.tau2,
    QM = res$QM, QMp = res$QMp,
    delta = num(res$delta), vd = mat(res$vd), se_delta = num(res$se.delta), zval_delta = num(res$zval.delta),
    pval_delta = num(res$pval.delta), ci_lb_delta = num(res$ci.lb.delta), ci_ub_delta = num(res$ci.ub.delta),
    deltas = res$deltas, delta_fix = res$delta.fix,
    ll = res$ll, ll0 = res$ll0, LRT = res$LRT, LRTdf = res$LRTdf, LRTp = res$LRTp,
    LRT_tau2 = res$LRT.tau2, LRTp_tau2 = res$LRTp.tau2,
    parms = res$parms, fit_ml = num(res$fit.stats$ML), fit_reml = num(res$fit.stats$REML),
    pvals = num(res$pvals), k = res$k, k0 = res$k0, k1 = res$k1, p = res$p,
    ptable = table_cols(if (is.data.frame(res$ptable)) res$ptable else NULL),
    precis = as.list(res$precis), hessian = mat(res$hessian), hest = res$hest,
    type = res$type, steps = num(res$steps), stepsspec = res$stepsspec, decreasing = res$decreasing,
    alternative = res$alternative, method = res$method, tau2_fix = res$tau2.fix, test = res$test,
    delta_min = num(res$delta.min), delta_max = num(res$delta.max), pval_min = res$pval.min,
    coef_names = rownames(res$beta), print = printed(res), warnings = w
  )
}

fit_case <- function(cs) {
  dat <- datasets[[cs$data]]
  rma_args <- list(yi = dat$yi, vi = dat$vi, data = dat)
  rma_args <- c(rma_args, cs$rma)
  if (!is.null(rma_args$mods)) rma_args$mods <- as.formula(rma_args$mods)
  res0 <- do.call(rma, rma_args)
  r <- warns(do.call(selmodel, c(list(res0), cs$sel)))
  list(res0 = res0, res = r$value, warnings = r$warnings)
}

cases <- list(
  hack_beta = list(data = "hack", rma = list(method = "ML"), sel = list(type = "beta")),
  hack_beta_steps = list(data = "hack", rma = list(method = "ML"), sel = list(type = "beta", steps = c(0.01, 0.99))),
  hack_beta_fixdelta = list(data = "hack", rma = list(method = "ML"), sel = list(type = "beta", delta = c(1, NA))),
  hack_halfnorm = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm")),
  hack_halfnorm_s = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm", steps = 0.025)),
  hack_halfnorm_less = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm", alternative = "less")),
  hack_halfnorm_two = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm", alternative = "two.sided", steps = 0.05)),
  hack_halfnorm_sei = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm", prec = "sei")),
  hack_halfnorm_fixdelta = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm", delta = 2)),
  hack_halfnorm_nm = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm", control = list(optmethod = "Nelder-Mead"))),
  hack_halfnorm_init = list(
    data = "hack", rma = list(method = "ML"),
    sel = list(type = "halfnorm", control = list(delta.init = 3, tau2.init = 0.1, delta.max = 10))
  ),
  hack_halfnorm_htransf = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm", control = list(htransf = TRUE))),
  hack_halfnorm_ee = list(data = "hack", rma = list(method = "EE"), sel = list(type = "halfnorm")),
  hack_negexp = list(data = "hack", rma = list(method = "ML"), sel = list(type = "negexp")),
  hack_negexp_vi = list(data = "hack", rma = list(method = "ML"), sel = list(type = "negexp", prec = "vi")),
  hack_logistic = list(data = "hack", rma = list(method = "ML"), sel = list(type = "logistic")),
  hack_power = list(data = "hack", rma = list(method = "ML"), sel = list(type = "power")),
  hack_negexppow = list(data = "hack", rma = list(method = "ML"), sel = list(type = "negexppow", delta = c(NA, 1))),
  hack_halfnorm1 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm1", steps = 0.05)),
  hack_negexp1 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "negexp1", steps = 0.05)),
  hack_logistic1 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "logistic1", steps = 0.05)),
  hack_power1 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "power1", steps = 0.05)),
  hack_halfnorm2 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "halfnorm2", steps = 0.05, delta = c(NA, 1))),
  hack_negexp2 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "negexp2", steps = 0.05, delta = c(NA, 1))),
  hack_logistic2 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "logistic2", steps = 0.05, delta = c(NA, 1))),
  hack_power2 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "power2", steps = 0.05, delta = c(NA, 1))),
  hack_stepfun_a = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.5))),
  hack_stepfun_b = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = 0.025)),
  hack_stepfun_less = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.5), alternative = "less")),
  hack_stepfun_two = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.05, 0.5), alternative = "two.sided")),
  hack_stepfun_fixed = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.5), delta = c(1, 0.5, 0.2))),
  hack_stepfun_partial = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.5), delta = c(NA, NA, 0.3))),
  hack_stepfun_decr = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.5), decreasing = TRUE)),
  hack_stepfun_deltamax = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.5), control = list(delta.max = 1))),
  hack_stepfun_sei = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = 0.025, prec = "sei")),
  hack_stepfun_ee = list(data = "hack", rma = list(method = "EE"), sel = list(type = "stepfun", steps = c(0.025, 0.5))),
  hack_stepfun_tau2 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = 0.025, tau2 = 0.05)),
  hack_stepfun_reml = list(data = "hack", rma = list(method = "REML"), sel = list(type = "stepfun", steps = 0.025)),
  hack_stepfun_level90 = list(data = "hack", rma = list(method = "ML", level = 90), sel = list(type = "stepfun", steps = 0.025)),
  hack_stepfun_subset = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = 0.025, subset = hack$year < 1995)),
  hack_trunc = list(data = "hack", rma = list(method = "ML"), sel = list(type = "trunc")),
  hack_trunc_less = list(data = "hack", rma = list(method = "ML"), sel = list(type = "trunc", steps = 0.1, alternative = "less")),
  hack_truncest = list(data = "hack", rma = list(method = "ML"), sel = list(type = "truncest", delta = c(NA, 0.1))),
  bang_stepfun_int = list(data = "bang", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.5))),
  bang_stepfun_mods = list(data = "bang", rma = list(method = "ML", mods = "~ grade + meta"), sel = list(type = "stepfun", steps = 0.025)),
  bang_stepfun_beta = list(data = "bang", rma = list(method = "ML", mods = "~ grade"), sel = list(type = "stepfun", steps = 0.025, beta = c(NA, 0.05))),
  bang_halfnorm_mods = list(data = "bang", rma = list(method = "ML", mods = "~ grade + imag"), sel = list(type = "halfnorm", steps = 0.025)),
  bang_halfnorm_ninv = list(data = "bang", rma = list(method = "ML"), sel = list(type = "halfnorm", prec = "ninv")),
  bang_negexp_sqrtninv = list(data = "bang", rma = list(method = "ML"), sel = list(type = "negexp", prec = "sqrtninv", steps = 0.025)),
  rau_halfnorm_mods = list(data = "rau", rma = list(method = "ML", mods = "~ weeks"), sel = list(type = "halfnorm")),
  rau_stepfun = list(data = "rau", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.1, 0.5))),
  rau_stepfun_mods = list(data = "rau", rma = list(method = "ML", mods = "~ weeks + tester"), sel = list(type = "stepfun", steps = 0.1)),
  rau_stepfun_red = list(data = "rau", rma = list(method = "ML", mods = "~ weeks"), sel = list(type = "stepfun", steps = 0.1))
)

out <- list(hack_data = frame_cols(hack), bang_data = frame_cols(bang), rau_data = frame_cols(rau))
out$specs <- lapply(cases, function(cs) list(data = cs$data, rma = cs$rma, sel = cs$sel))
fits <- list()
out$fits <- list()
for (nm in names(cases)) {
  message("fitting ", nm)
  r <- tryCatch(
    {
      f <- fit_case(cases[[nm]])
      fits[[nm]] <- f
      core(f$res, f$warnings)
    },
    error = function(e) list(error = conditionMessage(e))
  )
  out$fits[[nm]] <- r
}

### p-value tables only
out$ptable <- list(
  hack = table_cols(selmodel(fits$hack_stepfun_a$res0, type = "stepfun", steps = c(0.025, 0.1, 0.5), ptable = TRUE)),
  bang = table_cols(selmodel(fits$bang_stepfun_mods$res0, type = "stepfun", steps = c(0.001, 0.025, 0.1, 0.5), ptable = TRUE, alternative = "two.sided"))
)

### predict() and the accessors
pred_cols <- function(p) list(pred = num(p$pred), se = num(p$se), ci_lb = num(p$ci.lb), ci_ub = num(p$ci.ub), pi_lb = num(p$pi.lb), pi_ub = num(p$pi.ub))
out$predict <- list(
  hack_stepfun_b = pred_cols(predict(fits$hack_stepfun_b$res)),
  hack_stepfun_a_level = pred_cols(predict(fits$hack_stepfun_a$res, level = 90)),
  hack_stepfun_ee = pred_cols(predict(fits$hack_stepfun_ee$res)),
  rau_halfnorm_mods = pred_cols(predict(fits$rau_halfnorm_mods$res, newmods = c(0, 10, 20))),
  bang_stepfun_mods = pred_cols(predict(fits$bang_stepfun_mods$res)),
  bang_stepfun_mods_new = pred_cols(predict(fits$bang_stepfun_mods$res, newmods = rbind(c(2, 0), c(3, 1))))
)
out$accessors <- list(
  coef_delta = num(coef(fits$hack_stepfun_a$res)$delta), vcov_delta = mat(vcov(fits$hack_stepfun_a$res)$delta),
  se_delta = num(se(fits$hack_stepfun_a$res)$delta), logLik = c(logLik(fits$hack_stepfun_a$res)), AIC = AIC(fits$hack_stepfun_a$res),
  BIC = BIC(fits$hack_stepfun_a$res), fitstats = num(fitstats(fits$hack_stepfun_a$res)),
  fitstats_two = mat(fitstats(fits$hack_stepfun_a$res, fits$hack_stepfun_b$res))
)

### confint()
ci_cols <- function(ci) {
  list(
    random = if (!is.null(ci$random)) mat(ci$random) else NULL, random_names = rownames(ci$random),
    fixed = if (!is.null(ci$fixed)) mat(ci$fixed) else NULL, fixed_names = rownames(ci$fixed),
    lb_sign = ci$lb.sign, ub_sign = ci$ub.sign
  )
}
ci_all <- function(res, ...) {
  ci <- warns(confint(res, ...))
  v <- ci$value
  if (inherits(v, "list.confint.rma")) {
    v <- v[names(v) != "digits"]
    list(list = lapply(v, ci_cols), warnings = ci$warnings)
  } else {
    list(list = list(ci_cols(v)), warnings = ci$warnings)
  }
}
out$confint <- list(
  hack_stepfun_b = ci_all(fits$hack_stepfun_b$res),
  hack_stepfun_b_fixed = ci_all(fits$hack_stepfun_b$res, fixed = TRUE),
  hack_stepfun_b_delta = ci_all(fits$hack_stepfun_b$res, delta = 2, level = 90),
  hack_stepfun_b_tau2 = ci_all(fits$hack_stepfun_b$res, tau2 = 1),
  hack_stepfun_a = ci_all(fits$hack_stepfun_a$res),
  hack_stepfun_ee = ci_all(fits$hack_stepfun_ee$res),
  rau_stepfun_mods = ci_all(fits$rau_stepfun_mods$res),
  rau_halfnorm_mods_delta = ci_all(fits$rau_halfnorm_mods$res, delta = 1)
)

### profile()
prof_cols <- function(pr, comp) {
  list(
    grid = num(pr[[comp]]), ll = num(pr$ll), beta = frame_cols(pr$beta), ci_lb = frame_cols(pr$ci.lb), ci_ub = frame_cols(pr$ci.ub),
    ylim = num(pr$ylim), maxll = pr$maxll, vc = pr$vc, method = pr$method
  )
}
prof_one <- function(res, comp, ...) {
  pr <- warns(profile(res, ..., plot = FALSE, progbar = FALSE))
  list(prof = prof_cols(pr$value, comp), warnings = pr$warnings)
}
out$profile <- list(
  hack_stepfun_b_tau2 = prof_one(fits$hack_stepfun_b$res, "tau2", tau2 = 1, steps = 6),
  hack_stepfun_b_delta = prof_one(fits$hack_stepfun_b$res, "delta", delta = 2, steps = 6),
  hack_stepfun_b_delta_xlim = prof_one(fits$hack_stepfun_b$res, "delta", delta = 2, xlim = c(0.1, 1.5), steps = 5),
  rau_halfnorm_mods_delta = prof_one(fits$rau_halfnorm_mods$res, "delta", delta = 1, steps = 4)
)

### anova(): Wald tests and likelihood-ratio tests
wald_cols <- function(a) list(QM = a$QM, QMdf = num(a$QMdf), QMp = a$QMp, btt = num(a$btt), m = a$m, Xb = num(a$Xb), se = num(a$se), zval = num(a$zval), pval = num(a$pval))
lrt_cols <- function(a) {
  list(
    fit_f = num(a$fit.stats.f), fit_r = num(a$fit.stats.r), parms_f = a$parms.f, parms_r = a$parms.r, LRT = a$LRT, pval = a$pval,
    tau2_f = a$tau2.f, tau2_r = a$tau2.r, R2 = a$R2, QE_f = a$QE.f, QE_r = a$QE.r
  )
}
out$anova <- list(
  bang_wald = wald_cols(anova(fits$bang_stepfun_mods$res, btt = 2)),
  bang_wald_all = wald_cols(anova(fits$bang_stepfun_mods$res)),
  bang_wald_X = wald_cols(anova(fits$bang_stepfun_mods$res, X = rbind(c(1, 2, 0), c(0, 1, -1)))),
  rau_lrt = lrt_cols(anova(fits$rau_stepfun_mods$res, fits$rau_stepfun_red$res)),
  hack_lrt = lrt_cols(anova(fits$hack_stepfun_a$res, fits$hack_stepfun_b$res))
)

### the data behind plot()
plot_cols <- function(res, ...) {
  pdf(NULL)
  on.exit(dev.off())
  d <- plot(res, ...)
  list(xs = num(d$xs), ys = num(d$ys), ys_lb = num(d$ys.lb), ys_ub = num(d$ys.ub))
}
out$plot <- list(
  hack_halfnorm_s = plot_cols(fits$hack_halfnorm_s$res, n = 51),
  hack_halfnorm_s_prec = plot_cols(fits$hack_halfnorm_sei$res, n = 51, prec = "min"),
  hack_halfnorm_s_wald = plot_cols(fits$hack_halfnorm_s$res, n = 51, ci = "wald"),
  hack_stepfun_a = plot_cols(fits$hack_stepfun_a$res),
  hack_stepfun_a_wald = plot_cols(fits$hack_stepfun_a$res, ci = "wald", scale = TRUE),
  hack_beta = plot_cols(fits$hack_beta$res, n = 51, xlim = c(0.01, 0.99)),
  hack_power = plot_cols(fits$hack_power$res, n = 51)
)

dir.create("src/tests/reference/phase5", showWarnings = FALSE, recursive = TRUE)
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = FALSE), "src/tests/reference/phase5/selmodel.json")
