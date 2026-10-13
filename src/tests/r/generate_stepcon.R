# Reference values for selmodel(..., type = "stepcon"): step function selection models whose weights are constrained to be non-increasing
# (delta[1] = 1 >= delta[2] >= ...), fitted by Rsolnp::solnp (metafor's default for this type) on dat.hackshaw1998, dat.bangertdrowns2004 and
# dat.raudenbush1985: one to three steps, alternative greater/two.sided, mixed-effects and equal-effects inputs, moderators, fixed tau^2,
# partly fixed deltas, with printed output and the Wald/LRT tests, plus the same models as unconstrained
# type = "stepfun" fits for comparison.
#
# Requires the Rsolnp package.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_stepcon.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
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
  hack_stepcon_a = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepcon", steps = c(0.025, 0.5))),
  hack_stepcon_b = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepcon", steps = 0.025)),
  hack_stepcon_c = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepcon", steps = c(0.025, 0.1, 0.5))),
  hack_stepcon_two = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepcon", steps = c(0.05, 0.5), alternative = "two.sided")),
  hack_stepcon_ee = list(data = "hack", rma = list(method = "EE"), sel = list(type = "stepcon", steps = 0.025)),
  hack_stepcon_reml = list(data = "hack", rma = list(method = "REML"), sel = list(type = "stepcon", steps = 0.025)),
  hack_stepcon_tau2 = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepcon", steps = 0.025, tau2 = 0.05)),
  hack_stepcon_fixed = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepcon", steps = c(0.025, 0.5), delta = c(1, NA, 0.2))),
  hack_stepcon_level90 = list(data = "hack", rma = list(method = "ML", level = 90), sel = list(type = "stepcon", steps = 0.025)),
  hack_stepcon_subset = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepcon", steps = 0.025, subset = hack$year < 1995)),
  hack_stepcon_prec = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepcon", steps = 0.025, prec = "sei")),
  bang_stepcon_int = list(data = "bang", rma = list(method = "ML"), sel = list(type = "stepcon", steps = c(0.025, 0.5))),
  bang_stepcon_mods = list(data = "bang", rma = list(method = "ML", mods = "~ grade + meta"), sel = list(type = "stepcon", steps = 0.025)),
  rau_stepcon = list(data = "rau", rma = list(method = "ML"), sel = list(type = "stepcon", steps = c(0.025, 0.1, 0.5))),
  rau_stepcon_mods = list(data = "rau", rma = list(method = "ML", mods = "~ weeks"), sel = list(type = "stepcon", steps = 0.1)),
  hack_stepfun_a = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.5))),
  hack_stepfun_c = list(data = "hack", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.1, 0.5))),
  bang_stepfun_int = list(data = "bang", rma = list(method = "ML"), sel = list(type = "stepfun", steps = c(0.025, 0.5)))
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
out$ptable <- table_cols(selmodel(fits$hack_stepcon_a$res0, type = "stepcon", steps = c(0.025, 0.1, 0.5), ptable = TRUE))

### predict(), the accessors and confint()
pred_cols <- function(p) list(pred = num(p$pred), se = num(p$se), ci_lb = num(p$ci.lb), ci_ub = num(p$ci.ub), pi_lb = num(p$pi.lb), pi_ub = num(p$pi.ub))
out$predict <- list(
  hack_stepcon_a = pred_cols(predict(fits$hack_stepcon_a$res)),
  bang_stepcon_mods = pred_cols(predict(fits$bang_stepcon_mods$res, newmods = rbind(c(2, 0), c(3, 1))))
)
out$accessors <- list(
  coef_delta = num(coef(fits$hack_stepcon_a$res)$delta), vcov_delta = mat(vcov(fits$hack_stepcon_a$res)$delta),
  logLik = c(logLik(fits$hack_stepcon_a$res)), AIC = AIC(fits$hack_stepcon_a$res), BIC = BIC(fits$hack_stepcon_a$res)
)
out$confint_error <- tryCatch(
  {
    confint(fits$hack_stepcon_a$res)
    NULL
  },
  error = function(e) conditionMessage(e)
)
out$profile_error <- tryCatch(
  {
    profile(fits$hack_stepcon_a$res, tau2 = 1, plot = FALSE, progbar = FALSE)
    NULL
  },
  error = function(e) conditionMessage(e)
)

dir.create("src/tests/reference/phase6", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase6/stepcon.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null")
