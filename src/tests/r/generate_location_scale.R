# Reference values for location-scale models, rma.uni(..., scale = ~ ...) (class "rma.ls"), on dat.bangertdrowns2004 (the help-page
# examples of rma.uni()), dat.bcg and dat.molloy2014: log and identity links, intercept-only and moderator scale models, location
# moderators, ML and REML, test = "z", "t" and "knha", optbeta = TRUE, fixed scale coefficients, unweighted and user weights, btt/att,
# missing values in the scale variables; with printed output, hat values, weights, residuals, BLUPs, standardized residuals,
# predict() (newmods, newscale, prediction intervals, transformations), confint() (profile likelihood CIs for the scale
# coefficients), profile(), permutest() (approximate and exact), anova() (Wald tests and likelihood ratio tests), and vif().
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_location_scale.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
mat <- function(x) {
  if (is.null(x)) {
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

core <- function(res, w = character()) {
  out <- list(
    beta = num(res$beta), se = num(res$se), zval = num(res$zval), pval = num(res$pval),
    ci_lb = num(res$ci.lb), ci_ub = num(res$ci.ub), vb = mat(res$vb),
    alpha = num(res$alpha), va = mat(res$va), se_alpha = num(res$se.alpha), zval_alpha = num(res$zval.alpha),
    pval_alpha = num(res$pval.alpha), ci_lb_alpha = num(res$ci.lb.alpha), ci_ub_alpha = num(res$ci.ub.alpha),
    alpha_fix = res$alpha.fix, beta_fix = if (res$optbeta) res$beta.fix else NULL,
    tau2 = num(res$tau2), I2 = num(res$I2), H2 = num(res$H2), vt = res$vt,
    QE = res$QE, QEp = res$QEp, QM = res$QM, QMdf = num(res$QMdf), QMp = res$QMp,
    QS = res$QS, QSdf = num(res$QSdf), QSp = res$QSp,
    k = res$k, p = res$p, q = res$q, m = res$m, m_alpha = res$m.alpha, parms = res$parms,
    fit_ml = num(res$fit.stats$ML), fit_reml = num(res$fit.stats$REML),
    ddf = num(res$ddf), ddf_alpha = num(res$ddf.alpha), s2w = res$s2w, test = res$test, method = res$method,
    link = res$link, optbeta = res$optbeta, weighted = res$weighted,
    int_only = res$int.only, int_incl = res$int.incl, Z_int_only = res$Z.int.only, Z_int_incl = res$Z.int.incl,
    coef_names = rownames(res$beta), alpha_names = rownames(res$alpha), btt = num(res$btt), att = num(res$att),
    level = res$level,
    print = printed(res), print_showfit = printed(res, showfit = TRUE),
    hat = num(hatvalues(res)), weights = num(weights(res)), residuals = num(residuals(res)),
    fitted_location = num(fitted(res)$location), fitted_scale = num(fitted(res)$scale),
    warnings = w
  )
  out
}

diag_extras <- function(res) {
  rs <- rstandard(res)
  bl <- blup(res)
  rf <- ranef(res)
  list(
    rstandard = list(resid = num(rs$resid), se = num(rs$se), z = num(rs$z)),
    blup = list(pred = num(bl$pred), se = num(bl$se), pi_lb = num(bl$pi.lb), pi_ub = num(bl$pi.ub), print = printed(bl)),
    ranef = list(pred = num(rf$pred), se = num(rf$se), pi_lb = num(rf$pi.lb), pi_ub = num(rf$pi.ub)),
    slab = as.character(res$slab)
  )
}

fit <- function(expr, extra = function(res) list()) {
  tryCatch(
    {
      r <- warns(expr)
      c(core(r$value, r$warnings), extra(r$value))
    },
    error = function(e) list(error = conditionMessage(e))
  )
}

pred_list <- function(pr) {
  out <- list(
    pred = num(pr$pred), se = num(pr$se), ci_lb = num(pr$ci.lb), ci_ub = num(pr$ci.ub),
    pi_lb = num(pr$pi.lb), pi_ub = num(pr$pi.ub), pred_type = pr$pred.type, ddf = num(pr$ddf),
    pi_se = num(pr$pi.se), pi_dist = pr$pi.dist, slab = as.character(pr$slab), print = printed(pr)
  )
  if (!is.null(pr$X)) out$X <- mat(pr$X)
  if (!is.null(pr$Z)) out$Z <- mat(pr$Z)
  if (!is.null(pr$prob)) out$prob <- num(pr$prob)
  out
}

out <- list()

### dat.bangertdrowns2004: the help-page examples ----------------------------------------------------------------------------

bd <- dat.bangertdrowns2004
bd$ni100 <- bd$ni / 100
out$bd_data <- frame_cols(bd[, c("author", "year", "ni", "ni100", "meta", "imag", "subject", "yi", "vi")])

re <- rma(yi, vi, data = bd)
out$bd_re <- list(tau2 = re$tau2, ll_reml = num(re$fit.stats$REML))

out$bd_int <- fit(rma(yi, vi, scale = ~ 1, data = bd), function(res) {
  c(diag_extras(res), list(
    predict_newscale_exp = pred_list(predict(res, newscale = 1, transf = exp)),
    predict_newscale = pred_list(predict(res, newscale = 1)),
    predict_default = pred_list(predict(res)),
    predict_newmods = pred_list(predict(res, newmods = 1)),
    confint_alpha = {
      ci <- confint(res, alpha = 1)
      list(estimate = num(ci$random[, "estimate"]), ci_lb = num(ci$random[, "ci.lb"]), ci_ub = num(ci$random[, "ci.ub"]), print = printed(ci))
    },
    confint_fixed = {
      ci <- confint(res, fixed = TRUE)
      list(estimate = num(ci$fixed[, "estimate"]), ci_lb = num(ci$fixed[, "ci.lb"]), ci_ub = num(ci$fixed[, "ci.ub"]))
    },
    profile = {
      pf <- profile(res, alpha = 1, steps = 12, plot = FALSE, progbar = FALSE)
      list(alpha = num(pf$alpha), ll = num(pf$ll), beta = mat(pf$beta), ci_lb = mat(pf$ci.lb), ci_ub = mat(pf$ci.ub), vc = pf$vc, maxll = pf$maxll, ylim = num(pf$ylim))
    },
    fitstats = num(unlist(fitstats(res)))
  ))
})
out$bd_int_ml <- fit(rma(yi, vi, scale = ~ 1, data = bd, method = "ML"))
out$bd_int_identity <- fit(rma(yi, vi, scale = ~ 1, data = bd, link = "identity"))

out$bd_ni <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ ni100, data = bd))
out$bd_ni_ml <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ ni100, data = bd, method = "ML"), function(res) {
  list(vif_loc = NULL)
})

out$bd_3 <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd), function(res) {
  ex <- diag_extras(res)
  pr <- predict(res)
  ex$predict_default <- pred_list(pr)
  ex$predict_addxz <- pred_list(predict(res, addx = TRUE, addz = TRUE))
  ex$predict_newmods <- pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1), c(2.5, 1))))
  ex$predict_newmods_noint <- pred_list(predict(res, newmods = c(1, 1), intercept = FALSE))
  ex$predict_newscale <- pred_list(predict(res, newscale = rbind(c(0.5, 0), c(1.5, 1))))
  ex$predict_newscale_exp <- pred_list(predict(res, newscale = rbind(c(0.5, 0), c(1.5, 1)), transf = exp))
  ex$predict_both <- pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1))))
  ex$predict_both_single <- pred_list(predict(res, newmods = c(1, 1), newscale = c(1, 1)))
  ex$predict_both_t <- pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1)), level = 90, adjust = TRUE))
  ex$predict_simple <- pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1)), predtype = "simple"))
  ex$predict_riley <- pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1)), predtype = "riley"))
  ex$predict_newvi <- pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1)), newvi = c(0.01, 0.02)))
  ex$predict_prob <- pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1)), prob = ">0.2"))
  ex$predict_vcov <- {
    pv <- predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1)), vcov = TRUE)
    list(vcov = mat(pv$vcov), pred = num(pv$pred$pred))
  }
  ex$predict_vcov_scale <- {
    pv <- predict(res, newscale = rbind(c(0.5, 0), c(1.5, 1)), vcov = TRUE)
    list(vcov = mat(pv$vcov), pred = num(pv$pred$pred))
  }
  ex$confint_all <- {
    ci <- confint(res)
    lapply(ci[names(ci) != "digits"], function(e) list(estimate = num(e$random[, "estimate"]), ci_lb = num(e$random[, "ci.lb"]), ci_ub = num(e$random[, "ci.ub"])))
  }
  ex$confint_fixed_alpha <- {
    ci <- confint(res, fixed = TRUE, alpha = 2, level = 90)
    list(fixed = mat(ci$fixed), random = num(ci$random), print = printed(ci))
  }
  ex$profile <- {
    pf <- profile(res, alpha = 2, steps = 10, plot = FALSE, progbar = FALSE)
    list(alpha = num(pf$alpha), ll = num(pf$ll), beta = mat(pf$beta), ci_lb = mat(pf$ci.lb), ci_ub = mat(pf$ci.ub), vc = pf$vc, maxll = pf$maxll, ylim = num(pf$ylim))
  }
  ex$anova_btt <- {
    a <- anova(res, btt = 3)
    list(QM = a$QM, QMdf = num(a$QMdf), QMp = a$QMp, print = printed(a))
  }
  ex$anova_att <- {
    a <- anova(res, att = 3)
    list(QS = a$QS, QSdf = num(a$QSdf), QSp = a$QSp, print = printed(a))
  }
  ex$anova_att_23 <- {
    a <- anova(res, att = 2:3)
    list(QS = a$QS, QSdf = num(a$QSdf), QSp = a$QSp, rhs0 = TRUE)
  }
  ex$anova_X <- {
    a <- anova(res, X = rbind(c(0, 1, 0), c(0, 0, 1)))
    list(QM = a$QM, QMdf = num(a$QMdf), QMp = a$QMp, Xb = num(a$Xb), se = num(a$se), zval = num(a$zval), pval = num(a$pval), print = printed(a))
  }
  ex$anova_Z <- {
    a <- anova(res, Z = rbind(c(0, 1, 0), c(0, 0, 1)), rhs = c(0, 1))
    list(QS = a$QS, QSdf = num(a$QSdf), QSp = a$QSp, Za = num(a$Za), se = num(a$se), zval = num(a$zval), pval = num(a$pval), print = printed(a))
  }
  ex$anova_Z1 <- {
    a <- anova(res, Z = c(0, 1, -1))
    list(QS = a$QS, QSdf = num(a$QSdf), QSp = a$QSp, Za = num(a$Za), se = num(a$se), zval = num(a$zval), pval = num(a$pval), print = printed(a))
  }
  ex$vif <- {
    v <- vif(res)
    list(beta = num(v$beta$vifs), beta_names = names(v$beta$vifs), alpha = num(v$alpha$vifs), alpha_names = names(v$alpha$vifs),
         beta_sif = num(sapply(v$beta$vif, function(e) e$sif)), print = printed(v))
  }
  ex$vif_table <- {
    v <- vif(res, table = TRUE)
    list(print = printed(v))
  }
  ex$vif_reest <- {
    v <- vif(res, reestimate = TRUE)
    list(beta = num(v$beta$vifs), alpha = num(v$alpha$vifs))
  }
  ex$vif_att <- {
    v <- vif(res, btt = 2, att = list(2, 2:3))
    list(beta = num(v$beta$vifs), alpha = num(v$alpha$vifs), print = printed(v))
  }
  ex$coef <- list(beta = num(coef(res)$beta), alpha = num(coef(res)$alpha), vcov_alpha = mat(vcov(res)$alpha))
  ex$se <- list(beta = num(se(res)$beta), alpha = num(se(res)$alpha))
  ex
})

out$bd_3_ml <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, method = "ML"))
out$bd_3_t <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, test = "t"), function(res) {
  list(
    predict_default = pred_list(predict(res)),
    predict_both = pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1)))),
    predict_newscale = pred_list(predict(res, newscale = rbind(c(0.5, 0), c(1.5, 1)))),
    predict_both_riley = pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1)), predtype = "riley")),
    predict_both_tt = pred_list(predict(res, newmods = rbind(c(0.5, 0), c(1.5, 1)), newscale = rbind(c(0.5, 0), c(1.5, 1)), predtype = "t")),
    confint_alpha2 = {
      ci <- confint(res, alpha = 2)
      list(estimate = num(ci$random[, "estimate"]), ci_lb = num(ci$random[, "ci.lb"]), ci_ub = num(ci$random[, "ci.ub"]))
    },
    anova_att = {
      a <- anova(res, att = 2:3)
      list(QS = a$QS, QSdf = num(a$QSdf), QSp = a$QSp, print = printed(a))
    },
    anova_Z = {
      a <- anova(res, Z = rbind(c(0, 1, 0), c(0, 0, 1)))
      list(QS = a$QS, QSdf = num(a$QSdf), QSp = a$QSp, Za = num(a$Za), se = num(a$se), zval = num(a$zval), pval = num(a$pval), print = printed(a))
    },
    anova_X = {
      a <- anova(res, X = rbind(c(0, 1, 0), c(0, 0, 1)))
      list(QM = a$QM, QMdf = num(a$QMdf), QMp = a$QMp, Xb = num(a$Xb), se = num(a$se), zval = num(a$zval), pval = num(a$pval), ddf = num(a$ddf), print = printed(a))
    }
  )
})
out$bd_3_knha <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, test = "knha"))
out$bd_3_adhoc <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, test = "adhoc"))
out$bd_3_btt <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, btt = 3, att = c(2, 3)))
out$bd_3_level <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, level = 90))
out$bd_3_unweighted <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, weighted = FALSE))
out$bd_3_weights <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, weights = ni))
out$bd_3_alpha <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, alpha = c(NA, 0, NA)), function(res) {
  list(confint_alpha1 = {
    ci <- confint(res, alpha = 1)
    list(estimate = num(ci$random[, "estimate"]), ci_lb = num(ci$random[, "ci.lb"]), ci_ub = num(ci$random[, "ci.ub"]))
  })
})
out$bd_3_alpha_ml <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, alpha = c(NA, 0, NA), method = "ML"))
out$bd_3_alpha_ml_all <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, alpha = c(NA, 0, 0), method = "ML"))
out$bd_3_ml_full_for_lrt <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, method = "ML"))

out$bd_optbeta <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bd, method = "ML", optbeta = TRUE), function(res) {
  list(vcovmat = mat(res$vcovmat), confint_error = tryCatch(confint(res), error = function(e) conditionMessage(e)),
       profile_error = tryCatch(profile(res, plot = FALSE, progbar = FALSE), error = function(e) conditionMessage(e)))
})
out$bd_optbeta_reml <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ ni100, data = bd, optbeta = TRUE))
out$bd_optbeta_beta <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ 1, data = bd, optbeta = TRUE, beta = c(NA, NA, 0.1), method = "ML"))

out$bd_identity <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ imag, data = bd, link = "identity"), function(res) {
  list(
    predict_newscale = pred_list(predict(res, newscale = c(0, 1))),
    confint_alpha1 = {
      ci <- confint(res, alpha = 1)
      list(estimate = num(ci$random[, "estimate"]), ci_lb = num(ci$random[, "ci.lb"]), ci_ub = num(ci$random[, "ci.ub"]))
    }
  )
})
out$bd_identity_ml <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ imag, data = bd, link = "identity", method = "ML"))
out$bd_identity_ni <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ meta + imag, data = bd, link = "identity"))

bdna <- bd
bdna$imag[c(3, 10)] <- NA
bdna$ni100[20] <- NA
out$bd_na <- fit(rma(yi, vi, mods = ~ ni100 + meta, scale = ~ imag + ni100, data = bdna), function(res) {
  list(predict_default = pred_list(predict(res)), diag = diag_extras(res))
})

# scale given as a matrix (no intercept is added), mods as a matrix
Zm <- cbind(1, bd$ni100, bd$imag)
colnames(Zm) <- c("i", "n", "m")
out$bd_matrix <- fit(rma(bd$yi, bd$vi, mods = cbind(bd$ni100), scale = Zm))
out$bd_poly <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ ni100 + I(ni100^2), data = bd))
out$bd_alpha_bounds <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ ni100, data = bd, control = list(alpha.min = c(-5, -3), alpha.max = c(0, 3))))
out$bd_alpha_init <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ ni100, data = bd, control = list(alpha.init = c(-2, -0.5))))
out$bd_noscaleZ <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ ni100 + imag, data = bd, control = list(scaleZ = FALSE)))

# likelihood ratio tests (ML), on complete cases so that all models share the same studies
bdc <- bd[complete.cases(bd[, c("yi", "vi", "ni100", "meta", "imag")]), ]
mf <- rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bdc, method = "ML")
mr1 <- rma(yi, vi, mods = ~ ni100 + meta, scale = ~ 1, data = bdc, method = "ML")
mr2 <- rma(yi, vi, mods = ~ 1, scale = ~ ni100 + imag, data = bdc, method = "ML")
mr3 <- rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bdc, method = "ML", alpha = c(NA, 0, NA))
lrt <- function(a) {
  list(LRT = a$LRT, pval = a$pval, fit_stats_f = num(unlist(a$fit.stats.f)), fit_stats_r = num(unlist(a$fit.stats.r)), parms = c(a$parms.f, a$parms.r),
       QE = c(a$QE.f, a$QE.r), print = printed(a))
}
out$lrt_scale <- lrt(anova(mf, mr1))
out$lrt_location <- lrt(anova(mf, mr2))
out$lrt_alpha_fixed <- lrt(anova(mf, mr3))
out$lrt_swapped <- lrt(anova(mr1, mf))
mfr <- rma(yi, vi, mods = ~ ni100 + meta, scale = ~ ni100 + imag, data = bdc)
mrr <- rma(yi, vi, mods = ~ ni100 + meta, scale = ~ 1, data = bdc)
out$lrt_reml <- lrt(anova(mfr, mrr))

# permutation tests
pt <- function(p) {
  list(
    pval = num(p$pval), QMp = p$QMp, QM = p$QM, QMdf = num(p$QMdf), pval_alpha = num(p$pval.alpha), QSp = p$QSp, QS = p$QS, QSdf = num(p$QSdf),
    exact_iter = num(p$exact.iter), Z_exact_iter = num(p$Z.exact.iter), skip_beta = p$skip.beta, skip_alpha = p$skip.alpha,
    QM_perm = num(p$QM.perm), QS_perm = num(p$QS.perm), zval_perm = mat(p$zval.perm), zval_alpha_perm = mat(p$zval.alpha.perm),
    beta_perm = mat(p$beta.perm), alpha_perm = mat(p$alpha.perm), print = printed(p)
  )
}
bc <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
out$bcg_data <- frame_cols(bc[, c("trial", "author", "year", "ablat", "alloc", "yi", "vi")])
bcm <- rma(yi, vi, mods = ~ ablat, scale = ~ ablat, data = bc)
out$bcg_ablat <- fit(bcm, function(res) {
  list(
    permutest = pt(permutest(res, iter = 100, seed = 1234, progbar = FALSE)),
    permutest_beta = pt(permutest(res, iter = 60, seed = 99, progbar = FALSE, skip.alpha = TRUE)),
    permutest_alpha = pt(permutest(res, iter = 60, seed = 99, progbar = FALSE, skip.beta = TRUE)),
    permutest_btt = pt(permutest(res, iter = 40, seed = 5, progbar = FALSE, btt = 2, att = 2)),
    permutest_less = pt(permutest(res, iter = 50, seed = 7, progbar = FALSE, control = list(alternative = "less", stat = "coef"))),
    permutest_px2 = pt(permutest(res, iter = 50, seed = 7, progbar = FALSE, control = list(p2defn = "px2"))),
    permutest_iter = num(permutest(res, exact = "i", progbar = FALSE)),
    predict_default = pred_list(predict(res)),
    predict_both_exp = pred_list(predict(res, newmods = c(20, 40), newscale = c(20, 40), transf = exp)),
    diag = diag_extras(res)
  )
})
out$bcg_ablat_ml <- fit(rma(yi, vi, mods = ~ ablat, scale = ~ ablat, data = bc, method = "ML"))
out$bcg_ablat_t <- fit(rma(yi, vi, mods = ~ ablat, scale = ~ ablat, data = bc, test = "t"), function(res) {
  list(permutest = pt(permutest(res, iter = 50, seed = 321, progbar = FALSE)))
})
out$bcg_int_scale <- fit(rma(yi, vi, mods = ~ ablat, scale = ~ 1, data = bc), function(res) {
  list(permutest = pt(permutest(res, iter = 50, seed = 11, progbar = FALSE)))
})
out$bcg_year_identity <- fit(rma(yi, vi, mods = ~ ablat, scale = ~ I(ablat / 10), data = bc, link = "identity", method = "ML"))

# exact permutation tests on a small data set (6 studies of dat.bangertdrowns2004 for which the scale model is identified)
bd6 <- bdc[29:34, ]
out$bd6_data <- frame_cols(bd6[, c("author", "year", "ni", "ni100", "meta", "imag", "subject", "yi", "vi")])
out$bd6_exact <- fit(rma(yi, vi, mods = ~ ni100, scale = ~ ni100, data = bd6), function(res) {
  list(permutest = pt(permutest(res, exact = TRUE, progbar = FALSE)), permutest_iter = num(permutest(res, exact = "i", progbar = FALSE)))
})

# dat.molloy2014 (correlations)
mo <- escalc(measure = "ZCOR", ri = ri, ni = ni, data = dat.molloy2014)
out$mo_data <- frame_cols(mo[, c("authors", "year", "ni", "ri", "controls", "design", "meanage", "quality", "yi", "vi")])
out$mo_int <- fit(rma(yi, vi, scale = ~ 1, data = mo), function(res) {
  list(predict_newscale_exp = pred_list(predict(res, newscale = 1, transf = exp)), diag = diag_extras(res))
})
out$mo_quality <- fit(rma(yi, vi, mods = ~ quality, scale = ~ quality, data = mo, method = "ML"))
out$mo_design <- fit(rma(yi, vi, scale = ~ design, data = mo), function(res) {
  list(predict_newscale = pred_list(predict(res, newscale = c(1, 0, 0), transf = exp)))
})

# dat.mcdaniel1994 (factors in location and scale parts)
mc <- escalc(measure = "ZCOR", ri = ri, ni = ni, data = dat.mcdaniel1994)
mc <- mc[1:60, ]
out$mc_data <- frame_cols(mc[, c("study", "ni", "ri", "type", "struct", "yi", "vi")])
out$mc_factors <- fit(rma(yi, vi, mods = ~ type + struct, scale = ~ struct, data = mc, method = "ML"), function(res) {
  v <- vif(res)
  list(vif = list(beta = num(v$beta$vifs)), diag = diag_extras(res))
})

out$meta <- list(metafor = as.character(packageVersion("metafor")), metadat = as.character(packageVersion("metadat")), R = R.version.string)
dir.create("src/tests/reference/phase5", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase5/location_scale.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase5/location_scale.json\n")
