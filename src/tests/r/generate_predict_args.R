# Reference values for the predict() arguments hetvar, prob, addx, vcov, adjust (and tau2.levels, which rma.uni ignores)
# for rma.uni, rma.glmm and rma.mv fits.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_predict_args.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
pr <- function(p) {
  if (!is.null(p$vcov)) {
    v <- unclass(p$vcov)
    p <- p$pred
  } else {
    v <- NULL
  }
  list(pred = num(p$pred), se = num(p$se), ci_lb = num(p$ci.lb), ci_ub = num(p$ci.ub), pi_lb = num(p$pi.lb), pi_ub = num(p$pi.ub),
       prob = num(p$prob), X = if (is.null(p$X)) NULL else unname(p$X), vcov = if (is.null(v)) NULL else unname(as.matrix(v)))
}

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, mods = ~ablat, data = dat)
res0 <- rma(yi, vi, data = dat)
resk <- rma(yi, vi, mods = ~ablat, data = dat, test = "knha")
resfe <- rma(yi, vi, data = dat, method = "FE")

out <- list()
out$uni_newmods <- pr(predict(res, newmods = c(10, 30, 50), hetvar = 0.1, prob = ">-0.5", addx = TRUE, vcov = TRUE))
out$uni_adjust <- pr(predict(res, newmods = c(10, 30, 50), adjust = TRUE, tau2.levels = 1))
out$uni_fitted <- pr(predict(res, prob = "<-0.5", addx = TRUE, vcov = TRUE))
out$uni_int <- pr(predict(res0, prob = "<0", hetvar = c(0.2)))
out$uni_knha <- pr(predict(resk, newmods = c(10, 50), prob = ">0", vcov = TRUE))
out$uni_fe <- pr(predict(resfe, prob = 0, vcov = TRUE))
out$uni_transf <- pr(predict(res, newmods = c(10, 50), transf = exp, vcov = TRUE, addx = TRUE, prob = ">0"))

dat2 <- dat.konstantopoulos2011
resmv <- rma.mv(yi, vi, mods = ~I(year - 1990), random = ~1 | district/school, data = dat2)
out$mv_newmods <- pr(predict(resmv, newmods = c(-10, 0, 10), prob = ">0", addx = TRUE, vcov = TRUE))
out$mv_hetvar <- pr(predict(resmv, newmods = c(-10, 0, 10), hetvar = 0.05, adjust = TRUE))

dat3 <- dat.berkey1998
V <- bldiag(lapply(split(dat3[, c("v1i", "v2i")], dat3$trial), as.matrix))
resun <- rma.mv(yi, V, mods = ~0 + outcome, random = ~outcome | trial, struct = "UN", data = dat3)
out$un_levels <- pr(predict(resun, newmods = diag(2), tau2.levels = c("AL", "PD"), prob = "<0.2", vcov = TRUE))
out$un_hetvar <- pr(suppressWarnings(predict(resun, newmods = diag(2), tau2.levels = c("AL", "PD"), hetvar = c(0.01, 0.02))))

dat4 <- dat.bcg
resgl <- rma.glmm(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat4)
out$glmm <- pr(predict(resgl, prob = "<0", hetvar = 0.1, vcov = TRUE, addx = TRUE))
write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), "src/tests/reference/phase7/predict_args.json")
