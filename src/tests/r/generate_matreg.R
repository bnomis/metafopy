# Reference values for matreg() and its methods (coef, vcov, sigma, logLik, AIC, BIC, confint, predict, print and
# summary), following the examples on the matreg help page: regression on the covariance and correlation matrices of
# mtcars (with n, and with V from rcalc), on the meta-analytic correlation matrix of dat.craft2003 (also with r-to-z
# transformed correlations) and on the between-study covariance matrix of the dat.colditz1994 bivariate model.
# The input matrices are stored in the fixture, so the Python tests need neither mtcars nor rma.mv.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_matreg.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

options(width = 200)
num <- function(x) unname(as.vector(x))
mat <- function(x) unname(as.matrix(x))
txt <- function(x) paste(capture.output(print(x)), collapse = "\n")

fit <- function(res) {
  out <- list(names = rownames(res$tab), beta = num(res$tab$beta), se = num(res$tab$se), pval = num(res$tab$pval),
              ci_lb = num(res$tab$ci.lb), ci_ub = num(res$tab$ci.ub), vb = mat(res$vb), R2 = res$R2, test = res$test,
              print = txt(res), summary = txt(summary(res)), confint90 = num(as.matrix(confint(res, level = 90)$tab)),
              confint_print = txt(confint(res)), coef = num(coef(res)))
  if (res$test == "t") {
    out$tval <- num(res$tab$tval)
    out$df <- res$df.residual
    out$R2adj <- res$R2adj
    out$F <- num(res$F)
    out$Fp <- res$Fp
    out$fit_stats <- mat(res$fit.stats)
    out$sigma <- c(sigma(res), sigma(res, REML = FALSE))
    out$logLik <- c(as.numeric(logLik(res)), as.numeric(logLik(res, REML = TRUE)))
    out$AIC <- c(AIC(res), AIC(res, correct = TRUE), AIC(res, REML = TRUE), AIC(res, REML = TRUE, correct = TRUE))
    out$BIC <- c(BIC(res), BIC(res, REML = TRUE))
    out$parms <- res$parms
  } else {
    out$zval <- num(res$tab$zval)
    out$QM <- res$QM
    out$QMp <- res$QMp
  }
  out
}

pred <- function(p) {
  if (is.null(p$pred) || !is.numeric(p$pred)) {
    v <- p$vcov
    p <- p$pred
    out <- list(vcov = mat(v))
  } else {
    out <- list()
  }
  c(out, list(pred = num(p$pred), se = num(p$se), ci_lb = num(p$ci.lb), ci_ub = num(p$ci.ub), slab = as.character(p$slab),
              print = txt(p), X = if (is.null(p$X)) NULL else mat(p$X)))
}

out <- list()

### mtcars: covariance matrix (lm() equivalent)
S <- cov(mtcars)
mc <- list(S = mat(S), R = mat(cor(mtcars)), names = colnames(mtcars), means = num(colMeans(mtcars)), n = nrow(mtcars))
mc$V <- mat(rcalc(cor(mtcars), ni = nrow(mtcars))$V)
out$mtcars <- mc

res <- matreg(mpg ~ hp + wt + am, R = S, cov = TRUE, means = colMeans(mtcars), n = nrow(mtcars))
out$cov_n <- fit(res)
out$cov_n$pred_one <- pred(predict(res, newmods = c(100, 3, 1)))
out$cov_n$pred_two <- pred(predict(res, newmods = rbind(c(100, 3, 1), c(150, 2.5, 0)), vcov = TRUE, addx = TRUE))
nm <- rbind(a = c(wt = 3, am = 1, hp = 100), b = c(wt = 2.5, am = 0, hp = 150))
out$cov_n$pred_named <- pred(predict(res, newmods = nm, level = 90, adjust = TRUE))
out$cov_n$pred_noint <- pred(predict(res, newmods = c(100, 3, 1), intercept = FALSE))
out$cov_n$pred_full <- pred(predict(res, newmods = c(1, 100, 3, 1)))
out$cov_n$pred_transf <- pred(predict(res, newmods = c(100, 3, 1), transf = log))
res2 <- matreg(mpg ~ hp + wt, R = S, cov = TRUE, means = colMeans(mtcars), n = nrow(mtcars))
out$cov_n$AIC_multi <- num(as.matrix(AIC(res, res2)))
out$cov_n$BIC_multi <- num(as.matrix(BIC(res, res2)))
out$cov_n$AICc_multi <- num(as.matrix(AIC(res, res2, correct = TRUE, REML = TRUE)))

out$cov_n_nomeans <- fit(matreg(y = 1, x = c(4, 6, 9), R = S, cov = TRUE, n = nrow(mtcars)))

### mtcars: correlation matrix, with n and with V
R <- cor(mtcars)
res <- matreg(y = "mpg", x = c("hp", "wt", "am"), R = R, n = nrow(mtcars))
out$cor_n <- fit(res)
out$cor_n$pred <- pred(predict(res, newmods = rbind(c(1, 0, 0), c(0.5, -0.5, 1))))
out$cor_n$pred_noint <- pred(predict(res, newmods = c(1, 1, 1), intercept = FALSE))
res <- matreg(y = "mpg", x = c("hp", "wt", "am"), R = R, V = rcalc(R, ni = nrow(mtcars))$V)
out$cor_V <- fit(res)
out$cor_V$pred <- pred(predict(res, newmods = c(1, -1, 0.5), vcov = TRUE))
out$cor_V$pred_two <- pred(predict(res, newmods = rbind(c(1, -1, 0.5), c(0, 1, 0)), vcov = TRUE))
out$cov_V_nomeans <- fit(matreg(y = 1, x = c(4, 6), R = S, cov = TRUE, V = diag(seq_len(66) / 1000)))
out$cov_V_nomeans$V <- num(seq_len(66) / 1000)
out$cor_n_all <- fit(matreg(y = 1, R = R[1:4, 1:4], n = 32))

### dat.craft2003: meta-analytic correlation matrix
dat <- dat.craft2003
tmp <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dat)
dat <- tmp$dat
dat$var1.var2 <- factor(dat$var1.var2, levels = c("acog.perf", "asom.perf", "conf.perf", "acog.asom", "acog.conf", "asom.conf"))
res <- rma.mv(yi, tmp$V, mods = ~ 0 + var1.var2, random = ~ var1.var2 | study, struct = "UN", data = dat)
R <- vec2mat(coef(res))
rownames(R) <- colnames(R) <- c("perf", "acog", "asom", "conf")
craft <- list(R = mat(R), names = rownames(R), V = mat(vcov(res)))
# the rcalc() -> rma.mv() pipeline itself (rcalc leaves NA rows/columns for missing correlations; .anyNAv() decides
# which rows rma.mv() drops)
mvfit <- function(res, tmp) list(rcalc_V = mat(tmp$V), rcalc_yi = num(tmp$dat$yi), k = res$k, k_all = res$k.all,
                                 not_na = res$not.na, beta = num(coef(res)), vb = mat(vcov(res)), tau2 = num(res$tau2),
                                 rho = num(res$rho), ll = as.numeric(logLik(res)))
craft$mv <- mvfit(res, tmp)
mr <- matreg(1, 2:4, R = R, V = vcov(res))
craft$fit <- fit(mr)
craft$fit$pred <- pred(predict(mr, newmods = c(0.2, 0.1, 0.3)))
craft$fit_n <- fit(matreg("perf", c("acog", "asom", "conf"), R = R, n = 500))

dat <- dat.craft2003
tmp <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dat, rtoz = TRUE)
dat <- tmp$dat
dat$var1.var2 <- factor(dat$var1.var2, levels = c("acog.perf", "asom.perf", "conf.perf", "acog.asom", "acog.conf", "asom.conf"))
res <- rma.mv(yi, tmp$V, mods = ~ 0 + var1.var2, random = ~ var1.var2 | study, struct = "UN", data = dat)
R <- vec2mat(coef(res))
rownames(R) <- colnames(R) <- c("perf", "acog", "asom", "conf")
craft$Z <- mat(R)
craft$mv_z <- mvfit(res, tmp)
craft$VZ <- mat(vcov(res))
craft$fit_ztor <- fit(matreg(1, 2:4, R = R, V = vcov(res), ztor = TRUE))
craft$fit_ztor_n <- fit(matreg(1, 2:4, R = R, n = 500, ztor = TRUE))
out$craft <- craft

### dat.colditz1994: bivariate model, regression of log(odds)_exp on log(odds)_con
dat.long <- to.long(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.colditz1994, append = FALSE)
dat.long <- escalc(measure = "PLO", xi = out1, mi = out2, data = dat.long)
dat.long$group <- factor(dat.long$group, levels = c(2, 1), labels = c("con", "exp"))
res <- rma.mv(yi, vi, mods = ~ 0 + group, random = ~ group | study, struct = "UN", data = dat.long, method = "ML",
              cvvc = "varcov", control = list(nearpd = TRUE))
colditz <- list(G = mat(res$G), means = num(coef(res)), n = res$g.levels.comb.k, vvc = mat(res$vvc))
mr <- matreg(y = 2, x = 1, R = res$G, cov = TRUE, means = coef(res), n = res$g.levels.comb.k)
colditz$fit_n <- fit(mr)
colditz$fit_n$pred <- pred(predict(mr, newmods = c(-4, -3, -2)))
colditz$fit_n$pred_matrix <- pred(predict(mr, newmods = cbind(c(-4, -3)), intercept = FALSE))
mr <- matreg(y = 2, x = 1, R = res$G, cov = TRUE, means = coef(res), V = res$vvc)
colditz$fit_V <- fit(mr)
colditz$fit_V$pred <- pred(predict(mr, newmods = c(-4, -3, -2)))
out$colditz <- colditz

### nearpd: R[x,x] is not positive definite
Rn <- matrix(c(1, .5, .4, .3,
               .5, 1, .8, .8,
               .4, .8, 1, -.3,
               .3, .8, -.3, 1), nrow = 4)
out$nearpd <- list(R = mat(Rn), fit = suppressMessages(fit(matreg(1, 2:4, R = Rn, n = 100, nearpd = TRUE))),
                   error = tryCatch(matreg(1, 2:4, R = Rn, n = 100), error = function(e) conditionMessage(e)))

write(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null"), "src/tests/reference/phase7/matreg.json")
