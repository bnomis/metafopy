# Reference values for influence(), rstandard(), rstudent(), cooks.distance(), dfbetas(),
# leave1out() and cumul() on rma.uni fits (from the examples on their help pages and a
# few variations: moderators, equal-effects, Knapp-Hartung, user weights, transf, NAs),
# plus the printed output.
#
# Usage (from the repository root):  Rscript tests/r/generate_diagnostics.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
strip <- function(x) {
  x <- unclass(x)
  x$data <- NULL
  x$digits <- NULL
  x$ids <- NULL
  x$level <- NULL
  x
}
infl <- function(res, ...) {
  i <- influence(res, ...)
  out <- c(lapply(unclass(i$inf)[c("rstudent", "dffits", "cook.d", "cov.r", "tau2.del", "QE.del", "hat", "weight")], num),
           list(inf = as.character(i$inf$inf), slab = i$inf$slab,
                dfbs = lapply(unclass(i$dfbs)[seq_len(res$p)], num),
                is.infl = num(i$is.infl), m = i$m, print = printed(i), print_infonly = printed(i, infonly = TRUE)))
  out
}
resid_list <- function(x) list(resid = num(x$resid), se = num(x$se), z = num(x$z), slab = x$slab)

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
dat$n <- with(dat.bcg, tpos + tneg + cpos + cneg)
dat$label <- paste(dat$author, dat$year)
out <- list()

### influence --------------------------------------------------------------------
res <- rma(yi, vi, data = dat)
out$infl_re       <- infl(res)
out$infl_mods     <- infl(rma(yi, vi, mods = ~ ablat + year, data = dat))
out$infl_mods_btt <- infl(rma(yi, vi, mods = ~ ablat + year, data = dat), btt = 2:3)
out$infl_ee       <- infl(rma(yi, vi, data = dat, method = "EE"))
out$infl_dl       <- infl(rma(yi, vi, data = dat, method = "DL"))
out$infl_ml_mods  <- infl(rma(yi, vi, mods = ~ ablat, data = dat, method = "ML"))
out$infl_knha     <- infl(rma(yi, vi, mods = ~ ablat, data = dat, test = "knha"))
out$infl_weights  <- infl(rma(yi, vi, mods = ~ ablat, weights = n, data = dat))
out$infl_unweighted <- infl(rma(yi, vi, mods = ~ ablat, weighted = FALSE, data = dat))
out$infl_slab     <- infl(rma(yi, vi, data = dat, slab = label))
out$infl_fixtau2  <- infl(rma(yi, vi, data = dat, tau2 = 0.2))

### missing values: yi for study 3 and ablat for study 7 set to NA
dna <- dat
dna$yi[3] <- NA
dna$ablat[7] <- NA
res_na <- suppressWarnings(rma(yi, vi, mods = ~ ablat, data = dna))
out$infl_na_omit <- infl(res_na)
options(na.action = "na.exclude")
out$infl_na_exclude <- infl(res_na)
options(na.action = "na.omit")

### residuals ---------------------------------------------------------------------
out$rstandard_marginal    <- resid_list(rstandard(res))
out$rstandard_conditional <- resid_list(rstandard(res, type = "conditional"))
mres <- rma(yi, vi, mods = ~ ablat + year, data = dat)
out$rstandard_mods        <- resid_list(rstandard(mres))
out$rstandard_mods_cond   <- resid_list(rstandard(mres, type = "conditional"))
out$rstandard_ee          <- resid_list(rstandard(rma(yi, vi, data = dat, method = "EE")))
out$rstandard_weights     <- resid_list(rstandard(rma(yi, vi, mods = ~ ablat, weights = n, data = dat)))
out$rstandard_unweighted  <- resid_list(rstandard(rma(yi, vi, mods = ~ ablat, weighted = FALSE, data = dat)))
out$rstandard_print       <- printed(rstandard(res))
out$rstudent_re           <- resid_list(rstudent(res))
out$rstudent_mods         <- resid_list(rstudent(mres))
out$rstudent_print        <- printed(rstudent(res))
out$cooks_re              <- list(cook = num(cooks.distance(res)), names = names(cooks.distance(res)))
out$cooks_mods            <- list(cook = num(cooks.distance(mres)))
out$dfbetas_mods          <- lapply(unclass(dfbetas(mres))[seq_len(3)], num)
out$hat_matrix            <- list(diag = num(hatvalues(mres)), full = lapply(seq_len(13), function(i) num(hatvalues(mres, type = "matrix")[i, ])))
options(na.action = "na.exclude")
out$rstandard_na_exclude  <- resid_list(rstandard(res_na))
out$rstudent_na_exclude   <- resid_list(rstudent(res_na))
options(na.action = "na.omit")
out$rstandard_na_omit     <- resid_list(rstandard(res_na))

### leave1out ----------------------------------------------------------------------
l1o <- function(x, ...) {
  r <- do.call(leave1out, list(x, ...))
  c(lapply(strip(r), function(v) if (is.character(v)) v else num(v)), list(print = printed(r)))
}
out$l1o_re      <- l1o(res)
out$l1o_exp     <- l1o(res, transf = exp)
out$l1o_ee      <- l1o(rma(yi, vi, data = dat, method = "EE"))
out$l1o_knha    <- l1o(rma(yi, vi, data = dat, test = "knha"))
out$l1o_dl      <- l1o(rma(yi, vi, data = dat, method = "DL"))
out$l1o_ml      <- l1o(rma(yi, vi, data = dat, method = "ML"))
out$l1o_exp_int <- l1o(res, transf = transf.exp.int)
out$l1o_cluster <- l1o(res, cluster = dat$alloc)
out$l1o_weights <- l1o(rma(yi, vi, weights = n, data = dat))
out$l1o_level90 <- l1o(rma(yi, vi, data = dat, level = 90))
options(na.action = "na.exclude")
out$l1o_na_exclude <- l1o(suppressWarnings(rma(yi, vi, data = dna)))
options(na.action = "na.omit")
out$l1o_na_omit <- l1o(suppressWarnings(rma(yi, vi, data = dna)))

### cumul ---------------------------------------------------------------------------
cum <- function(x, ...) {
  r <- do.call(cumul, list(x, ...))
  c(lapply(strip(r), function(v) if (is.character(v)) v else num(v)), list(print = printed(r)))
}
out$cumul_re       <- cum(res)
out$cumul_year     <- cum(res, order = dat$year)
out$cumul_year_dec <- cum(res, order = dat$year, decreasing = TRUE)
out$cumul_collapse <- cum(res, order = dat$alloc, collapse = TRUE)
out$cumul_collapse_year <- cum(res, order = dat$year, collapse = TRUE)
out$cumul_exp      <- cum(res, order = dat$year, transf = exp)
out$cumul_ee       <- cum(rma(yi, vi, data = dat, method = "EE"), order = dat$year)
out$cumul_knha     <- cum(rma(yi, vi, data = dat, test = "knha"), order = dat$year)
out$cumul_dl       <- cum(rma(yi, vi, data = dat, method = "DL"), order = dat$ablat)

### permutest -----------------------------------------------------------------------
perm <- function(x, ...) {
  r <- permutest(x, progbar = FALSE, ...)
  list(pval = num(r$pval), QMp = r$QMp, QM = r$QM, zval = num(r$zval), beta = num(r$beta), ci.lb = num(r$ci.lb), ci.ub = num(r$ci.ub),
       exact.iter = r$exact.iter, permci = r$permci, alternative = r$alternative, QM.perm = num(r$QM.perm),
       zval.perm = lapply(r$zval.perm, num), beta.perm = lapply(r$beta.perm, num), print = printed(r))
}
d8 <- dat[1:8, ]
d9 <- dat[1:9, ]
r8  <- rma(yi, vi, data = d8)
r9m <- rma(yi, vi, mods = ~ alloc, data = d9)
r13 <- rma(yi, vi, data = dat)
r13m <- rma(yi, vi, mods = ~ ablat, data = dat)
out$perm_exact_int     <- perm(r8, exact = TRUE)
out$perm_exact_int_knha <- perm(rma(yi, vi, data = d8, test = "knha"), exact = TRUE)
out$perm_exact_int_ee  <- perm(rma(yi, vi, data = d8, method = "EE"), exact = TRUE)
out$perm_exact_int_coef <- perm(r8, exact = TRUE, control = list(stat = "coef"))
out$perm_exact_int_px2 <- perm(r8, exact = TRUE, control = list(p2defn = "px2"))
out$perm_exact_int_less <- perm(r8, exact = TRUE, control = list(alternative = "less"))
out$perm_exact_int_greater <- perm(r8, exact = TRUE, control = list(alternative = "greater"))
out$perm_exact_mods    <- perm(r9m, exact = TRUE)
out$perm_exact_mods_knha <- perm(rma(yi, vi, mods = ~ alloc, data = d9, test = "knha"), exact = TRUE)
out$perm_exact_iter_i  <- list(int = permutest(r13, exact = "i"), mods = permutest(r13m, exact = "i"), mods9 = permutest(r9m, exact = "i"))
out$perm_exact_auto    <- perm(r8, iter = 1000)
out$perm_exact_permci  <- perm(r8, exact = TRUE, permci = TRUE)
out$perm_exact_permci_mods <- perm(r9m, exact = TRUE, permci = 2)
out$perm_approx_int    <- perm(r13, iter = 200, seed = 1234)
out$perm_approx_mods   <- perm(r13m, iter = 200, seed = 99)
out$perm_approx_mods_multi <- perm(rma(yi, vi, mods = ~ ablat + year, data = dat), iter = 150, seed = 7)
out$perm_approx_btt    <- perm(rma(yi, vi, mods = ~ ablat + year, data = dat), iter = 150, seed = 7, btt = 2)
out$perm_approx_knha   <- perm(rma(yi, vi, mods = ~ ablat, data = dat, test = "knha"), iter = 100, seed = 5)
out$perm_approx_dl     <- perm(rma(yi, vi, data = dat, method = "DL"), iter = 100, seed = 3)
set.seed(42)
out$perm_approx_global <- perm(r13m, iter = 100)
out$perm_approx_global2 <- perm(r13, iter = 100)
set.seed(11)
out$perm_approx_permci <- perm(r13m, iter = 100, permci = TRUE)
out$perm_tau2_fixed    <- perm(rma(yi, vi, data = d8, tau2 = 0.1), exact = TRUE)

### profile -------------------------------------------------------------------------
prof <- function(x, ...) {
  r <- suppressWarnings(profile(x, progbar = FALSE, plot = FALSE, ...))
  nm <- if (!is.null(r[["tau"]])) "tau" else "tau2"
  list(grid = num(r[[nm]]), name = nm, ll = num(r$ll), beta = lapply(r$beta, num), ci.lb = lapply(r$ci.lb, num), ci.ub = lapply(r$ci.ub, num),
       xlim = num(r$xlim), ylim = num(r$ylim), vc = r$vc, maxll = r$maxll, I2 = num(r$I2), H2 = num(r$H2), method = r$method)
}
out$prof_reml     <- prof(r13)
out$prof_ml       <- prof(rma(yi, vi, data = dat, method = "ML"))
out$prof_dl       <- prof(rma(yi, vi, data = dat, method = "DL"))
out$prof_mods     <- prof(r13m)
out$prof_steps    <- prof(r13, steps = 10)
out$prof_xlim     <- prof(r13, xlim = c(0.05, 1.2), steps = 12)
out$prof_ylim     <- prof(r13, ylim = c(-30, -10))
out$prof_seq      <- prof(r13, steps = c(0.01, 0.1, 0.2, 0.5))
out$prof_sqrt     <- prof(r13, sqrt = TRUE)
out$prof_exp      <- prof(r13, exp = TRUE)
out$prof_knha     <- prof(rma(yi, vi, mods = ~ ablat, data = dat, test = "knha"))

### gosh ----------------------------------------------------------------------------
gsh <- function(x, ...) {
  r <- suppressMessages(gosh(x, progbar = FALSE, ...))
  list(dbinom = sprintf("%.17g", dbinom(x$p:x$k, x$k, 0.5)), res = lapply(r$res, num), incl = lapply(seq_len(nrow(r$incl)), function(i) as.integer(r$incl[i, ])), fit = num(r$fit), k = r$k,
       print = printed(r))
}
out$gosh_int_reml <- gsh(r8)
out$gosh_int_ee   <- gsh(rma(yi, vi, data = d8, method = "EE"))
out$gosh_int_dl   <- gsh(rma(yi, vi, data = d8, method = "DL"))
out$gosh_int_ml   <- gsh(rma(yi, vi, data = d8, method = "ML"))
out$gosh_mods     <- gsh(rma(yi, vi, mods = ~ ablat, data = d9))
out$gosh_mods_knha <- gsh(rma(yi, vi, mods = ~ ablat, data = d9, test = "knha"))
out$gosh_weights  <- gsh(rma(yi, vi, weights = n, data = d8))
out$gosh_random_int <- gsh(r13, subsets = 300, seed = 123)
out$gosh_random_mods <- gsh(r13m, subsets = 200, seed = 7)
set.seed(99)
out$gosh_random_global <- gsh(r13, subsets = 100)

out$meta <- list(metafor = as.character(packageVersion("metafor")), R = R.version.string)
write_json(out, "tests/reference/phase2/diagnostics.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote tests/reference/phase2/diagnostics.json\n")
