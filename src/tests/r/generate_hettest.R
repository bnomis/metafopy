# Reference values for hettest() (new in metafor 5.2-1): statistics, bootstrap p-values (R's generator via set.seed), tau2_i estimates and printed output.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_hettest.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE, width = 200)
printed <- function(x, ...) paste(capture.output(print(x, ...)), collapse = "\n")
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
hack <- dat.hackshaw1998

ht <- function(..., seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  r <- hettest(..., progbar = FALSE)
  list(statistic = num(r$statistic), df = num(r$df), pval = num(r$pval), method = r$method, test = r$test, boot = r$boot, iter = r$iter,
       tau2i = num(r$tau2i), se.tau2i = num(r$se.tau2i),
       nboot = if (is.null(r$statistic.boot)) NULL else as.vector(colSums(!is.na(r$statistic.boot))),
       statistic.boot = if (is.null(r$statistic.boot)) NULL else lapply(seq_len(ncol(r$statistic.boot)), function(j) num(r$statistic.boot[, j])),
       print = printed(r))
}

all_tests <- c("score", "lrt", "wald", "ksn", "ksx2", "adn", "adx2")

out <- list()
out$vec_reml <- ht(dat$yi, vi = dat$vi, test = all_tests, boot = 200, seed = 1234)
out$vec_ml <- ht(dat$yi, vi = dat$vi, method = "ML", test = all_tests, boot = 200, seed = 99)
out$vec_noboot <- ht(dat$yi, vi = dat$vi, test = c("score", "lrt", "wald"), boot = FALSE, seed = 5)
out$vec_noboot_ml <- ht(dat$yi, vi = dat$vi, method = "ML", test = c("score", "lrt", "wald"), boot = FALSE, seed = 5)
out$vec_sei <- ht(dat$yi, sei = sqrt(dat$vi), boot = 100, seed = 7)
out$vec_mom <- ht(dat$yi, vi = dat$vi, test = c("lrt", "wald"), mom = TRUE, boot = 100, seed = 11)
out$vec_hack <- ht(hack$yi, vi = hack$vi, test = all_tests, boot = 150, seed = 3)
out$vec_hack_ml <- ht(hack$yi, vi = hack$vi, method = "ML", test = c("score", "lrt"), boot = 100, seed = 4)
out$vec_boot_mixed <- ht(dat$yi, vi = dat$vi, test = c("score", "lrt", "ksn"), boot = c(TRUE, FALSE, FALSE), seed = 8)
out$vec_control <- ht(dat$yi, vi = dat$vi, test = c("lrt", "wald"), boot = FALSE, control = list(tau2i.init = 0.5))
out$vec_default_seeded <- ht(dat$yi, vi = dat$vi, boot = 300, seed = 2024)

res_reml <- rma(yi, vi, data = dat)
res_ml <- rma(yi, vi, data = dat, method = "ML")
res_hack <- rma(yi, vi, data = hack)
out$mod_reml <- ht(res_reml, test = all_tests, boot = 200, seed = 21)
out$mod_ml <- ht(res_ml, test = all_tests, boot = 200, seed = 22)
out$mod_hack <- ht(res_hack, test = c("score", "lrt", "wald"), boot = 100, seed = 23)
out$mod_noboot <- ht(res_reml, test = c("score", "lrt", "wald"), boot = FALSE)
out$mod_dl <- suppressWarnings(ht(rma(yi, vi, data = dat, method = "DL"), test = c("score", "lrt"), boot = 100, seed = 24))
out$mod_dl_ml <- suppressWarnings(ht(rma(yi, vi, data = dat, method = "DL"), method = "ML", test = c("score", "lrt"), boot = 100, seed = 25))
out$mod_knha <- ht(rma(yi, vi, data = dat, test = "knha"), test = "score", boot = 100, seed = 26)
out$mod_fe <- suppressWarnings(ht(rma(yi, vi, data = dat, method = "FE"), test = c("score", "lrt"), boot = 100, seed = 27))
out$mod_mh <- suppressWarnings(ht(rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, measure = "RR"), test = "score", boot = 60, seed = 28))

yna <- dat$yi; yna[4] <- NA
out$vec_na <- suppressWarnings(ht(yna, vi = dat$vi, boot = 100, seed = 31))

# the global stream: set.seed once, then two calls without reseeding
set.seed(42)
a <- hettest(dat$yi, vi = dat$vi, boot = 80, progbar = FALSE)
b <- hettest(dat$yi, vi = dat$vi, boot = 80, test = "lrt", progbar = FALSE)
out$stream <- list(a = num(a$pval), b = num(b$pval), a_stat = num(a$statistic), b_stat = num(b$statistic))

errs <- list(
  mods = quote(hettest(rma(yi, vi, mods = ~ablat, data = dat), progbar = FALSE)),
  test = quote(hettest(dat$yi, vi = dat$vi, test = "foo", progbar = FALSE)),
  method = quote(hettest(dat$yi, vi = dat$vi, method = "DL", progbar = FALSE)),
  novi = quote(hettest(dat$yi, progbar = FALSE)),
  k1 = quote(hettest(1, vi = 0.1, progbar = FALSE)),
  mv = quote(hettest(rma.mv(yi, vi, random = ~1 | trial, data = dat), progbar = FALSE)),
  glmm = quote(hettest(rma.glmm(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, measure = "OR"), progbar = FALSE)),
  ls = quote(hettest(rma(yi, vi, scale = ~1, data = dat), progbar = FALSE)),
  sel = quote(hettest(selmodel(res_reml, type = "beta"), progbar = FALSE)),
  tau2i_init = quote(hettest(dat$yi, vi = dat$vi, control = list(tau2i.init = c(1, 2)), progbar = FALSE))
)
out$errors <- lapply(errs, function(e) tryCatch({ suppressWarnings(eval(e)); "no error" }, error = function(err) conditionMessage(err)))

write_json(out, "src/tests/reference/phase11/hettest.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null")
