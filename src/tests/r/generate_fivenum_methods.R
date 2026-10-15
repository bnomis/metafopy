# Reference values for the 'qe', 'bc', 'mln' and 'blue' methods of conv.fivenum() (R packages estmeansd 1.0.1 and metaBLUE 1.0.0),
# and for the underlying estimators (Luo.mean, Wan.std, BLUE_s, qe.mean.sd, bc.mean.sd, mln.mean.sd).
#
# Usage (from the repository root):  R_LIBS=/tmp/Rlib LC_ALL=C.UTF-8 Rscript src/tests/r/generate_fivenum_methods.R

suppressPackageStartupMessages({
  library(metafor)
  library(estmeansd)
  library(metaBLUE)
  library(jsonlite)
})

out_dir <- "src/tests/reference/phase10"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

num <- function(x) unname(as.vector(x))
df <- function(d) {
  d <- as.data.frame(d)
  cols <- lapply(d, function(col) if (is.factor(col)) as.character(col) else unname(as.vector(col)))
  names(cols) <- NULL
  list(names = names(d), columns = cols)
}
mat <- function(m) list(values = unname(as.matrix(m)), rownames = rownames(m), colnames = colnames(m))
save <- function(x, name) write(toJSON(x, auto_unbox = TRUE, digits = NA, na = "null", null = "null", matrix = "rowmajor"), file.path(out_dir, paste0(name, ".json")))

out <- list()

############################################################################
# metaBLUE and estmeansd on single studies

X1 <- c(1, 4, 10); X2 <- c(3, 5, 9); X3 <- c(1.5, 3, 5.5, 8, 12)
single <- list()
for (n in c(5, 6, 7, 13, 20, 45, 100, 331)) {
  for (type in c("S1", "S2", "S3")) {
    X <- list(S1 = X1, S2 = X2, S3 = X3)[[type]]
    b <- BLUE_s(X, n, type)
    single[[paste0(type, "_", n)]] <- list(type = type, n = n, X = X,
                                           luo = Luo.mean(X, n, type)$muhat, wan = Wan.std(X, n, type)$sigmahat,
                                           alpha = num(b$alpha), B = mat(b$B), muhat = num(b$muhat), sigmahat = num(b$sigmahat),
                                           var_mu = num(b$Var_mu), var_sigma = num(b$Var_sigma))
  }
}
out$single <- single

# the estmeansd estimators on a few skewed data sets (quantiles of rlnorm and rgamma samples)
mk <- function(x) c(min = min(x), q1 = unname(quantile(x, 0.25)), med = median(x), q3 = unname(quantile(x, 0.75)), max = max(x))
set.seed(1)
samples <- list(lnorm100 = rlnorm(100, 2.5, 1), gamma60 = rgamma(60, 2, 0.3), norm40 = rnorm(40, 50, 8), lnorm25 = rlnorm(25, 1, 0.4), weib200 = rweibull(200, 1.5, 10))
est <- list()
for (nm in names(samples)) {
  x <- samples[[nm]]; n <- length(x); q <- mk(x)
  for (sc in c("S1", "S2", "S3")) {
    args <- switch(sc,
                   S1 = list(min.val = q[["min"]], med.val = q[["med"]], max.val = q[["max"]], n = n),
                   S2 = list(q1.val = q[["q1"]], med.val = q[["med"]], q3.val = q[["q3"]], n = n),
                   S3 = list(min.val = q[["min"]], q1.val = q[["q1"]], med.val = q[["med"]], q3.val = q[["q3"]], max.val = q[["max"]], n = n))
    qe <- suppressWarnings(do.call(qe.mean.sd, args))
    qev <- list()
    for (d in names(qe$values)) qev[[d]] <- list(value = num(qe$values[[d]]), mean = num(qe[[paste0(d, ".est.mean")]]), sd = num(qe[[paste0(d, ".est.sd")]]))
    set.seed(42); bc <- suppressWarnings(do.call(bc.mean.sd, args))
    set.seed(42); mln <- suppressWarnings(do.call(mln.mean.sd, args))
    est[[paste0(nm, "_", sc)]] <- list(args = args, qe = list(mean = qe$est.mean, sd = qe$est.sd, dist = qe$selected.dist, fits = qev),
                                       bc = list(mean = bc$est.mean, sd = bc$est.sd, location = bc$location, scale = bc$scale, shape = bc$shape),
                                       mln = list(mean = mln$est.mean, sd = mln$est.sd, location = mln$location, scale = mln$scale, shape = mln$shape))
  }
}
out$estimators <- est

############################################################################
# conv.fivenum()

fivenum_case <- function(dat, ...) {
  res <- conv.fivenum(min = min, q1 = q1, median = median, q3 = q3, max = max, n = n, data = dat, ...)
  a <- attributes(res$mean)
  list(data = df(res), tval = num(a$tval), crit = num(a$crit), sig = num(a$sig), dist = num(a$dist), est = num(attr(res$sd, "est")))
}

dat <- data.frame(min = c(2, NA, 2, 1, NA, 1, 1.2, NA, NA, 1),
                  q1 = c(NA, 4, 4, NA, 5, 3, 3.1, 3, NA, 4),
                  median = c(6, 6, 6, 4, 9, 5.5, 5.5, 5, 5, 9),
                  q3 = c(NA, 10, 10, NA, 14, 11, 12, 9, NA, 15),
                  max = c(14, NA, 14, 70, NA, 40, 80, NA, 22, 55),
                  n = c(20, 20, 20, 80, 60, 45, 100, 6, 4, 150))
out$input <- df(dat)
for (m in c("qe", "bc", "mln", "blue")) {
  for (tst in c(FALSE, TRUE)) {
    key <- paste0(m, if (tst) "_test" else "")
    out[[key]] <- fivenum_case(dat, method = m, test = tst, seed = 1234)
  }
}
out$bc_seed2 <- fivenum_case(dat, method = "bc", seed = 7)
out$mln_seed2 <- fivenum_case(dat, method = "mln", seed = 7)
out$noappend <- fivenum_case(dat, method = "qe", append = FALSE)

# negative values: qe tests for skewness (and the test flag stays on after a skipped study), blue works, bc and mln stop
neg <- data.frame(min = c(-5, 1, NA, -3, 2, NA), q1 = c(-2, 3, 3, NA, 4, -1), median = c(0, 5, 5.5, 1, 6, 0.5), q3 = c(3, 9, 10, NA, 20, 4),
                  max = c(9, 20, NA, 12, 40, NA), n = c(30, 30, 30, 50, 50, 25))
out$input_neg <- df(neg)
out$qe_neg <- fivenum_case(neg, method = "qe")
out$qe_neg_notest <- fivenum_case(neg, method = "qe", test = FALSE)
out$blue_neg <- fivenum_case(neg, method = "blue", test = FALSE)
msg <- function(expr) tryCatch({ expr; NA_character_ }, error = function(e) conditionMessage(e))
out$errors <- list(
  bc_neg = msg(fivenum_case(neg, method = "bc")),
  mixed = msg(fivenum_case(dat, method = c("qe", "bc"))),
  mixed_blue = msg(fivenum_case(dat, method = c("blue", "wan2014")))
)
save(out, "conv_fivenum_methods")
