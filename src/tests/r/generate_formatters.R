# Reference values for fmtx(), fmtp(), fmtp2(), fmtt() and for the printout of rma() under setmfopt(digits=, space=).
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_formatters.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
s <- function(x) as.character(unclass(x))
pv <- c(0.0002, 0.00008, NA, 0.5, 1.2, -0.1, 0.04999, 0.00010)
out <- list()
out$fmtp <- list(
  list(args = list(digits = 4), out = s(fmtp(pv, digits = 4))),
  list(args = list(digits = 4, equal = TRUE), out = s(fmtp(pv, digits = 4, equal = TRUE))),
  list(args = list(digits = 4, equal = TRUE, sep = TRUE), out = s(fmtp(pv, digits = 4, equal = TRUE, sep = TRUE))),
  list(args = list(digits = 4, equal = TRUE, sep = TRUE, add0 = TRUE), out = s(fmtp(pv, digits = 4, equal = TRUE, sep = TRUE, add0 = TRUE))),
  list(args = list(digits = 2, pname = "p", sep = TRUE), out = s(fmtp(pv, digits = 2, pname = "p", sep = TRUE))),
  list(args = list(digits = 0), out = s(fmtp(pv, digits = 0)))
)
pv2 <- c(0.0005, 0.001, 0.002, 0.0423, 0.0543, 0.0578, 0.0623, 0.5329, NA, 1)
out$fmtp2 <- list(
  list(args = list(), out = s(fmtp2(pv2))),
  list(args = list(add0 = TRUE, sep = FALSE), out = s(fmtp2(pv2, add0 = TRUE, sep = FALSE))),
  list(args = list(cutoff = c(0.05, 0.0001), pname = "P"), out = s(fmtp2(pv2, cutoff = c(0.05, 0.0001), pname = "P")))
)
xs <- c(1.0002, 2.00008, 3.00004, -1, 1, -0.00001, NA, Inf, -Inf, 0.5, 12345.678, 2.5)
out$fmtx <- list(
  list(args = list(digits = 4), out = s(fmtx(xs, digits = 4))),
  list(args = list(digits = 4, flag = " "), out = s(fmtx(xs, digits = 4, flag = " "))),
  list(args = list(digits = 2, flag = "-", addwidth = 6), out = s(fmtx(xs, digits = 2, flag = "-", addwidth = 6))),
  list(args = list(digits = 3, add0 = FALSE, postfix = "%"), out = s(fmtx(xs, digits = 3, add0 = FALSE, postfix = "%"))),
  list(args = list(digits = 1, flag = "+"), out = s(fmtx(xs, digits = 1, flag = "+"))),
  list(args = list(digits = 0), out = s(fmtx(xs, digits = 0)))
)
out$fmtx_drop0 <- s(fmtx(c(1.5, 2, 2.10, 0.25), digits = 2L, drop0ifint = TRUE))
out$fmtx_thresh <- c(s(fmtx(1e-9, digits = 4, thresh = 1e-8)), s(fmtx(0.5, digits = 4, thresh = 1e-8)))
out$fmtt <- c(
  s(fmtt(2.45, "z", pval = 0.01429, digits = 2)),
  s(fmtt(3.45, "z", pval = 0.00056, digits = 2)),
  s(fmtt(2.45, "t", df = 23, pval = 0.02232, digits = 2)),
  s(fmtt(3.45, "t", df = 23, pval = 0.00218, digits = 2)),
  s(fmtt(3.45, "t", df = 23, pval = 0.00218, digits = 2, format = 2)),
  s(fmtt(46.23, "Q", df = 29, pval = 0.0226, digits = 2)),
  s(fmtt(46.23, "Q", df = 29, pval = 0.0226, digits = 2, format = 2)),
  s(fmtt(8.75, "F", df1 = 2, df2 = 35, pval = 0.00083, digits = c(2, 3))),
  s(fmtt(8.75, "F", df1 = 2, df2 = 35.4567, pval = 0.00083, digits = c(2, 3), format = 2, pname = "p")),
  s(fmtt(8.75, "F", df1 = 2, df2 = 35, pval = 0.00083, digits = c(2, 3), format = 2, pname = "p", sep = FALSE)),
  s(fmtt(-1.5, "z", pval = 0.13, digits = 3, flag = " ", sep = FALSE))
)

dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, data = dat)
setmfopt(digits = 2)
out$print_digits2 <- capture.output(print(res))
setmfopt(digits = NULL, space = FALSE)
out$print_nospace <- capture.output(print(res))
setmfopt(space = TRUE)
out$print_default <- capture.output(print(res))
write(toJSON(out, auto_unbox = FALSE, digits = NA, na = "null"), "src/tests/reference/phase7/formatters.json")
