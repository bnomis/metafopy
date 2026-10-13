# Reference values for metafor's transformation functions (R/transf.r): every
# exported transf.* function on a grid of inputs, including values at and
# beyond the edges of each domain, NA, and the error conditions, plus a few
# summary.escalc() cases (including transf= and out.names=).
#
# Each case stores the call ("fun" and its arguments) next to the result, so
# the Python tests need no R. Cases with "each": true were evaluated one xi at
# a time (sapply), which is how predict() and summary() call these functions.
#
# Usage (from the repository root):  Rscript src/tests/r/generate_transf.R

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

num <- function(x) if (is.null(x)) NULL else enc(unname(as.vector(x)))

# na = "null" would also turn +-Inf into null, so infinite values are written as "Inf"/"-Inf"
enc <- function(x) {
  if (!is.numeric(x) || !any(is.infinite(x))) return(x)
  l <- as.list(x)
  l[is.infinite(x)] <- ifelse(x[is.infinite(x)] > 0, "Inf", "-Inf")
  l
}

cases <- list()

run <- function(name, fun, args = list(), each = FALSE) {
  f <- get(fun, envir = asNamespace("metafor"))
  call1 <- function(a) suppressWarnings(do.call(f, a))
  res <- tryCatch(
    if (each) sapply(args$xi, function(x) call1(c(list(xi = x), args[names(args) != "xi"]))) else call1(args),
    error = function(e) e
  )
  out <- list(fun = fun, args = lapply(args, enc), each = each)
  if (inherits(res, "error")) {
    out$error <- conditionMessage(res)
  } else {
    out$out <- num(res)
  }
  cases[[name]] <<- out
}

x    <- c(-Inf, -3, -2, -1, -0.5, 0, 0.3, 0.5, 1, 1.5, 2, 3, Inf, NA)
p    <- c(-0.1, 0, 0.001, 0.2, 0.5, 0.8, 0.999, 1, 1.1, NA)
asin <- c(-0.5, 0, 0.3, 0.785, 1.5, base::pi / 2, 1.6, 2, NA)

### closed forms of a single argument

for (fn in c("transf.rtoz", "transf.ztor", "transf.ztor2", "transf.logit", "transf.ilogit", "transf.probit",
             "transf.iprobit", "transf.arcsin", "transf.iarcsin", "transf.isqrt", "transf.ahw", "transf.iahw",
             "transf.abt", "transf.iabt", "transf.r2toz", "transf.dtou1", "transf.dtou2", "transf.dtou3",
             "transf.dtoovl", "transf.dtocles", "transf.clestod", "transf.dtocliffd", "transf.dtobesd",
             "transf.dtorpb", "transf.dtorbis", "transf.rpbtorbis", "transf.rtorpb", "transf.rtod", "transf.rpbtod",
             "transf.lnortod.norm", "transf.lnortod.logis", "transf.dtolnor.norm", "transf.dtolnor.logis",
             "transf.lnortortet.pearson", "transf.lnortortet.digby")) {
  run(paste0(fn, "/x"), fn, list(xi = x))
  run(paste0(fn, "/p"), fn, list(xi = p))
}
run("transf.iarcsin/asin", "transf.iarcsin", list(xi = asin))
run("transf.arcsin/asin", "transf.arcsin", list(xi = asin))

### Freeman-Tukey transformations

ft <- c(-0.1, 0, 0.05, 0.1, 0.2, 0.5, 0.785, 1.2, 1.5, 1.56, 1.6, NA)
run("transf.pft/n20", "transf.pft", list(xi = p, ni = 20))
run("transf.pft/nvec", "transf.pft", list(xi = p, ni = c(5, 10, 20, 40, 80, 5, 10, 20, 40, 80)))
run("transf.ipft/n10", "transf.ipft", list(xi = ft, ni = 10))
run("transf.ipft/n200", "transf.ipft", list(xi = ft, ni = 200))
run("transf.ipft/nvec", "transf.ipft", list(xi = ft, ni = c(5, 10, 20, 40, 80, 160, 5, 10, 20, 40, 80, 160)))
run("transf.ipft/scalar_x_nvec", "transf.ipft", list(xi = 0.4, ni = c(5, 10, 20, NA)))
run("transf.ipft.hm/ni", "transf.ipft.hm", list(xi = ft, targs = list(ni = c(10, 20, NA, 40))))
run("transf.ipft.hm/missing", "transf.ipft.hm", list(xi = ft, targs = list(tau2 = 1)))
run("transf.ipft.hm/null", "transf.ipft.hm", list(xi = ft, targs = NULL))

ir <- c(-0.1, 0, 0.01, 0.04, 0.05, 0.1, 0.5, 1, 2, NA)
run("transf.irft/t100", "transf.irft", list(xi = ir, ti = 100))
run("transf.irft/tvec", "transf.irft", list(xi = ir, ti = c(1, 10, 100, 1000, 50, 1, 10, 100, 1000, 50)))
run("transf.iirft/t100", "transf.iirft", list(xi = ir, ti = 100))
run("transf.iirft/t1", "transf.iirft", list(xi = ir, ti = 1))
run("transf.iirft/tvec", "transf.iirft", list(xi = ir, ti = c(1, 10, 100, 1000, 50, 1, 10, 100, 1000, 50)))
run("transf.iirft/roundtrip", "transf.iirft", list(xi = transf.irft(0, 1:50), ti = 1:50))

### functions with sample sizes, proportions or risks

d6 <- c(-1.5, -0.2, 0, 0.4, 1, NA)
r6 <- c(-1, -0.3, 0, 0.4, 0.95, 1)
n1 <- c(10, 25, 40, 5, 100, 12)
n2 <- c(10, 15, 60, 50, 3, 12)
run("transf.dtorpb/n", "transf.dtorpb", list(xi = d6, n1i = n1, n2i = n2))
run("transf.dtorpb/len", "transf.dtorpb", list(xi = d6, n1i = n1[-1], n2i = n2[-1]))
run("transf.dtorpb/len12", "transf.dtorpb", list(xi = d6, n1i = n1, n2i = n2[-1]))
run("transf.dtorbis/n", "transf.dtorbis", list(xi = d6, n1i = n1, n2i = n2))
run("transf.dtorbis/len", "transf.dtorbis", list(xi = d6, n1i = n1[-1], n2i = n2[-1]))
run("transf.rtod/n", "transf.rtod", list(xi = r6, n1i = n1, n2i = n2))
run("transf.rtod/neg", "transf.rtod", list(xi = r6, n1i = -n1, n2i = n2))
run("transf.rtod/len", "transf.rtod", list(xi = r6, n1i = n1[-1], n2i = n2[-1]))
run("transf.rpbtod/n", "transf.rpbtod", list(xi = r6, n1i = n1, n2i = n2))
run("transf.rpbtod/len", "transf.rpbtod", list(xi = r6, n1i = n1, n2i = n2[-1]))
run("transf.rpbtorbis/pi03", "transf.rpbtorbis", list(xi = r6, pi = 0.3))
run("transf.rpbtorbis/pivec", "transf.rpbtorbis", list(xi = r6, pi = c(0, 0.1, 0.5, 0.7, 1, NA)))
run("transf.rpbtorbis/pibad", "transf.rpbtorbis", list(xi = r6, pi = 1.2))
run("transf.rpbtorbis/pilen", "transf.rpbtorbis", list(xi = r6, pi = c(0.2, 0.3)))
run("transf.rtorpb/pi03", "transf.rtorpb", list(xi = r6, pi = 0.3))
run("transf.rtorpb/pivec", "transf.rtorpb", list(xi = r6, pi = c(0, 0.1, 0.5, 0.7, 1, NA)))
run("transf.rtorpb/pibad", "transf.rtorpb", list(xi = r6, pi = -0.2))
for (fn in c("transf.lnortord", "transf.lnortorr")) {
  run(paste0(fn, "/pc02"), fn, list(xi = x, pc = 0.2))
  run(paste0(fn, "/pc0"), fn, list(xi = x, pc = 0))
  run(paste0(fn, "/pc1"), fn, list(xi = x, pc = 1))
  run(paste0(fn, "/pcvec"), fn, list(xi = d6, pc = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
  run(paste0(fn, "/pcbad"), fn, list(xi = d6, pc = 1.5))
  run(paste0(fn, "/pclen"), fn, list(xi = d6, pc = c(0.1, 0.2)))
}
run("transf.dtomd/scalar", "transf.dtomd", list(xi = x, targs = 15))
run("transf.dtomd/list", "transf.dtomd", list(xi = x, targs = list(sd = 2.5)))
run("transf.dtomd/null", "transf.dtomd", list(xi = x, targs = NULL))
run("transf.dtomd/len2", "transf.dtomd", list(xi = x, targs = c(1, 2)))

### integral transformations

xint <- c(-3, -1, -0.5, 0, 0.3, 1, 2.5, NA)
for (fn in c("transf.ztor.int", "transf.exp.int", "transf.ilogit.int", "transf.iprobit.int", "transf.dtocles.int")) {
  for (t2 in c(0, 1e-20, 0.02, 0.3, 2)) {
    run(sprintf("%s/tau2=%g", fn, t2), fn, list(xi = xint, targs = list(tau2 = t2)))
  }
  run(paste0(fn, "/inf"), fn, list(xi = c(-Inf, Inf), targs = list(tau2 = 0.3)))
  run(paste0(fn, "/vector_targs"), fn, list(xi = xint, targs = c(0.3, -1, 1.5)))
  run(paste0(fn, "/named_targs"), fn, list(xi = xint, targs = list(tau2 = 0.3, lower = -2, upper = 3)))
  run(paste0(fn, "/scalar_targs"), fn, list(xi = xint, targs = 0.3))
  run(paste0(fn, "/null"), fn, list(xi = xint))
  run(paste0(fn, "/neg"), fn, list(xi = xint, targs = list(tau2 = -0.1)))
  run(paste0(fn, "/len4"), fn, list(xi = xint, targs = c(0.3, -1, 1, 2)))
  run(paste0(fn, "/no_tau2"), fn, list(xi = xint, targs = list(lower = -1)))
}

# these three bound the integral by max/min over *all* xi values, so a vector
# call can differ from elementwise calls (and an NA anywhere gives all NA)
dom <- list("transf.iarcsin.int" = c(0, 0.2, 0.785, 1.2, 1.5, base::pi / 2, NA),
            "transf.iahw.int" = c(0, 0.2, 0.5, 0.9, 0.99, 1, NA),
            "transf.iabt.int" = c(0, 0.2, 1, 3, 6, NA))
for (fn in names(dom)) {
  xd <- dom[[fn]]
  for (t2 in c(0, 0.001, 0.02, 0.3)) {
    run(sprintf("%s/each/tau2=%g", fn, t2), fn, list(xi = xd, targs = list(tau2 = t2)), each = TRUE)
    run(sprintf("%s/vec/tau2=%g", fn, t2), fn, list(xi = xd[!is.na(xd)], targs = list(tau2 = t2)))
  }
  run(paste0(fn, "/vec_na"), fn, list(xi = xd, targs = list(tau2 = 0.02)))
  run(paste0(fn, "/named_targs"), fn, list(xi = xd, targs = list(tau2 = 0.3, lower = 0.1, upper = 0.9)))
  run(paste0(fn, "/null"), fn, list(xi = xd))
}

### modes

xmode <- c(-2, -0.5, 0, 0.3, 1, NA)
mdom <- list("transf.ztor.mode" = xmode, "transf.exp.mode" = xmode, "transf.ilogit.mode" = xmode,
             "transf.iprobit.mode" = xmode, "transf.iarcsin.mode" = c(0, 0.2, 0.785, 1.5, NA),
             "transf.iahw.mode" = c(0, 0.2, 0.5, 0.9, NA), "transf.iabt.mode" = c(0, 0.2, 1, 3, NA))
for (fn in names(mdom)) {
  xd <- mdom[[fn]]
  run(paste0(fn, "/tau2=0"), fn, list(xi = xd, targs = 0), each = TRUE)
  for (t2 in c(0.02, 0.3, 3)) {
    run(sprintf("%s/tau2=%g", fn, t2), fn, list(xi = xd, targs = list(tau2 = t2)))
  }
  run(paste0(fn, "/scalar_targs"), fn, list(xi = xd, targs = 0.3))
  run(paste0(fn, "/null"), fn, list(xi = xd))
  run(paste0(fn, "/neg"), fn, list(xi = xd, targs = -1))
  run(paste0(fn, "/len2"), fn, list(xi = xd, targs = c(0.1, 0.2)))
}

### summary.escalc()

sumcases <- list()
pick <- function(s) {
  out <- list(columns = names(s))
  for (nm in names(s)) out[[nm]] <- num(s[[nm]])
  out$attrs <- list(yi_names = attr(s, "yi.names"), vi_names = attr(s, "vi.names"),
                    sei_names = attr(s, "sei.names"), ci_lb_names = attr(s, "ci.lb.names"))
  out
}

dat <- escalc(measure = "PFT", xi = c(2, 5, 0, 12, NA), ni = c(10, 20, 15, 30, 25))
sumcases$pft <- list(input = list(measure = "PFT", xi = c(2, 5, 0, 12, NA), ni = c(10, 20, 15, 30, 25)),
                     ni = num(attr(dat$yi, "ni")))
sumcases$pft$default <- pick(summary(dat))
sumcases$pft$level90_H0 <- pick(summary(dat, level = 90, H0 = 0.3))
sumcases$pft$transf_ipft <- pick(summary(dat, transf = transf.ipft, ni = attr(dat$yi, "ni")))
sumcases$pft$transf_noappend <- pick(summary(dat, transf = transf.ipft, ni = attr(dat$yi, "ni"), append = FALSE))
sumcases$pft$out_names <- pick(summary(dat, out.names = c("se", "z", "p", "lo", "hi")))
sumcases$pft$noappend <- pick(summary(dat, append = FALSE))

dat <- escalc(measure = "OR", ai = c(4, 6, 10, 0), n1i = c(50, 60, 80, 20), ci = c(9, 12, 10, 3), n2i = c(50, 55, 85, 20))
sumcases$or <- list(input = list(measure = "OR", ai = c(4, 6, 10, 0), n1i = c(50, 60, 80, 20), ci = c(9, 12, 10, 3), n2i = c(50, 55, 85, 20)))
sumcases$or$transf_exp <- pick(summary(dat, transf = exp))
sumcases$or$olim <- pick(summary(dat, transf = exp, olim = c(0.5, 1)))
sumcases$or$exp_int <- pick(summary(dat, transf = transf.exp.int, targs = 0.2))
s1 <- summary(dat)
s1$sei[2] <- NA
s1$ci.lb[c(1, 3)] <- NA
sumcases$or$replace_false_input <- pick(s1)
sumcases$or$replace_false <- pick(summary(s1, replace = FALSE, level = 80))

dat <- escalc(measure = "ZCOR", ri = c(0.2, 0.5, -0.1), ni = c(30, 40, 50), var.names = c("z", "v"))
sumcases$zcor <- list(input = list(measure = "ZCOR", ri = c(0.2, 0.5, -0.1), ni = c(30, 40, 50), var_names = c("z", "v")))
sumcases$zcor$ztor <- pick(summary(dat, transf = transf.ztor))
sumcases$zcor$ztor_int <- pick(summary(dat, transf = transf.ztor.int, targs = 0.05))

out <- list(cases = cases, summary = sumcases,
            meta = list(metafor = as.character(packageVersion("metafor")), R = R.version.string))
dir.create("src/tests/reference/extras", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/extras/transf.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/extras/transf.json (", length(cases), " transf cases)\n", sep = "")
