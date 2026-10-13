# Reference values for escalc(): every measure, every vtype it accepts, the
# zero-cell options (add/to/drop00, addyi/addvi/onlyo1), bias corrections,
# conversions from di/ti/pi, and the data handling options (subset, include,
# flip, slab, var.names, add.measure).
#
# Inputs are stored with each case so the Python tests need no datasets.
# Usage (from the repository root):  Rscript tests/r/generate_escalc.R
# Needs the 'gsl' (UCOR, SMCR with vtype="UB") and 'mvtnorm' (RTET) packages.

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))

cases <- list()

run <- function(name, measure, args, opts = list()) {
  warns <- character(0)
  res <- withCallingHandlers(
    tryCatch(do.call(escalc, c(list(measure = measure), args, opts)), error = function(e) e),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
  out <- list(measure = measure, args = args, opts = opts)
  if (inherits(res, "error")) {
    out$error <- conditionMessage(res)
  } else {
    out$yi <- num(res$yi)
    out$vi <- num(res$vi)
    out$ni <- num(attr(res$yi, "ni"))
    out$warnings <- warns
  }
  cases[[name]] <<- out
}

vtypes_for <- function(measure) {
  switch(measure,
    RR = c("LS", "AV", "H0"), OR = c("LS", "AV", "H0"), RD = c("UB", "LS", "AV", "H0"),
    PHI = c("ST", "CS", "LS"), ZPHI = c("ST", "CS", "LS"),
    MD = c("LS", "UB", "HO"), POMPMD = c("LS", "UB", "HO"),
    SMD = c("LS", "LS2", "LS3", "UB", "AV", "H0"), SMD1 = c("LS", "LS2", "LS3", "UB", "AV", "H0"),
    SMDH = c("LS", "LS2", "LS3"), ROM = c("LS", "HO", "AV", "AVHO", "LS2"),
    RPB = c("ST", "CS", "LS"), ZPB = c("ST", "CS", "LS"),
    VR = c("LS", "LS2"), CVR = c("LS", "LS2"),
    CLES = c("LS", "LS2", "H0", "MAX"), AUC = c("LS", "LS2", "H0", "MAX"),
    CLESN = c("LS", "LS2", "LS3", "HO", "H0"), AUCN = c("LS", "LS2", "LS3", "HO", "H0"),
    COR = c("LS", "AV", "H0", "H0a", "H0b"), UCOR = c("LS", "UB", "AV"),
    PCOR = c("LS", "AV"), SPCOR = c("LS", "AV"), ZSPCOR = c("LS", "AV"),
    R2 = c("LS", "AV", "LS2", "AV2"),
    PR = c("LS", "UB", "AV"), PLN = c("LS", "AV"), PLO = c("LS", "AV"),
    MN = c("LS", "HO"), POMPMN = c("LS", "HO"),
    SMCC = c("LS", "LS2", "UB", "H0"), SMCR = c("LS", "LS2", "UB", "H0"),
    SMCRH = c("LS", "LS2"), SMCRPH = c("LS", "LS2"),
    ROMC = c("LS", "LS2"), VRC = c("LS", "LS2"), CVRC = c("LS", "LS2"),
    "LS")
}

grid <- function(prefix, measures, args, extra = list()) {
  for (m in measures) {
    for (vt in vtypes_for(m)) {
      run(paste(prefix, m, vt, sep = "_"), m, args, c(list(vtype = vt), extra))
    }
  }
}

### 2x2 tables --------------------------------------------------------------

bcg <- with(dat.bcg, list(ai = tpos, bi = tneg, ci = cpos, di = cneg))
zero <- list(ai = c(0, 5, 10, 3, 0, 12, 7, NA, 4, 9),
             bi = c(10, 0, 5, 7, 20, 8, 13, 9, 6, 0),
             ci = c(2, 4, 0, 6, 0, 9, 0, 4, 5, 3),
             di = c(8, 6, 15, 0, 25, 11, 20, 6, 5, 0))
tbl_measures <- c("RR", "OR", "PETO", "RD", "AS", "PHI", "ZPHI", "YUQ", "YUY", "PBIT", "OR2D", "OR2DN", "OR2DL",
                  "MPRD", "MPRR", "MPOR", "MPORC", "MPPETO")

grid("bcg", tbl_measures, bcg)
grid("zero", tbl_measures, zero)
for (to in c("all", "only0", "if0all", "none")) {
  for (m in c("RR", "OR", "RD", "AS", "PETO", "YUQ")) {
    run(paste("zero_to", to, m, sep = "_"), m, zero, list(to = to))
    run(paste("zero_to", to, m, "add1", sep = "_"), m, zero, list(to = to, add = 1))
  }
}
for (m in c("RR", "OR", "RD")) {
  for (vt in vtypes_for(m)) {
    run(paste("zero_drop00", m, vt, sep = "_"), m, zero, list(drop00 = TRUE, vtype = vt))
    run(paste("zero_noaddyi", m, vt, sep = "_"), m, zero, list(addyi = FALSE, vtype = vt))
    run(paste("zero_noaddvi", m, vt, sep = "_"), m, zero, list(addvi = FALSE, vtype = vt))
    run(paste("zero_onlyo1", m, vt, sep = "_"), m, zero, list(onlyo1 = TRUE, vtype = vt))
  }
}
run("bcg_n1i_n2i_OR", "OR", with(dat.bcg, list(ai = tpos, n1i = tpos + tneg, ci = cpos, n2i = cpos + cneg)))
run("bcg_vtype_vector_RR", "RR", bcg, list(vtype = rep(c("LS", "AV", "H0"), length.out = 13)))
run("bcg_MPORM_ri", "MPORM", c(bcg, list(ri = rep(c(0.2, 0.5, NA), length.out = 13))), list())
run("bcg_MPORM_pi", "MPORM", c(bcg, list(pi = rep(c(0.01, 0.02), length.out = 13))), list())
run("bcg_MPORM_ri_noaddvi", "MPORM", c(zero, list(ri = rep(0.3, 10))), list(addvi = FALSE))
run("bcg_count_error", "RR", list(ai = c(5, 3), bi = c(-1, 4), ci = c(2, 2), di = c(3, 3)))

tet <- list(ai = c(20, 15, 40, 0, 10, 30, 5, 12), bi = c(10, 5, 10, 8, 0, 10, 15, 0),
            ci = c(5, 15, 10, 9, 10, 0, 15, 0), di = c(25, 15, 40, 20, 10, 30, 5, 30))
run("tet_RTET", "RTET", tet)
run("tet_ZTET", "ZTET", tet)

### person-time data ------------------------------------------------------------

hart <- with(dat.hart1999, list(x1i = x1i, t1i = t1i, x2i = x2i, t2i = t2i))
ptz <- list(x1i = c(0, 4, 10, 0, 7), t1i = c(100, 120, 300, 80, 150), x2i = c(3, 0, 12, 0, 9), t2i = c(110, 100, 280, 90, 160))
for (m in c("IRR", "IRD", "IRSD")) {
  run(paste("hart", m, sep = "_"), m, hart)
  run(paste("ptz", m, sep = "_"), m, ptz)
  run(paste("ptz", m, "drop00", sep = "_"), m, ptz, list(drop00 = TRUE))
  run(paste("ptz", m, "all", sep = "_"), m, ptz, list(to = "all"))
  run(paste("ptz", m, "if0all", sep = "_"), m, ptz, list(to = "if0all"))
  run(paste("ptz", m, "noaddyi", sep = "_"), m, ptz, list(addyi = FALSE))
  run(paste("ptz", m, "noaddvi", sep = "_"), m, ptz, list(addvi = FALSE))
}

### two-group means ------------------------------------------------------------

norm <- with(dat.normand1999, list(m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i))
curtis <- with(dat.curtis1998, list(m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i))
mean_measures <- c("MD", "SMD", "SMDH", "SMD1", "SMD1H", "ROM", "RPB", "ZPB", "RBIS", "ZBIS", "D2OR", "D2ORN", "D2ORL", "VR", "CVR")
grid("normand", mean_measures, norm)
grid("curtis", mean_measures, curtis)
grid("normand_pomp", "POMPMD", c(norm, list(mini = rep(0, 9), maxi = rep(200, 9))))
for (m in c("SMD", "SMD1", "SMDH")) {
  run(paste("normand", m, "approx", sep = "_"), m, norm, list(correct = "approx"))
  run(paste("normand", m, "nocorrect", sep = "_"), m, norm, list(correct = FALSE))
}
for (m in c("ROM", "VR", "CVR")) run(paste("curtis", m, "nocorrect", sep = "_"), m, curtis, list(correct = FALSE))
conv <- list(m1i = c(10, NA, NA, NA, 12), sd1i = c(2, NA, NA, NA, 3), m2i = c(9, NA, NA, NA, 11), sd2i = c(2.5, NA, NA, NA, 2),
             n1i = c(20, 30, 25, 40, 15), n2i = c(22, 28, 25, 35, 18),
             di = c(NA, 0.4, NA, NA, NA), ti = c(NA, NA, 2.1, NA, NA), pi = c(NA, NA, NA, -0.03, NA))
for (m in c("SMD", "RPB", "RBIS", "D2OR")) run(paste("conv", m, sep = "_"), m, conv)
run("conv_ri_SMD", "SMD", list(n1i = c(20, 30), n2i = c(22, 28), ri = c(0.3, -0.2)))
run("smd_large_warn", "SMD", list(m1i = c(10, 50), sd1i = c(1, 1), n1i = c(10, 10), m2i = c(9, 1), sd2i = c(1, 1), n2i = c(10, 10)))

### CLES / AUC ----------------------------------------------------------------

auc <- list(ai = c(0.6, 0.72, 0.55, 0.81, 0.5), n1i = c(20, 35, 50, 15, 40), n2i = c(25, 30, 45, 20, 40), mi = c(0.1, NA, 0, 0.2, 0.05))
grid("auc", c("CLES", "AUC"), auc)
grid("normand", c("CLESN", "AUCN"), norm)
clesn_conv <- list(m1i = c(10, NA, NA, NA, 12), sd1i = c(2, NA, NA, NA, NA), m2i = c(9, NA, NA, NA, 11), sd2i = c(2.5, NA, NA, NA, NA),
                   n1i = c(20, 30, 25, 40, 15), n2i = c(22, 28, 25, 35, 18),
                   di = c(NA, 0.4, NA, NA, NA), ti = c(NA, NA, 2.1, NA, NA), ai = c(NA, NA, NA, 0.65, NA))
run("conv_CLESN", "CLESN", clesn_conv)

### correlations ----------------------------------------------------------------

mol <- with(dat.molloy2014, list(ri = ri, ni = ni))
cors <- list(ri = c(0.3, -0.5, 0.9, 0.1, 0, 0.45), ni = c(20, 4, 50, 3, 100, 5))
grid("molloy", c("COR", "UCOR", "ZCOR"), mol)
grid("small", c("COR", "UCOR", "ZCOR"), cors)
run("conv_COR", "COR", list(ri = c(0.3, NA, NA), ni = c(30, 40, 50), ti = c(NA, 2.5, NA), pi = c(NA, NA, 0.04)))
pc <- list(ri = c(0.3, -0.2, 0.5, NA), mi = c(2, 3, 1, 4), ni = c(50, 80, 30, 60), r2i = c(0.4, 0.3, 0.6, 0.5), ti = c(NA, NA, NA, 1.8))
grid("pc", c("PCOR", "ZPCOR", "SPCOR", "ZSPCOR"), pc)
r2 <- list(r2i = c(0.2, 0.45, 0.1, NA), mi = c(3, 2, 5, 4), ni = c(60, 40, 100, 80), fi = c(NA, NA, NA, 4.2))
grid("r2", c("R2", "ZR2", "R2F", "ZR2F"), r2)

### proportions and rates -------------------------------------------------------

pritz <- with(dat.pritz1997, list(xi = xi, ni = ni))
pz <- list(xi = c(0, 5, 20, 12, 30), ni = c(25, 40, 20, 50, 60))
grid("pritz", c("PR", "PLN", "PLO", "PRZ", "PAS", "PFT"), pritz)
grid("pz", c("PR", "PLN", "PLO", "PRZ", "PAS", "PFT"), pz)
for (m in c("PR", "PLN", "PLO")) {
  run(paste("pz", m, "all", sep = "_"), m, pz, list(to = "all"))
  run(paste("pz", m, "if0all", sep = "_"), m, pz, list(to = "if0all"))
  run(paste("pz", m, "noaddyi", sep = "_"), m, pz, list(addyi = FALSE))
  run(paste("pz", m, "noaddvi_AV", sep = "_"), m, pz, list(addvi = FALSE, vtype = "AV"))
}
run("pz_mi_PLO", "PLO", list(xi = c(0, 5, 20), mi = c(25, 35, 0)))
ir <- list(xi = c(0, 5, 20, 12), ti = c(100, 250, 400, 180))
for (m in c("IR", "IRLN", "IRS", "IRFT")) {
  run(paste("ir", m, sep = "_"), m, ir)
  run(paste("ir", m, "all", sep = "_"), m, ir, list(to = "all"))
  run(paste("ir", m, "noaddyi", sep = "_"), m, ir, list(addyi = FALSE))
  run(paste("ir", m, "noaddvi", sep = "_"), m, ir, list(addvi = FALSE))
}

### single-group means ------------------------------------------------------------

mn <- list(mi = c(10, 25, 7.5, 40), sdi = c(3, 8, 2, 10), ni = c(20, 35, 15, 60))
grid("mn", c("MN", "SMN", "MNLN", "SDLN", "CVLN"), mn)
grid("mn_pomp", "POMPMN", c(mn, list(mini = rep(0, 4), maxi = rep(50, 4))))
for (m in c("SMN", "SDLN", "CVLN")) run(paste("mn", m, "nocorrect", sep = "_"), m, mn, list(correct = FALSE))

### dependent samples ------------------------------------------------------------

mc <- list(m1i = c(12, 20, 8.5, 30), m2i = c(10, 21, 7, 25), sd1i = c(3, 5, 2, 7), sd2i = c(3.5, 4, 2.5, 6), ni = c(20, 35, 15, 40), ri = c(0.5, 0.7, 0.3, 0.6))
change_measures <- c("MC", "SMCC", "SMCR", "SMCRH", "SMCRP", "SMCRPH", "CLESCN", "AUCCN", "ROMC", "VRC", "CVRC")
grid("mc", change_measures, mc)
grid("mc_pomp", "POMPMC", c(mc, list(mini = rep(0, 4), maxi = rep(50, 4))))
run("mc_scalar_ri_SMCR", "SMCR", c(mc[c("m1i", "m2i", "sd1i", "ni")], list(ri = 0.5)))
for (m in c("SMCC", "SMCR", "ROMC", "CVRC")) run(paste("mc", m, "nocorrect", sep = "_"), m, mc, list(correct = FALSE))
run("conv_SMCC", "SMCC", list(m1i = c(12, NA, NA, NA), m2i = c(10, NA, NA, NA), sd1i = c(3, NA, NA, NA), sd2i = c(3.5, NA, NA, NA),
                               ni = c(20, 30, 25, 40), ri = c(0.5, NA, NA, NA), di = c(NA, 0.4, NA, NA), ti = c(NA, NA, 2.1, NA), pi = c(NA, NA, NA, 0.02)))

### alpha, REH ------------------------------------------------------------------

alpha <- list(ai = c(0.8, 0.65, 0.9, 0.72), mi = c(10, 5, 20, 8), ni = c(100, 50, 200, 80))
for (m in c("ARAW", "AHW", "ABT")) run(paste("alpha", m, sep = "_"), m, alpha)
run("reh_REH", "REH", list(ai = c(30, 50, 12), bi = c(50, 40, 30), ci = c(20, 10, 8)))

### generic outcomes and data handling -------------------------------------------

run("gen_vi", "GEN", list(yi = c(0.1, 0.5, -0.2, NA), vi = c(0.01, 0.04, 0.02, 0.03)))
run("gen_sei", "GEN", list(yi = c(0.1, 0.5, -0.2), sei = c(0.1, 0.2, 0.15), ni = c(30, 40, 50)))
run("bcg_subset", "RR", bcg, list(subset = c(TRUE, FALSE, rep(TRUE, 10), FALSE)))
run("bcg_include", "RR", bcg, list(include = c(rep(TRUE, 5), FALSE, rep(TRUE, 7))))
run("bcg_flip", "RR", bcg, list(flip = c(TRUE, FALSE, TRUE, rep(FALSE, 10))))
run("bcg_flip_numeric", "RR", bcg, list(flip = c(-1, 1, 2, rep(1, 10))))

payload <- list(meta = list(metafor = as.character(packageVersion("metafor")), R = R.version.string), cases = cases)
writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = FALSE), "tests/reference/escalc/escalc.json")
cat("wrote", length(cases), "cases\n")
