# Reference values for the data-preparation helpers: aggregate.escalc(), conv.2x2(), conv.delta(), conv.fivenum(),
# conv.wald(), to.long(), to.table(), to.wide(), contrmat(), vec2mat(), dfround(), replmiss() and deltamethod(), mostly
# taken from the examples on their help pages.
#
# Usage (from the repository root):  R_LIBS=/tmp/Rlib LC_ALL=C.UTF-8 Rscript src/tests/r/generate_data_helpers.R
# (deltamethod() needs the 'calculus' package.)

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

out_dir <- "src/tests/reference/phase7"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

num <- function(x) unname(as.vector(x))
# a data frame as a list of columns (factors as character), with the column order kept in 'names'
df <- function(d) {
  d <- as.data.frame(d)
  cols <- lapply(d, function(col) if (is.factor(col)) as.character(col) else unname(as.vector(col)))
  names(cols) <- NULL
  list(names = names(d), columns = cols)
}
mat <- function(m) list(values = unname(as.matrix(m)), rownames = rownames(m), colnames = colnames(m))
save <- function(x, name) write(toJSON(x, auto_unbox = TRUE, digits = NA, na = "null", null = "null", matrix = "rowmajor"), file.path(out_dir, paste0(name, ".json")))
warns <- function(expr) {
  w <- character(0)
  val <- withCallingHandlers(expr, warning = function(cnd) {
    w <<- c(w, conditionMessage(cnd))
    invokeRestart("muffleWarning")
  })
  list(value = val, warnings = w)
}

############################################################################
# aggregate.escalc()

agg <- list()

dat <- dat.konstantopoulos2011
agg$id_addk <- df(aggregate(dat, cluster = district, struct = "ID", addk = TRUE))

dat <- escalc(measure = "SMD", yi = yi, vi = vi, data = as.data.frame(dat.assink2016))
agg$cs <- df(aggregate(dat, cluster = study, rho = 0.6))
agg$cs_unweighted <- df(aggregate(dat, cluster = study, rho = 0.6, weighted = FALSE))
rho_cl <- seq(0.1, 0.8, length.out = length(unique(dat$study)))
agg$rho_vector <- rho_cl
agg$cs_rho_vector <- df(aggregate(dat, cluster = study, rho = rho_cl))
agg$cs_subset <- df(aggregate(dat, cluster = study, rho = 0.6, subset = dat$pubstatus == 1))
agg$cs_select <- df(aggregate(dat, cluster = study, rho = 0.6, select = c("study", "yi", "vi", "deltype"), addk = TRUE))
V1 <- vcalc(vi, cluster = study, obs = esid, data = dat, rho = 0.6)
V2 <- vcalc(vi, cluster = study, type = deltype, obs = esid, data = dat, rho = c(0.7, 0.5))
agg$V2 <- unname(as.matrix(V2))
agg$V1 <- df(aggregate(dat, cluster = study, V = V1))
agg$V2_agg <- df(aggregate(dat, cluster = study, V = V2))
agg$V_vector <- df(aggregate(dat, cluster = study, V = dat$vi))

# missing values
datna <- dat
datna$yi[c(2, 10)] <- NA
datna$deltype[c(1, 3)] <- NA
datna$year[5] <- NA
agg$na_rm <- df(aggregate(datna, cluster = study, rho = 0.6))
agg$na_rm_ff <- df(aggregate(datna, cluster = study, rho = 0.6, na.rm = FALSE))
agg$na_rm_tf <- df(aggregate(datna, cluster = study, rho = 0.6, na.rm = c(TRUE, FALSE)))

# var.names (columns renamed)
datvn <- as.data.frame(dat)
names(datvn)[4:5] <- c("es", "var")
agg$var_names <- df(aggregate(escalc(measure = "SMD", yi = es, vi = var, data = datvn, var.names = c("es", "var")), cluster = study, rho = 0.6, var.names = c("es", "var")))

# CAR and its combinations with CS on dat.ishak2007 in long format
dat <- reshape(dat.ishak2007, direction = "long", idvar = "study", v.names = c("yi", "vi"), varying = list(c(2, 4, 6, 8), c(3, 5, 7, 9)))
dat <- dat[order(dat$study, dat$time), ]
dat <- dat[!is.na(dat$yi), ]
rownames(dat) <- NULL
dat <- escalc(measure = "MD", yi = yi, vi = vi, data = dat)
agg$ishak_long <- df(dat)
agg$car <- df(aggregate(dat, cluster = study, struct = "CAR", time = time, phi = 0.9))
agg$cs_car <- df(aggregate(dat, cluster = study, struct = "CS+CAR", time = time, phi = 0.9, rho = 0.3))
dat$obs <- ifelse(dat$time <= 2, 1, 2)
agg$obs <- dat$obs
agg$csxcar <- df(aggregate(dat, cluster = study, struct = "CS*CAR", time = time, obs = obs, phi = 0.9, rho = 0.3))
agg$car_unweighted <- df(aggregate(dat, cluster = study, struct = "CAR", time = time, phi = 0.5, weighted = FALSE, addk = TRUE))
agg$digits <- attr(aggregate(dat, cluster = study, struct = "CAR", time = time, phi = 0.5, digits = 2), "digits")

save(agg, "aggregate_escalc")

############################################################################
# conv.2x2()

c2 <- list()

mk <- function(scale) {
  dat <- data.frame(ai = c(36, NA, NA, NA) * scale, bi = c(86, NA, NA, NA) * scale, ci = c(20, NA, NA, NA) * scale, di = c(98, NA, NA, NA) * scale,
                    oddsratio = NA, phi = NA, chisq = NA, ni = NA, n1i = NA, n2i = NA)
  dat$oddsratio[2] <- round(exp(escalc(measure = "OR", ai = ai, bi = bi, ci = ci, di = di, data = dat)$yi[1]), 2)
  dat$phi[3] <- round(escalc(measure = "PHI", ai = ai, bi = bi, ci = ci, di = di, data = dat)$yi[1], 2)
  dat$chisq[4] <- round(chisq.test(matrix(c(t(dat[1, 1:4])), nrow = 2, byrow = TRUE))$statistic, 2)
  dat$ni[2:4] <- with(dat, ai[1] + bi[1] + ci[1] + di[1])
  dat$n1i[2:4] <- with(dat, ai[1] + bi[1])
  dat$n2i[2:4] <- with(dat, ai[1] + ci[1])
  dat
}
dat <- mk(1)
c2$input1 <- df(dat)
c2$data1 <- df(conv.2x2(ri = phi, ori = oddsratio, x2i = chisq, ni = ni, n1i = n1i, n2i = n2i, data = dat))
c2$data1_noappend <- df(conv.2x2(ri = phi, ori = oddsratio, x2i = chisq, ni = ni, n1i = n1i, n2i = n2i, data = dat, append = FALSE))
c2$data1_names <- df(conv.2x2(ri = phi, ori = oddsratio, x2i = chisq, ni = ni, n1i = n1i, n2i = n2i, data = dat, var.names = c("a", "b", "c", "d")))
c2$data1_all <- df(conv.2x2(ri = phi, ori = oddsratio, x2i = chisq, ni = ni, n1i = n1i, n2i = n2i, data = dat, replace = "all"))
c2$data1_include <- df(conv.2x2(ri = phi, ori = oddsratio, x2i = chisq, ni = ni, n1i = n1i, n2i = n2i, data = dat, include = c(TRUE, TRUE, FALSE, TRUE)))
dat <- mk(10)
c2$input10 <- df(dat)
c2$data10 <- df(conv.2x2(ri = phi, ori = oddsratio, x2i = chisq, ni = ni, n1i = n1i, n2i = n2i, data = dat))
c2$true_marg <- df(conv.2x2(ri = 0.54, ni = 400, n1i = 200, n2i = 248))
c2$guess_marg <- df(conv.2x2(ri = 0.54, ni = 400, n1i = 200, n2i = 200))
tmp <- warns(conv.2x2(ri = 0.54, ni = 400, n1i = 200, n2i = 50))
c2$incompat <- df(tmp$value)
c2$incompat_warn <- tmp$warnings
chisq <- round(chisq.test(matrix(c(40, 60, 60, 40), nrow = 2, byrow = TRUE))$statistic, 2)
c2$chisq_sign <- num(chisq)
c2$neg_chisq <- df(conv.2x2(x2i = -chisq, ni = 200, n1i = 100, n2i = 100))
c2$pos_chisq <- df(conv.2x2(x2i = chisq, ni = 200, n1i = 100, n2i = 100))
tab <- matrix(c(28, 14, 12, 18), nrow = 2, byrow = TRUE)
c2$chisq_yates <- num(round(chisq.test(tab)$statistic, 2))
c2$yates <- df(conv.2x2(x2i = c2$chisq_yates, ni = 72, n1i = 42, n2i = 40))
c2$chisq_noyates <- num(round(chisq.test(tab, correct = FALSE)$statistic, 2))
c2$noyates_correct <- df(conv.2x2(x2i = c2$chisq_noyates, ni = 72, n1i = 42, n2i = 40))
c2$noyates <- df(conv.2x2(x2i = c2$chisq_noyates, ni = 72, n1i = 42, n2i = 40, correct = FALSE))

tab <- matrix(c(28, 5, 7, 18), nrow = 2, byrow = TRUE)
dor <- tab[1, 1] * tab[2, 2] / (tab[1, 2] * tab[2, 1])
cases <- tab[1, 1] + tab[2, 1]
pos <- tab[1, 1] + tab[1, 2]
n <- sum(tab)
sens <- tab[1, 1] / sum(tab[, 1])
spec <- tab[2, 2] / sum(tab[, 2])
ppv <- tab[1, 1] / sum(tab[1, ])
npv <- tab[2, 2] / sum(tab[2, ])
c2$diag <- list(dor = dor, cases = cases, pos = pos, n = n, sens = sens, spec = spec, ppv = ppv, npv = npv)
c2$dor <- df(conv.2x2(ori = dor, n1i = pos, n2i = cases, ni = n))
c2$sspn <- df(conv.2x2(sens = sens, spec = spec, ppv = ppv, npv = npv, ni = n))
c2$ssp <- df(conv.2x2(sens = sens, spec = spec, ppv = ppv, ni = n))
c2$ssn <- df(conv.2x2(sens = sens, spec = spec, npv = npv, ni = n))
c2$spn <- df(conv.2x2(sens = sens, ppv = ppv, npv = npv, ni = n))
c2$xpn <- df(conv.2x2(spec = spec, ppv = ppv, npv = npv, ni = n))
r2 <- function(x) round(x, digits = 2)
c2$diag_rounded <- list(dor = r2(dor), sens = r2(sens), spec = r2(spec), ppv = r2(ppv), npv = r2(npv))
c2$dor_rounded <- df(conv.2x2(ori = r2(dor), n1i = pos, n2i = cases, ni = n))
c2$sspn_rounded <- df(conv.2x2(sens = r2(sens), spec = r2(spec), ppv = r2(ppv), npv = r2(npv), ni = n))
c2$sspn_rounded_simple <- df(conv.2x2(sens = r2(sens), spec = r2(spec), ppv = r2(ppv), npv = r2(npv), ni = n, method = "simple"))
c2$sspn_rounded_search <- df(conv.2x2(sens = r2(sens), spec = r2(spec), ppv = r2(ppv), npv = r2(npv), ni = n, method = "search"))
c2$drop01 <- df(conv.2x2(sens = c(sens, 1), spec = c(spec, spec), ppv = c(ppv, ppv), ni = c(n, n)))
c2$drop01_false <- df(conv.2x2(sens = c(sens, 1), spec = c(spec, spec), ppv = c(ppv, ppv), ni = c(n, n), drop01 = FALSE))

save(c2, "conv_2x2")

############################################################################
# conv.delta() and conv.wald()

cd <- list()
dat <- escalc(measure = "PR", xi = c(5, 12), ni = c(40, 80))
cd$pr <- df(dat)
cd$plo <- df(escalc(measure = "PLO", xi = c(5, 12), ni = c(40, 80)))
cd$logit_vec <- df(conv.delta(dat$yi, dat$vi, transf = transf.logit))
cd$logit_names <- df(conv.delta(yi, vi, data = dat, transf = transf.logit, var.names = c("yi.t", "vi.t")))
cd$logit_all <- df(conv.delta(yi, vi, data = dat, transf = transf.logit, replace = "all"))
cd$logit_noappend <- df(conv.delta(yi, vi, data = dat, transf = transf.logit, var.names = c("yi.t", "vi.t"), append = FALSE))
cd$logit_include <- df(conv.delta(dat$yi, dat$vi, transf = transf.logit, include = c(FALSE, TRUE)))
cd$logit_ni <- num(attr(conv.delta(dat$yi, dat$vi, transf = transf.logit, ni = c(40, 80))$yi, "ni"))

dat <- escalc(measure = "SMD", m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i, data = dat.normand1999, correct = FALSE)
cd$smd <- df(dat)
cd$d2orn <- df(escalc(measure = "D2ORN", m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i, data = dat.normand1999))
cd$dtolnor <- df(conv.delta(yi, vi, data = dat, transf = transf.dtolnor.norm, replace = "all"))
cd$rpb <- df(escalc(measure = "RPB", m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i, data = dat.normand1999))
cd$dtorpb <- df(conv.delta(yi, vi, data = dat, transf = transf.dtorpb, n1i = n1i, n2i = n2i, replace = "all"))

# the SMD data preparation example
dat <- data.frame(study = 1:6,
                  m1i = c(2.03, NA, NA, NA, NA, NA), sd1i = c(0.95, NA, NA, NA, NA, NA), n1i = c(32, 95, 145, NA, NA, NA),
                  m2i = c(1.25, NA, NA, NA, NA, NA), sd2i = c(1.04, NA, NA, NA, NA, NA), n2i = c(30, 99, 155, NA, NA, NA),
                  tval = c(NA, 2.12, NA, NA, NA, NA), dval = c(NA, NA, 0.37, NA, NA, NA),
                  ai = c(NA, NA, NA, 26, NA, NA), bi = c(NA, NA, NA, 58, NA, NA),
                  ci = c(NA, NA, NA, 11, NA, NA), di = c(NA, NA, NA, 74, NA, NA),
                  or = c(NA, NA, NA, NA, 2.56, NA), lower = c(NA, NA, NA, NA, 1.23, NA), upper = c(NA, NA, NA, NA, 5.30, NA),
                  corr = c(NA, NA, NA, NA, NA, 0.32), ntot = c(NA, NA, NA, NA, NA, 86))
cd$smd_input <- df(dat)
dat <- escalc(measure = "SMD", m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i, ti = tval, di = dval, data = dat)
dat <- escalc(measure = "OR2DN", ai = ai, bi = bi, ci = ci, di = di, data = dat, replace = FALSE)
cd$smd_step2 <- df(dat)
dat <- conv.wald(out = or, ci.lb = lower, ci.ub = upper, data = dat, transf = log, var.names = c("lnor", "vlnor"))
cd$smd_step3 <- df(dat)
dat <- conv.delta(lnor, vlnor, data = dat, transf = transf.lnortod.norm, var.names = c("yi", "vi"))
cd$smd_step4 <- df(dat)
dat$lnor <- NULL
dat$vlnor <- NULL
dat <- escalc(measure = "COR", ri = corr, ni = ntot, data = dat, var.names = c("ri", "vri"))
dat <- conv.delta(ri, vri, data = dat, transf = transf.rtod, var.names = c("yi", "vi"))
cd$smd_step5 <- df(dat)

# the correlation data preparation example
dat <- data.frame(study = 1:6,
                  ri = c(0.42, NA, NA, NA, NA, NA), tval = c(NA, 2.85, NA, NA, NA, NA), phi = c(NA, NA, NA, 0.27, NA, NA),
                  ni = c(93, 182, NA, 112, NA, NA),
                  ai = c(NA, NA, NA, NA, 61, NA), bi = c(NA, NA, NA, NA, 36, NA), ci = c(NA, NA, NA, NA, 39, NA), di = c(NA, NA, NA, NA, 57, NA),
                  or = c(NA, NA, NA, NA, NA, 1.86), lower = c(NA, NA, NA, NA, NA, 1.12), upper = c(NA, NA, NA, NA, NA, 3.10),
                  m1i = c(NA, NA, 54.1, NA, NA, NA), sd1i = c(NA, NA, 5.79, NA, NA, NA), n1i = c(NA, NA, 66, 75, NA, NA),
                  m2i = c(NA, NA, 51.7, NA, NA, NA), sd2i = c(NA, NA, 6.23, NA, NA, NA), n2i = c(NA, NA, 65, 88, NA, NA))
cd$cor_input <- df(dat)
dat <- escalc(measure = "COR", ri = ri, ni = ni, ti = tval, data = dat)
dat <- escalc(measure = "RBIS", m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i, data = dat, replace = FALSE)
dat <- conv.2x2(ri = phi, ni = ni, n1i = n1i, n2i = n2i, data = dat)
cd$cor_step3 <- df(dat)
dat <- escalc(measure = "RTET", ai = ai, bi = bi, ci = ci, di = di, data = dat, replace = FALSE)
dat <- conv.wald(out = or, ci.lb = lower, ci.ub = upper, data = dat, transf = log, var.names = c("lnor", "vlnor"))
dat <- conv.delta(lnor, vlnor, data = dat, transf = transf.lnortortet.pearson, var.names = c("yi", "vi"))
cd$cor_step6 <- df(dat)

# conv.wald() examples
dat <- data.frame(or = c(1.37, 1.89), or.lb = c(1.03, 1.60), or.ub = c(1.82, 2.23))
cd$wald_simple <- df(conv.wald(out = or, ci.lb = or.lb, ci.ub = or.ub, data = dat, transf = log))
cd$wald_simple_level <- df(conv.wald(out = or, ci.lb = or.lb, ci.ub = or.ub, data = dat, transf = log, level = c(95, 90)))
cd$wald_noappend <- df(conv.wald(out = or, ci.lb = or.lb, ci.ub = or.ub, data = dat, transf = log, append = FALSE, var.names = c("lor", "vlor")))
cd$wald_ub_only <- df(conv.wald(out = log(c(1.37, 1.89)), ci.ub = log(c(1.82, 2.23))))
cd$wald_zval <- df(conv.wald(out = c(0.3, -0.2), zval = c(2.1, -1.5)))
tmp <- warns(conv.wald(out = c(1.37, 1.0), ci.lb = c(1.03, 1.60), ci.ub = c(1.82, 2.23), transf = log))
cd$wald_check <- df(tmp$value)
cd$wald_check_warn <- tmp$warnings

dat <- dat.bcg[, c(2:7)]
dat1 <- escalc(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat)
dat2 <- data.frame(summary(dat1))
dat2[c("yi", "ci.lb", "ci.ub")] <- data.frame(summary(dat1, transf = exp))[c("yi", "ci.lb", "ci.ub")]
names(dat2)[which(names(dat2) == "yi")] <- "or"
dat2[, c("or", "ci.lb", "ci.ub", "pval")] <- round(dat2[, c("or", "ci.lb", "ci.ub", "pval")], digits = 2)
dat2$vi <- dat2$sei <- dat2$zi <- NULL
dat2$ntot <- with(dat2, tpos + tneg + cpos + cneg)
dat2[c(1, 12), c(3:6, 9:10)] <- NA
dat2[c(4, 9), c(3:6, 8)] <- NA
dat2[c(2:3, 5:8, 10:11, 13), c(7:10)] <- NA
dat2$ntot[!is.na(dat2$tpos)] <- NA
cd$bcg_input <- df(dat2)
dat2 <- escalc(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat2)
dat2 <- conv.wald(out = or, ci.lb = ci.lb, ci.ub = ci.ub, pval = pval, n = ntot, data = dat2, transf = log)
cd$bcg <- df(dat2)
cd$bcg_ni <- num(attr(dat2$yi, "ni"))

save(cd, "conv_delta_wald")

############################################################################
# conv.fivenum()

cf <- list()
mkf <- function(q3, max) data.frame(case = c(1:3, NA), min = c(2, NA, 2, NA), q1 = c(NA, 4, 4, NA), median = c(6, 6, 6, NA), q3 = q3, max = max,
                                    mean = c(NA, NA, NA, 7.0), sd = c(NA, NA, NA, 4.2), n = c(20, 20, 20, 20))
attrs <- function(x) {
  a <- attributes(x)
  list(tval = num(a$tval), crit = num(a$crit), sig = num(a$sig), dist = num(a$dist), est = num(a$est))
}
fivenum_case <- function(dat, ...) {
  out <- conv.fivenum(min = min, q1 = q1, median = median, q3 = q3, max = max, n = n, data = dat, ...)
  list(data = df(out), mean = attrs(out$mean), sd = attrs(out$sd))
}
dat <- mkf(c(NA, 10, 10, NA), c(14, NA, 14, NA))
cf$input1 <- df(dat)
cf$default <- fivenum_case(dat)
for (m in c("hozo2005", "wan2014", "bland2015", "walter2007")) cf[[m]] <- fivenum_case(dat, method = m)
cf$shi2020 <- fivenum_case(dat, method = c("luo2016", "shi2020"))
cf$noappend <- fivenum_case(dat, append = FALSE)
cf$names <- fivenum_case(dat, var.names = c("m", "s"))
cf$include <- fivenum_case(dat, include = c(TRUE, FALSE, TRUE, TRUE))
dat <- mkf(c(NA, 10, 14, NA), c(14, NA, 20, NA))
cf$input3 <- df(dat)
cf$skewed <- fivenum_case(dat)
cf$skewed_notest <- fivenum_case(dat, test = FALSE)
cf$lnorm <- fivenum_case(dat, dist = c("norm", "norm", "lnorm", "norm"), replace = "all")
cf$lnorm_all <- fivenum_case(dat, dist = "lnorm")
cf$lnorm_notransf <- fivenum_case(dat, dist = "lnorm", transf = FALSE)
tmp <- warns(fivenum_case(dat, dist = "lnorm", method = "wan2014"))
cf$lnorm_switch <- tmp$value
cf$lnorm_switch_warn <- tmp$warnings
# skewness tests for the three cases (with other sample sizes)
dat <- data.frame(min = c(1, NA, 1, 1, NA, 1), q1 = c(NA, 3, 3, NA, 3, 3), median = c(5, 5, 5, 5, 5, 5), q3 = c(NA, 9, 9, NA, 9, 9),
                  max = c(20, NA, 20, 20, NA, 20), n = c(50, 50, 50, 300, 300, 4))
cf$input_n <- df(dat)
cf$by_n <- fivenum_case(dat)
cf$by_n_hozo <- fivenum_case(dat, method = "hozo2005", test = FALSE)
cf$by_n_walter <- fivenum_case(dat, method = "walter2007", test = FALSE)

save(cf, "conv_fivenum")

############################################################################
# to.long(), to.table(), to.wide(), contrmat()

tl <- list()
tl$or <- df(to.long(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg))
tl$or_vlong <- df(to.long(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, vlong = TRUE))
tl$or_append_names <- df(to.long(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, append = c("author", "year")))
tl$or_append_pos <- df(to.long(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, append = 2:3))
tl$or_noappend_slab <- df(to.long(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, append = FALSE, slab = author))
tl$or_subset_n1i <- df(to.long(measure = "OR", ai = tpos, n1i = tpos + tneg, ci = cpos, n2i = cpos + cneg, data = dat.bcg, subset = 1:5))
datz <- dat.bcg[1:4, ]
datz$tpos[2] <- 0
datz$cneg[3] <- NA
tmp <- warns(to.long(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = datz, to = "only0"))
tl$or_only0 <- df(tmp$value)
tl$or_only0_warn <- tmp$warnings
tl$or_all <- df(to.long(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = datz, to = "all", add = 1, append = FALSE))
tl$or_if0all <- df(to.long(measure = "RD", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = datz, to = "if0all", append = FALSE))
tl$mpor <- df(to.long(measure = "MPOR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:3, ], append = FALSE))
tl$mpor_vlong <- df(to.long(measure = "MPOR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:3, ], vlong = TRUE))
tl$mporc <- df(to.long(measure = "MPORC", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:3, ], append = FALSE))
tl$mporc_vlong <- df(to.long(measure = "MPORC", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:3, ], vlong = TRUE, append = FALSE))
tl$irr <- df(to.long(measure = "IRR", x1i = x1i, x2i = x2i, t1i = t1i, t2i = t2i, data = dat.hart1999, var.names = c("id", "group", "events", "ptime")))
tl$md <- df(to.long(measure = "MD", m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i, data = dat.normand1999,
                    var.names = c("id", "group", "mean", "sd", "n")))
tl$cor <- df(to.long(measure = "COR", ri = ri, ni = ni, data = dat.mcdaniel1994[1:5, ]))
tl$pr <- df(to.long(measure = "PR", xi = xi, ni = ni, data = dat.pritz1997[1:4, ], append = FALSE))
tl$pr_vlong <- df(to.long(measure = "PLO", xi = xi, ni = ni, data = dat.pritz1997[1:4, ], vlong = TRUE, append = FALSE))
tl$ir <- df(to.long(measure = "IR", xi = x1i, ti = t1i, data = dat.hart1999[1:4, ], append = FALSE))
tl$mn <- df(to.long(measure = "MN", mi = m1i, sdi = sd1i, ni = n1i, data = dat.normand1999[1:4, ], append = FALSE))
tl$mc <- df(to.long(measure = "MC", m1i = m1i, m2i = m2i, sd1i = sd1i, sd2i = sd2i, ni = n1i, ri = c(0.5, 0.4, 0.3, 0.2), data = dat.normand1999[1:4, ], append = FALSE))
tl$smcr <- df(to.long(measure = "SMCR", m1i = m1i, m2i = m2i, sd1i = sd1i, ni = n1i, ri = c(0.5, 0.4, 0.3, 0.2), data = dat.normand1999[1:4, ], append = FALSE))
tl$araw <- df(to.long(measure = "ARAW", ai = c(0.8, 0.7, 0.9), mi = c(10, 12, 8), ni = c(100, 120, 80)))
tl$dup_slab <- df(to.long(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:4, ], slab = c("a", "b", "a", "c"), append = FALSE))
save(tl, "to_long")

tt <- list()
tab_out <- function(x) list(values = num(x), dim = dim(x), rows = dimnames(x)[[1]], cols = dimnames(x)[[2]], slab = dimnames(x)[[3]])
tt$or <- tab_out(to.table(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, slab = paste(author, year, sep = ", "),
                          rows = c("Vaccinated", "Not Vaccinated"), cols = c("TB+", "TB-")))
tt$or_default <- tab_out(to.table(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, subset = c(1, 3, 5)))
tt$irr <- tab_out(to.table(measure = "IRR", x1i = x1i, x2i = x2i, t1i = t1i, t2i = t2i, data = dat.hart1999, slab = paste(study, year, sep = ", "),
                           rows = c("Warfarin Group", "Placebo/Control Group")))
tt$md <- tab_out(to.table(measure = "MD", m1i = m1i, sd1i = sd1i, n1i = n1i, m2i = m2i, sd2i = sd2i, n2i = n2i, data = dat.normand1999,
                          slab = source, rows = c("Specialized Care", "Routine Care")))
tt$mprd <- tab_out(to.table(measure = "MPRD", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:3, ]))
tt$mppeto <- tab_out(to.table(measure = "MPPETO", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg[1:3, ]))
tt$cor <- tab_out(to.table(measure = "ZCOR", ri = ri, ni = ni, data = dat.mcdaniel1994[1:5, ]))
tt$pr <- tab_out(to.table(measure = "PR", xi = xi, ni = ni, data = dat.pritz1997[1:4, ], to = "all"))
tt$ir <- tab_out(to.table(measure = "IR", xi = x1i, ti = t1i, data = dat.hart1999[1:4, ]))
tt$mn <- tab_out(to.table(measure = "MN", mi = m1i, sdi = sd1i, ni = n1i, data = dat.normand1999[1:4, ]))
tt$mc <- tab_out(to.table(measure = "MC", m1i = m1i, m2i = m2i, sd1i = sd1i, sd2i = sd2i, ni = n1i, ri = c(0.5, 0.4, 0.3, 0.2), data = dat.normand1999[1:4, ]))
tt$smcr <- tab_out(to.table(measure = "SMCR", m1i = m1i, m2i = m2i, sd1i = sd1i, ni = n1i, ri = c(0.5, 0.4, 0.3, 0.2), data = dat.normand1999[1:4, ]))
tt$araw <- tab_out(to.table(measure = "ARAW", ai = c(0.8, 0.7, 0.9), mi = c(10, 12, 8), ni = c(100, 120, 80)))
tmp <- warns(to.table(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = datz, to = "only0"))
tt$or_only0 <- tab_out(tmp$value)
save(tt, "to_table")

tw <- list()
dat <- dat.senn2013
dat <- dat[c(1, 4, 3, 2, 5, 6)]
tw$senn <- df(to.wide(dat, study = "study", grp = "treatment", ref = "placebo", grpvars = 4:6))
tw$senn_contr <- df(contrmat(to.wide(dat, study = "study", grp = "treatment", ref = "placebo", grpvars = 4:6), grp1 = "treatment.1", grp2 = "treatment.2"))
tw$senn_noref <- df(to.wide(dat, study = "study", grp = "treatment", grpvars = c("ni", "mi", "sdi"), postfix = c("_a", "_b"), addid = FALSE, minlen = 4))
tw$senn_contr_noappend <- df(contrmat(to.wide(dat, study = 1, grp = 3, ref = "placebo", grpvars = 4:6, adddesign = FALSE),
                                      grp1 = "treatment.1", grp2 = "treatment.2", append = FALSE, last = NA))
dat <- dat.hasselblad1998
tw$hass <- df(to.wide(dat, study = "study", grp = "trt", ref = "no_contact", grpvars = 6:7))
tw$hass_contr <- df(contrmat(to.wide(dat, study = "study", grp = "trt", ref = "no_contact", grpvars = 6:7), grp1 = "trt.1", grp2 = "trt.2", shorten = TRUE, minlen = 3))
tw$hass_contr_last <- df(contrmat(to.wide(dat, study = "study", grp = "trt", ref = "no_contact", grpvars = 6:7), grp1 = "trt.1", grp2 = "trt.2", last = "self_help", check = FALSE, append = FALSE))
dat1 <- data.frame(study = c(1, 1, 2, 3, 3, 3), grp = c("A", "B", "A", "C", "B", "A"), y = c(1.5, 2, 3, 4, 5, 6))
tw$onearm_input <- df(dat1)
tw$onearm <- df(to.wide(dat1, study = "study", grp = "grp", ref = "A", grpvars = "y"))
save(tw, "to_wide")

############################################################################
# vec2mat(), dfround(), replmiss()

mi <- list()
mi$v1 <- vec2mat(1:6, corr = FALSE)
mi$v2 <- vec2mat(seq(0.2, 0.7, by = 0.1), corr = TRUE)
mi$v3 <- vec2mat(1:10, diag = TRUE)
mi$v4 <- vec2mat(1:6, corr = FALSE, dimnames = c("A", "B", "C", "D"))
mi$v5 <- vec2mat(1:6)
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, mods = ~ ablat + year, data = dat)
cs <- coef(summary(res))
mi$cs <- df(cs)
mi$cs_rownames <- rownames(cs)
mi$dfround <- df(dfround(cs, digits = c(2, 3, 2, 3, 2, 2)))
mi$dfround0 <- df(dfround(cs))
mi$dfround_drop0 <- df(dfround(cs, digits = 3, drop0 = c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE)))
mi$dfround_mixed <- df(dfround(data.frame(a = c("x", "y"), b = c(1.23456, -2.5), c = c(NA, 0.0049)), digits = c(0, 1, 2), drop0 = FALSE))
mi$repl1 <- replmiss(c(4, 2, 7, NA, 1, NA, 5), 0)
mi$repl2 <- replmiss(c(4, 2, 7, NA, 1, NA, 5), c(2, 3, 6, 5, 8, 1, 2))
mi$repl3 <- replmiss(x, y, data = data.frame(x = c(NA, 1), y = c(5, 6)))
mi$repl4 <- replmiss(NULL, c(1, 2))
save(mi, "misc_helpers")

############################################################################
# deltamethod()

dm <- list()
tab <- function(d) list(coef = num(coef(d)), names = names(coef(d)), vcov = unname(vcov(d)), se = num(d$tab$se), zval = num(d$tab$zval),
                        pval = num(d$tab$pval), ci_lb = num(d$tab$ci.lb), ci_ub = num(d$tab$ci.ub), print = capture.output(print(d)))

dat <- dat.craft2003
tmp <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dat)
V <- tmp$V
dat <- tmp$dat
dat$var1.var2 <- factor(dat$var1.var2, levels = c("acog.perf", "asom.perf", "conf.perf", "acog.asom", "acog.conf", "asom.conf"))
res <- rma.mv(yi, V, mods = ~ 0 + var1.var2, random = ~ var1.var2 | study, struct = "UN", data = dat)
b <- coef(res)
vb <- vcov(res)
dm$b <- num(b)
dm$vb <- unname(vb)
f6 <- function(r1, r2, r3, r4, r5, r6) {
  R <- vec2mat(c(r1, r2, r3, r4, r5, r6))
  setNames(c(solve(R[-1, -1]) %*% R[2:4, 1]), c("acog", "asom", "conf"))
}
fv <- function(r) {
  R <- vec2mat(r)
  setNames(c(solve(R[-1, -1]) %*% R[2:4, 1]), c("acog", "asom", "conf"))
}
dm$reg_args <- tab(deltamethod(b, vb, fun = f6))
dm$reg_vec <- tab(deltamethod(b, vb, fun = fv))
dm$reg_vec_o2 <- tab(deltamethod(b, vb, fun = fv, order = 2))
dm$reg_vec_o3 <- tab(deltamethod(b, vb, fun = fv, order = 3))
dm$reg_level <- tab(deltamethod(b, vb, fun = fv, level = 90, H0 = c(0.1, 0.2, 0.3), digits = 3))

dat <- dat.craft2003
tmp <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dat, rtoz = TRUE)
V <- tmp$V
dat <- tmp$dat
dat$var1.var2 <- factor(dat$var1.var2, levels = c("acog.perf", "asom.perf", "conf.perf", "acog.asom", "acog.conf", "asom.conf"))
res <- rma.mv(yi, V, mods = ~ 0 + var1.var2, random = ~ var1.var2 | study, struct = "UN", data = dat)
dm$bz <- num(coef(res))
dm$vbz <- unname(vcov(res))
dm$diff_args <- tab(deltamethod(coef(res), vcov(res), fun = function(z1, z2, z3, z4, z5, z6) transf.ztor(z1) - transf.ztor(z2)))
dm$diff_vec <- tab(deltamethod(coef(res), vcov(res), fun = function(z) transf.ztor(z[1]) - transf.ztor(z[2])))
dm$diff_o2 <- tab(deltamethod(coef(res), vcov(res), fun = function(z) transf.ztor(z[1]) - transf.ztor(z[2]), order = 2))
dm$diff_o3 <- tab(deltamethod(coef(res), vcov(res), fun = function(z) transf.ztor(z[1]) - transf.ztor(z[2]), order = 3))

# a model object (rma.uni), a scalar parameter and a diagonal vcov given as a vector
dat <- escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
res <- rma(yi, vi, mods = ~ ablat, data = dat)
dm$model <- tab(deltamethod(res, fun = function(b) c(ratio = b[2] / b[1], expb0 = exp(b[1]))))
dm$model_o2 <- tab(deltamethod(res, fun = function(b0, b1) exp(b0 + 30 * b1), order = 2))
dm$scalar <- tab(deltamethod(0.5, 0.04, fun = function(x) exp(x)))
dm$scalar_o3 <- tab(deltamethod(0.5, 0.04, fun = function(x) exp(x) + x^3, order = 3))
dm$diagvec <- tab(deltamethod(c(0.2, 1.5), c(0.01, 0.09), fun = function(a, b) c(prod = a * b, quot = a / b)))
save(dm, "deltamethod")
