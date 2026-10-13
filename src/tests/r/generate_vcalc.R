# Reference values for vcalc(), rcalc(), bldiag() and blsplit(): vcalc() on dat.assink2016,
# dat.konstantopoulos2011, dat.berkey1998 (rvars interface and type/rho), dat.knapp2017 (grp1/grp2/w1/w2
# with a rho matrix) and small synthetic data sets (time1/time2/phi, grp1/grp2/w1/w2, subgroup, sparse, retdat,
# checkpd and nearpd on non-positive-definite cases), rcalc() on dat.craft2003 (formula, matrix and list
# interfaces) and bldiag()/blsplit() on small lists, together with the error and warning messages.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_vcalc.R

suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(jsonlite)
})

options(useFancyQuotes = FALSE)
num <- function(x) if (is.null(x)) NULL else unname(as.vector(x))
mat <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  x <- as.matrix(x)
  matrix(as.numeric(x), nrow = nrow(x), ncol = ncol(x)) # drops dimnames and the vcovmat class
}
err <- function(expr) tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))
warns <- function(expr) {
  w <- character()
  value <- withCallingHandlers(expr, warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = w)
}
# a vcalc() result (matrix or list of matrices) together with the warnings it raised
vres <- function(expr) {
  r <- warns(expr)
  list(V = mat(r$value), warnings = r$warnings)
}
sparse_res <- function(V) list(V = mat(V), is_sparse = inherits(V, "sparseMatrix"))
frame_cols <- function(d) lapply(as.list(d), function(col) if (is.numeric(col)) num(col) else as.character(col))

out <- list()

### dat.assink2016 ---------------------------------------------------------------------------

dat <- dat.assink2016
out$assink_rho06 <- vres(vcalc(vi, cluster = study, obs = esid, data = dat, rho = 0.6))
out$assink_rho0 <- vres(vcalc(vi, cluster = study, obs = esid, data = dat, rho = 0))
out$assink_cluster_only <- vres(vcalc(vi, cluster = study, data = dat))
out$assink_type_obs <- vres(vcalc(vi, cluster = study, type = deltype, obs = esid, data = dat, rho = c(0.7, 0.5)))
out$assink_type_obs_list <- vres(vcalc(vi, cluster = study, type = deltype, obs = esid, data = dat, rho = list(0.7, 0.5)))
out$assink_type_only <- vres(vcalc(vi, cluster = study, type = deltype, data = dat, rho = 0.3))
out$assink_sparse <- sparse_res(vcalc(vi, cluster = study, obs = esid, data = dat, rho = 0.6, sparse = TRUE))
out$assink_vi1 <- vres(vcalc(1, cluster = study, obs = esid, data = dat, rho = 0.6))
out$assink_vectors <- vres(vcalc(dat$vi, cluster = dat$study, obs = dat$esid, rho = 0.6))
out$assink_subset_cols <- vres(vcalc(vi, cluster = study, subgroup = pubstatus, obs = esid, data = dat, rho = 0.6))
out$assink_nopdcheck <- vres(vcalc(vi, cluster = study, obs = esid, data = dat, rho = 0.6, checkpd = FALSE))
rd <- vcalc(vi, cluster = study, obs = esid, data = dat[dat$study %in% 1:3, ], rho = 0.6, retdat = TRUE)
class(rd) <- "data.frame" # vcalc() leaves the class of the matrix on the data frame
out$assink_retdat <- list(cols = frame_cols(rd[1:10]), V = mat(rd[11:ncol(rd)]), names = names(rd)[1:10], vnames = names(rd)[11:ncol(rd)])
out$assink_blsplit <- lapply(blsplit(out_V <- vcalc(vi, cluster = study, obs = esid, data = dat, rho = 0.6), dat$study, round, 4)[1:3], mat)
out$assink_blsplit_cov2cor <- lapply(blsplit(out_V, dat$study, cov2cor)[16], mat)
out$assink_blsplit_sort <- lapply(blsplit(out_V, dat$study, round, 3, sort = TRUE)[c(1, 2, 3)], mat)
out$assink_blsplit_names <- names(blsplit(out_V, dat$study))
out$assink_blsplit_names_sorted <- names(blsplit(out_V, dat$study, sort = TRUE))

# missing and infinite sampling variances: dense results are NA across whole rows and columns, sparse ones are not
syn_na <- data.frame(s = c(1, 1, 2, 2, 1), vi = c(0.1, NA, 0.3, 0.4, 0.5), o = 1:5)
out$syn_na <- frame_cols(syn_na)
out$vi_na <- vres(vcalc(vi, cluster = s, obs = o, rho = 0.5, data = syn_na))
out$vi_na_sparse <- sparse_res(vcalc(vi, cluster = s, obs = o, rho = 0.5, data = syn_na, sparse = TRUE))
syn_inf <- syn_na
syn_inf$vi[2] <- Inf
out$vi_inf <- vres(vcalc(vi, cluster = s, obs = o, rho = 0.5, data = syn_inf))
out$vi_neg <- vres(vcalc(vi, cluster = s, obs = o, rho = 0.5, data = within(syn_na, vi[2] <- -0.2)))

### dat.konstantopoulos2011 ----------------------------------------------------------------

dk <- dat.konstantopoulos2011
out$konst_obs <- vres(vcalc(vi, cluster = district, obs = school, data = dk, rho = 0.4))
out$konst_obs_nearpd <- vres(vcalc(vi, cluster = district, obs = school, data = dk, rho = 0.9, nearpd = TRUE))

### dat.berkey1998 -------------------------------------------------------------------------

db <- dat.berkey1998
out$berkey_rvars <- vres(vcalc(vi = 1, cluster = author, rvars = c(v1i, v2i), data = db))
out$berkey_rvars_colon <- vres(vcalc(vi = 1, cluster = author, rvars = v1i:v2i, data = db))
out$berkey_type <- vres(vcalc(vi = vi, cluster = trial, type = outcome, data = db, rho = 0.4))
out$berkey_blsplit_fun <- lapply(blsplit(out$berkey_rvars$V, db$author, function(x) round(cov2cor(x), 2)), mat)
out$berkey_rvars_ignored <- vres(vcalc(vi = 1, cluster = author, rvars = c(v1i, v2i), data = db, rho = 0.3, phi = 0.5))

### dat.knapp2017 --------------------------------------------------------------------------

dn <- dat.knapp2017
dn$task.diff <- unlist(lapply(split(dn, dn$study), function(x) {
  task.int <- as.integer(factor(x$task))
  diff.int <- as.integer(factor(x$difficulty))
  diff.int[is.na(diff.int)] <- 1
  paste0(task.int, ".", diff.int)
}))
R <- matrix(0.4, nrow = 8, ncol = 8)
R[5:8, 1:4] <- R[1:4, 5:8] <- 0.28
diag(R[1:4, 5:8]) <- 0.7
diag(R[5:8, 1:4]) <- 0.7
diag(R) <- 1
rownames(R) <- colnames(R) <- paste0(rep(1:2, each = 4), ".", 1:4)
out$knapp_task_diff <- dn$task.diff
out$knapp_rho <- mat(R)
out$knapp_rho_names <- rownames(R)
Vk <- warns(vcalc(vi, cluster = study, grp1 = group1, grp2 = group2, w1 = n_sz, w2 = n_hc, obs = task.diff, rho = R, data = dn))
out$knapp <- list(V = mat(Vk$value), warnings = Vk$warnings)
out$knapp_cov2cor_study3 <- mat(cov2cor(blsplit(Vk$value, dn$study)[[3]]))
out$knapp_grp1_only <- vres(vcalc(vi, cluster = study, grp1 = group1, w1 = n_sz, data = dn))

### synthetic data -------------------------------------------------------------------------

# time1/time2/phi: three studies with irregular time points
syn_time <- data.frame(
  study = c(1, 1, 1, 2, 2, 3, 3, 3, 3), time = c(1, 2, 4, 1, 3, 2, 3, 5, 6),
  vi = c(0.10, 0.12, 0.08, 0.20, 0.15, 0.30, 0.25, 0.28, 0.22)
)
out$syn_time <- frame_cols(syn_time)
out$time_phi <- vres(vcalc(vi, cluster = study, time1 = time, phi = 0.7, data = syn_time))
out$time_phi_neg <- vres(vcalc(vi, cluster = study, time1 = time, phi = -0.5, data = syn_time))
out$time_phi_bad <- err(vcalc(vi, cluster = study, time1 = time, phi = 1.5, data = syn_time))

# two outcomes with time points: type + obs + time
syn_to <- data.frame(
  study = c(1, 1, 1, 1, 2, 2, 2, 2), type = c("a", "a", "b", "b", "a", "a", "b", "b"),
  obs = c(1, 2, 1, 2, 1, 2, 1, 2), time = c(1, 2, 1, 2, 1, 3, 1, 3),
  vi = c(0.10, 0.12, 0.08, 0.09, 0.20, 0.15, 0.18, 0.17)
)
out$syn_to <- frame_cols(syn_to)
out$type_obs_time <- vres(vcalc(vi, cluster = study, type = type, obs = obs, time1 = time, phi = 0.8, data = syn_to, rho = c(0.6, 0.3)))
rho_obs <- matrix(c(1, 0.5, 0.5, 1), 2, 2, dimnames = list(1:2, 1:2))
rho_type <- matrix(c(1, 0.2, 0.2, 1), 2, 2, dimnames = list(c("a", "b"), c("a", "b")))
out$rho_obs <- mat(rho_obs)
out$rho_type <- mat(rho_type)
out$type_obs_matrices <- vres(vcalc(vi, cluster = study, type = type, obs = obs, data = syn_to, rho = list(rho_obs, rho_type)))
out$type_only_matrix <- vres(vcalc(vi, cluster = study, type = type, data = syn_to, rho = rho_type))
out$obs_only_matrix <- vres(vcalc(vi, cluster = study, obs = obs, data = syn_to, rho = rho_obs))
out$obs_only_list <- vres(vcalc(vi, cluster = study, obs = obs, data = syn_to, rho = list(0.45)))

# grp1/grp2/w1/w2 (contrasts between groups that share a reference group), with and without time
syn_grp <- data.frame(
  study = c(1, 1, 1, 2, 2, 2, 2), g1 = c("t1", "t2", "t1", "a", "b", "a", "c"), g2 = c("ctl", "ctl", "t2", "z", "z", "b", "z"),
  n1 = c(20, 25, 20, 30, 35, 30, 40), n2 = c(30, 30, 25, 50, 50, 35, 50),
  t1 = c(1, 1, 1, 1, 1, 2, 2), t2 = c(1, 1, 1, 1, 1, 2, 2),
  vi = c(0.10, 0.12, 0.08, 0.20, 0.15, 0.18, 0.17)
)
out$syn_grp <- frame_cols(syn_grp)
out$grp_w <- vres(vcalc(vi, cluster = study, grp1 = g1, grp2 = g2, w1 = n1, w2 = n2, data = syn_grp))
out$grp_w_same <- vres(vcalc(vi, cluster = study, grp1 = g1, grp2 = g2, w1 = n1, data = syn_grp))
out$grp_w_time <- vres(vcalc(vi, cluster = study, grp1 = g1, grp2 = g2, w1 = n1, w2 = n2, time1 = t1, time2 = t2, phi = 0.6, data = syn_grp))
out$grp_w_time_obs <- vres(vcalc(vi, cluster = study, grp1 = g1, grp2 = g2, w1 = n1, w2 = n2, time1 = t1, time2 = t2, phi = 0.6, obs = t1, rho = 0.5, data = syn_grp))
out$grp_time1_only <- vres(vcalc(vi, cluster = study, grp1 = g1, time1 = t1, phi = 0.6, data = syn_grp))
out$grp_w_sparse <- sparse_res(vcalc(vi, cluster = study, grp1 = g1, grp2 = g2, w1 = n1, w2 = n2, data = syn_grp, sparse = TRUE))

# subgroup: the same study label in two subgroups form separate clusters
syn_sub <- data.frame(
  study = c(1, 1, 1, 1, 2, 2, 2), sub = c("x", "x", "y", "y", "x", "y", "y"), obs = c(1, 2, 1, 2, 1, 1, 2),
  vi = c(0.10, 0.12, 0.08, 0.09, 0.20, 0.15, 0.18)
)
out$syn_sub <- frame_cols(syn_sub)
out$subgroup <- vres(vcalc(vi, cluster = study, subgroup = sub, obs = obs, data = syn_sub, rho = 0.5))
sd <- vcalc(vi, cluster = study, subgroup = sub, obs = obs, data = syn_sub, rho = 0.5, retdat = TRUE)
class(sd) <- "data.frame"
out$subgroup_retdat <- list(cols = frame_cols(sd[1:10]), V = mat(sd[11:ncol(sd)]), vnames = names(sd)[11:ncol(sd)])
out$subgroup_sparse <- sparse_res(vcalc(vi, cluster = study, subgroup = sub, obs = obs, data = syn_sub, rho = 0.5, sparse = TRUE))

# non-positive definite correlation matrices
syn_pd <- data.frame(study = c(1, 1, 1, 2, 2, 2, 2), obs = c(1, 2, 3, 1, 2, 3, 4), vi = c(0.10, 0.12, 0.08, 0.20, 0.15, 0.18, 0.17))
rho_bad <- matrix(c(1, 0.9, -0.9, 0.9, 1, 0.9, -0.9, 0.9, 1), 3, 3, dimnames = list(1:3, 1:3))
rho_bad4 <- matrix(c(1, 0.9, -0.9, 0.2, 0.9, 1, 0.9, 0.5, -0.9, 0.9, 1, -0.3, 0.2, 0.5, -0.3, 1), 4, 4, dimnames = list(1:4, 1:4))
out$syn_pd <- frame_cols(syn_pd)
out$rho_bad <- mat(rho_bad)
out$rho_bad4 <- mat(rho_bad4)
out$pd_warn <- vres(vcalc(vi, cluster = study, obs = obs, data = syn_pd, rho = rho_bad4))
out$pd_nocheck <- vres(vcalc(vi, cluster = study, obs = obs, data = syn_pd, rho = rho_bad4, checkpd = FALSE))
out$pd_nearpd <- vres(vcalc(vi, cluster = study, obs = obs, data = syn_pd, rho = rho_bad4, nearpd = TRUE))
out$pd_nearpd_nocheck <- vres(vcalc(vi, cluster = study, obs = obs, data = syn_pd, rho = rho_bad4, nearpd = TRUE, checkpd = FALSE))
out$pd_one_cluster <- vres(vcalc(vi, cluster = study, obs = obs, data = syn_pd[1:3, ], rho = rho_bad, nearpd = TRUE))
np <- Matrix::nearPD(rho_bad4, corr = TRUE)
out$nearpd_direct <- list(
  mat = mat(np$mat), eigenvalues = num(np$eigenvalues), normF = np$normF, iterations = np$iterations,
  rel_tol = np$rel.tol, converged = np$converged
)
np2 <- Matrix::nearPD(matrix(c(2, -1, 0, -1, 2, -1.5, 0, -1.5, 1), 3, 3), keepDiag = TRUE)
out$nearpd_direct_keepdiag <- list(mat = mat(np2$mat), eigenvalues = num(np2$eigenvalues), iterations = np2$iterations, converged = np2$converged)
np3 <- Matrix::nearPD(matrix(c(1, 0.8, 0.3, 0.7, 1, 0.9, 0.3, 0.95, 1), 3, 3), corr = TRUE) # asymmetric input
out$nearpd_direct_asym <- list(mat = mat(np3$mat), eigenvalues = num(np3$eigenvalues), iterations = np3$iterations, converged = np3$converged)

### errors and warnings of vcalc() ---------------------------------------------------------

d <- syn_to
out$errors_vcalc <- list(
  no_vi = err(vcalc(cluster = study, data = d)),
  no_cluster = err(vcalc(vi, data = d)),
  rvars_no_data = err(vcalc(vi = 1, cluster = study, rvars = c(vi, vi))),
  k1 = err(vcalc(vi = 0.1, cluster = 1)),
  na_cluster = err(vcalc(vi, cluster = ifelse(study == 1, NA, study), data = d)),
  cluster_length = err(vcalc(vi, cluster = c(1, 2, 3), data = d)),
  na_subgroup = err(vcalc(vi, cluster = study, subgroup = ifelse(obs == 1, NA, obs), data = d)),
  subgroup_length = err(vcalc(vi, cluster = study, subgroup = c(1, 2), data = d)),
  type_no_rho = err(vcalc(vi, cluster = study, type = type, data = d)),
  na_type = err(vcalc(vi, cluster = study, type = ifelse(obs == 1, NA, type), data = d, rho = 0.5)),
  type_length = err(vcalc(vi, cluster = study, type = c("a", "b"), data = d, rho = 0.5)),
  obs_no_rho = err(vcalc(vi, cluster = study, obs = obs, data = d)),
  na_obs = err(vcalc(vi, cluster = study, obs = ifelse(obs == 1, NA, obs), data = d, rho = 0.5)),
  obs_length = err(vcalc(vi, cluster = study, obs = c(1, 2), data = d, rho = 0.5)),
  grp2_only = err(vcalc(vi, cluster = study, grp2 = type, data = d)),
  na_grp1 = err(vcalc(vi, cluster = study, grp1 = ifelse(obs == 1, NA, type), data = d)),
  na_grp2 = err(vcalc(vi, cluster = study, grp1 = type, grp2 = ifelse(obs == 1, NA, type), data = d)),
  grp1_length = err(vcalc(vi, cluster = study, grp1 = c(1, 2), data = d)),
  grp2_length = err(vcalc(vi, cluster = study, grp1 = type, grp2 = c(1, 2), data = d)),
  time2_only = err(vcalc(vi, cluster = study, time2 = time, phi = 0.5, data = d)),
  time2_no_grp2 = err(vcalc(vi, cluster = study, time1 = time, time2 = time, phi = 0.5, data = d)),
  time_no_phi = err(vcalc(vi, cluster = study, time1 = time, data = d)),
  phi_range = err(vcalc(vi, cluster = study, time1 = time, phi = -1.2, data = d)),
  na_time1 = err(vcalc(vi, cluster = study, time1 = ifelse(obs == 1, NA, time), phi = 0.5, data = d)),
  na_time2 = err(vcalc(vi, cluster = study, grp1 = type, grp2 = type, time1 = time, time2 = ifelse(obs == 1, NA, time), phi = 0.5, data = d)),
  time1_length = err(vcalc(vi, cluster = study, time1 = c(1, 2), phi = 0.5, data = d)),
  time1_not_numeric = err(vcalc(vi, cluster = study, time1 = type, phi = 0.5, data = d)),
  w2_only = err(vcalc(vi, cluster = study, w2 = time, data = d)),
  w2_no_grp2 = err(vcalc(vi, cluster = study, w1 = time, w2 = time, data = d)),
  na_w1 = err(vcalc(vi, cluster = study, w1 = ifelse(obs == 1, NA, time), data = d)),
  na_w2 = err(vcalc(vi, cluster = study, grp1 = type, grp2 = type, w1 = time, w2 = ifelse(obs == 1, NA, time), data = d)),
  w1_length = err(vcalc(vi, cluster = study, w1 = c(1, 2), data = d)),
  w2_length = err(vcalc(vi, cluster = study, grp1 = type, grp2 = type, w1 = time, w2 = c(1, 2), data = d)),
  w1_not_numeric = err(vcalc(vi, cluster = study, w1 = type, data = d)),
  both_vector_length = err(vcalc(vi, cluster = study, type = type, obs = obs, data = d, rho = 0.5)),
  both_matrix = err(vcalc(vi, cluster = study, type = type, obs = obs, data = d, rho = rho_obs)),
  both_list_length = err(vcalc(vi, cluster = study, type = type, obs = obs, data = d, rho = list(0.5))),
  type_vector_length = err(vcalc(vi, cluster = study, type = type, data = d, rho = c(0.5, 0.3))),
  type_list_length = err(vcalc(vi, cluster = study, type = type, data = d, rho = list(0.5, 0.3))),
  obs_vector_length = err(vcalc(vi, cluster = study, obs = obs, data = d, rho = c(0.5, 0.3))),
  obs_list_length = err(vcalc(vi, cluster = study, obs = obs, data = d, rho = list(0.5, 0.3))),
  rho_not_square = err(vcalc(vi, cluster = study, obs = obs, data = d, rho = matrix(0.5, 2, 3, dimnames = list(1:2, 1:3)))),
  rho_range = err(vcalc(vi, cluster = study, obs = obs, data = d, rho = 1.2)),
  rho_matrix_range = err(vcalc(vi, cluster = study, obs = obs, data = d, rho = matrix(c(1, 2, 2, 1), 2, 2, dimnames = list(1:2, 1:2)))),
  rho_no_dimnames = err(vcalc(vi, cluster = study, obs = obs, data = d, rho = matrix(c(1, 0.5, 0.5, 1), 2, 2))),
  rho_obs_missing = err(vcalc(vi, cluster = study, obs = obs, data = d, rho = matrix(c(1, 0.5, 0.5, 1), 2, 2, dimnames = list(c(1, 3), c(1, 3))))),
  rho_type_missing = err(vcalc(vi, cluster = study, type = type, data = d, rho = matrix(c(1, 0.5, 0.5, 1), 2, 2, dimnames = list(c("a", "c"), c("a", "c"))))),
  rvars_not_found = err(vcalc(vi = 1, cluster = study, rvars = c(nope1, nope2), data = d)),
  rvars_too_few = err(vcalc(vi = 1, cluster = study, rvars = c(vi), data = d)),
  rvars_too_many = err(vcalc(vi = 1, cluster = study, rvars = c(study, type, obs, time, vi), data = d)),
  rvars_too_few_one = err(vcalc(vi = 1, cluster = study, rvars = vi, data = d)),
  missing_column = err(vcalc(vi, cluster = nope, data = d))
)
out$warnings_vcalc <- list(
  rvars_missing = warns(vcalc(vi = 1, cluster = author, rvars = c(v1i, v2i), data = within(db, v1i[2] <- NA)))$warnings,
  rvars_missing_lower = warns(vcalc(vi = 1, cluster = author, rvars = c(v1i, v2i), data = within(db, v2i[1] <- NA)))$warnings,
  rvars_ignored_one = warns(vcalc(vi = 1, cluster = author, rvars = c(v1i, v2i), data = db, rho = 0.3))$warnings
)

### rcalc() --------------------------------------------------------------------------------

dc <- dat.craft2003
rc <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dc)
out$craft_dat <- list(cols = frame_cols(rc$dat), names = names(rc$dat))
out$craft_V <- mat(rc$V)
out$craft_V_dimnames <- rownames(rc$V)
Vs <- blsplit(rc$V, rc$dat$study, round, 4)
out$craft_blsplit <- lapply(Vs[c("1", "6", "17")], mat)
out$craft_blsplit_names <- names(Vs)
rcz <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dc, rtoz = TRUE)
out$craft_rtoz <- list(cols = frame_cols(rcz$dat), V = mat(rcz$V))
rch <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dc, nfun = "harmonic")
out$craft_harmonic <- list(cols = frame_cols(rch$dat), V = mat(rch$V))
rcm <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dc, nfun = "mean")
out$craft_mean <- list(cols = frame_cols(rcm$dat), V = mat(rcm$V))
rcf <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dc, nfun = function(x) max(x) - 1)
out$craft_nfun_function <- list(cols = frame_cols(rcf$dat), V = mat(rcf$V))
rcs <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dc, sparse = TRUE)
out$craft_sparse <- list(V = mat(rcs$V), is_sparse = inherits(rcs$V, "sparseMatrix"), cols = frame_cols(rcs$dat))
# rows in a shuffled order: dat and V come back in the order of the rows of the data
ord <- c(31:60, 1:30)
rco <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dc[ord, ])
out$craft_shuffled <- list(cols = frame_cols(rco$dat), V = mat(rco$V), order = ord)
# a single study without a study variable
rcn <- rcalc(ri ~ var1 + var2, ni = ni, data = dc[dc$study == 1, ])
out$craft_noid <- list(cols = frame_cols(rcn$dat), names = names(rcn$dat), V = mat(rcn$V))
# subset of the pairs of a study: the missing pairs stay in V as NA but the output has the rows of data only
dcs <- dc[-c(2, 4, 61), ]
rcp <- suppressWarnings(rcalc(ri ~ var1 + var2 | study, ni = ni, data = dcs))
out$craft_partial <- list(cols = frame_cols(rcp$dat), V = mat(rcp$V))
dna <- dc
dna$ri[5] <- NA
dna$ni[3] <- NA
rna <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dna)
out$craft_na <- list(cols = frame_cols(rna$dat), V = mat(rna$V))
dnames <- dc
dnames$study <- paste0("s", dnames$study)
rcnm <- rcalc(ri ~ var1 + var2 | study, ni = ni, data = dnames)
out$craft_string_ids <- list(cols = frame_cols(rcnm$dat))
rcfn <- rcalc(ri ~ var1 + var2 | study, ni = dc$ni, data = dc)
out$craft_ni_vector <- mat(rcfn$V)

R1 <- diag(4)
R1[lower.tri(R1)] <- dc$ri[dc$study == 1]
R1[upper.tri(R1)] <- t(R1)[upper.tri(R1)]
rownames(R1) <- colnames(R1) <- c("perf", "acog", "asom", "conf")
R3 <- diag(4)
R3[lower.tri(R3)] <- dc$ri[dc$study == 3]
R3[upper.tri(R3)] <- t(R3)[upper.tri(R3)]
rownames(R3) <- colnames(R3) <- c("perf", "acog", "asom", "conf")
out$R1 <- mat(R1)
out$R3 <- mat(R3)
r1 <- rcalc(R1, ni = 142)
out$rcalc_matrix <- list(cols = frame_cols(r1$dat), V = mat(r1$V), names = rownames(r1$V))
r1z <- rcalc(R1, ni = 142, rtoz = TRUE)
out$rcalc_matrix_rtoz <- list(cols = frame_cols(r1z$dat), V = mat(r1z$V))
r1u <- rcalc(R1, ni = 142, upper = TRUE)
out$rcalc_matrix_upper <- list(cols = frame_cols(r1u$dat), V = mat(r1u$V))
Rasym <- R1
Rasym[upper.tri(Rasym)] <- 0.1
Rasym[lower.tri(Rasym)] <- c(0.3, 0.2, 0.1, 0.25, 0.15, 0.05)
out$Rasym <- mat(Rasym)
rasym_l <- rcalc(Rasym, ni = 50)
rasym_u <- rcalc(Rasym, ni = 50, upper = TRUE)
out$rcalc_asym <- list(lower = list(cols = frame_cols(rasym_l$dat), V = mat(rasym_l$V)), upper = list(cols = frame_cols(rasym_u$dat), V = mat(rasym_u$V)))
rsmall <- rcalc(R1, ni = 4)
out$rcalc_small_n <- list(cols = frame_cols(rsmall$dat), V = mat(rsmall$V))
r2 <- rcalc(list("1" = R1, "3" = R3), ni = c(142, 37))
out$rcalc_list <- list(cols = frame_cols(r2$dat), V = mat(r2$V), names = rownames(r2$V))
r2s <- rcalc(list("1" = R1, "3" = R3), ni = c(142, 37), sparse = TRUE)
out$rcalc_list_sparse <- list(V = mat(r2s$V), is_sparse = inherits(r2s$V, "sparseMatrix"))
r2u <- rcalc(list(R1, R3), ni = c(142, 37))
out$rcalc_list_unnamed <- list(cols = frame_cols(r2u$dat))
r2n <- rcalc(list("1" = R1, "3" = R3), ni = c(142, 37), simplify = FALSE)
out$rcalc_list_nosimplify <- lapply(r2n, function(z) list(cols = frame_cols(z$dat), V = mat(z$V)))
r2one <- rcalc(list("a" = R1, "b" = matrix(1, 1, 1), "c" = R3), ni = c(142, 10, 37))
out$rcalc_list_with_1x1 <- list(cols = frame_cols(r2one$dat), V = mat(r2one$V))
out$rcalc_1x1 <- list(dat_null = is.null(rcalc(matrix(1, 1, 1), ni = 10)$dat), v_null = is.null(rcalc(matrix(1, 1, 1), ni = 10)$V))
R2 <- matrix(c(1, 0.4, 0.4, 1), 2, 2)
r_nonames <- rcalc(R2, ni = 30)
out$rcalc_nonames <- list(cols = frame_cols(r_nonames$dat), V = mat(r_nonames$V))
R4 <- matrix(c(1, 0.3, 0.2, 0.3, 1, 0.4, 0.2, 0.4, 1), 3, 3)
colnames(R4) <- c("b", "a", "c")
r_cn <- rcalc(R4, ni = 40)
out$rcalc_colnames_unsorted <- list(cols = frame_cols(r_cn$dat), V = mat(r_cn$V))
out$R4 <- mat(R4)

out$errors_rcalc <- list(
  bad_x = err(rcalc(1:3, ni = 10)),
  no_ni = err(rcalc(R1)),
  no_data = err(rcalc(ri ~ var1 + var2 | study, ni = ni)),
  bad_nfun = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = dc, nfun = "median")),
  four_vars = err(rcalc(ri ~ var1 + var2 + sport | study, ni = ni, data = dc)),
  two_vars = err(rcalc(ri ~ var1, ni = ni, data = dc)),
  na_var = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = within(dc, var1[2] <- NA))),
  ni_length = err(rcalc(ri ~ var1 + var2 | study, ni = 1:3, data = dc)),
  na_id = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = within(dc, study[2] <- NA))),
  ni_zero = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = within(dc, ni[2] <- 0))),
  nfun_not_single = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = dc, nfun = function(x) c(1, 2))),
  identical_pair = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = within(dc, var2[2] <- var1[2]))),
  identical_pairs = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = within(dc, var2[c(2, 3)] <- var1[c(2, 3)]))),
  duplicated_pair = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = within(dc, {var1[2] <- var1[1]; var2[2] <- var2[1]}))),
  duplicated_pairs = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = within(dc, {var1[c(2, 3)] <- var1[c(1, 1)]; var2[c(2, 3)] <- var2[c(1, 1)]}))),
  r_range = err(rcalc(ri ~ var1 + var2 | study, ni = ni, data = within(dc, ri[2] <- 1.5))),
  matrix_not_square = err(rcalc(matrix(0.5, 2, 3), ni = 10)),
  matrix_range = err(rcalc(matrix(c(1, 1.5, 1.5, 1), 2, 2), ni = 10)),
  matrix_ni = err(rcalc(R1, ni = 0)),
  list_ni_length = err(rcalc(list(R1, R3), ni = 10)),
  list_not_matrix = err(rcalc(list(R1, 1:3), ni = c(10, 10)))
)

### bldiag() and blsplit() -----------------------------------------------------------------

A <- matrix(1:4, 2, 2)
B <- matrix(c(5, 6, 6, 7), 2, 2)
Cm <- matrix(1:6, 2, 3)
out$bl_A <- mat(A)
out$bl_B <- mat(B)
out$bl_C <- mat(Cm)
out$bldiag_two <- mat(bldiag(A, B))
out$bldiag_list <- mat(bldiag(list(A, B, A)))
out$bldiag_scalar <- mat(bldiag(matrix(1, 3, 3), 2))
out$bldiag_vector <- mat(bldiag(c(1, 2, 3), A))
out$bldiag_rect <- mat(bldiag(A, Cm))
out$bldiag_empty <- mat(bldiag(matrix(nrow = 0, ncol = 0), A))
out$bldiag_all_empty <- dim(bldiag(matrix(nrow = 0, ncol = 0), matrix(nrow = 0, ncol = 3)))
out$bldiag_single <- mat(bldiag(A))
out$bldiag_sparse_input <- mat(bldiag(Matrix::Matrix(A, sparse = TRUE), B))
cl <- c(2, 1, 2, 1)
out$bldiag_order <- list(cluster = cl, V = mat(bldiag(A, B, order = cl)))
out$bldiag_order_perm <- mat(bldiag(A, B, order = order(cl)))
out$bldiag_order_str <- mat(bldiag(A, B, order = c("b", "a", "b", "a")))
out$errors_bldiag <- list(
  order_not_square = err(bldiag(A, Cm, order = 1:5)),
  order_length = err(bldiag(A, B, order = 1:3))
)

V4 <- bldiag(matrix(c(2, 3, 3, 4), 2, 2), B)
V4[3, 3] <- 7
cl2 <- c("p", "p", "q", "q")
out$V4 <- mat(V4)
out$blsplit_plain <- lapply(blsplit(V4, cl2), mat)
out$blsplit_round <- lapply(blsplit(V4, cl2, round, 1), mat)
out$blsplit_fun <- lapply(blsplit(V4, cl2, function(x) x / 2), mat)
out$blsplit_det <- blsplit(V4, cl2, det)
out$blsplit_interleaved <- lapply(blsplit(V4, c(2, 1, 2, 1)), mat)
out$blsplit_sorted <- names(blsplit(V4, c(2, 1, 2, 1), sort = TRUE))
out$blsplit_numeric_names <- names(blsplit(V4, c(10, 3, 10, 3)))
V4sp <- as(Matrix::Matrix(V4, sparse = TRUE), "generalMatrix") # blsplit() only accepts a dgCMatrix
out$blsplit_sparse <- lapply(blsplit(V4sp, cl2), mat)
out$blsplit_sparse_class <- sapply(blsplit(V4sp, cl2), function(z) inherits(z, "sparseMatrix"))
out$errors_blsplit <- list(
  no_cluster = err(blsplit(V4)),
  not_matrix = err(blsplit(1:4, cl2)),
  not_symmetric = err(blsplit(Cm[, 1:2], cl2[1:2])),
  asymmetric = err(blsplit(matrix(1:16, 4, 4), cl2)),
  cluster_length = err(blsplit(V4, 1:3))
)

out$meta <- list(metafor = as.character(packageVersion("metafor")), metadat = as.character(packageVersion("metadat")), Matrix = as.character(packageVersion("Matrix")), R = R.version.string)
dir.create("src/tests/reference/phase4", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/phase4/vcalc.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/phase4/vcalc.json\n")
