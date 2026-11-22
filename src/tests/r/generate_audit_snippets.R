# Records R's parse tree and value for each R snippet the tests of the audit harness run through the transpiler and the R-library shims
# (src/audit/transpile.py, src/audit/rlib.py), so the tests compare with R without R installed.
#
#   LC_ALL=C.UTF-8 R_LIBS=<pinned library> Rscript src/tests/r/generate_audit_snippets.R
#
# Writes src/tests/reference/audit/snippets.json. Stops when the installed R or metafor differs from src/audit/data/r-versions.json: a fixture
# from another build would shift what the tests expect.
script_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
root <- normalizePath(file.path(script_dir, "..", "..", ".."))
source(file.path(root, "src", "audit", "r", "common.R"))
suppressPackageStartupMessages(library(metafor))
suppressPackageStartupMessages(library(metadat))

pinned <- read_json(file.path(root, "src", "audit", "data", "r-versions.json"))
installed <- list(R = paste(R.version$major, R.version$minor, sep = "."), metafor = packageDescription("metafor")$Version)
for (nm in names(installed)) {
  if (!identical(installed[[nm]], pinned[[nm]])) {
    stop(sprintf("%s %s is installed but src/audit/data/r-versions.json pins %s", nm, installed[[nm]], pinned[[nm]]), call. = FALSE)
  }
}

# id = R text. Tasks 4 to 6 of agents/tasks/direct-tests-for-the-audit-harness/tasks-1.md add the cases each open candidate names.
snippets <- list(
  # -- rep, seq, head, tail
  rep_each = "rep(c(1, 2, 3), each = 2)",
  rep_times_vector = "rep(c(1, 2), times = c(3, 1))",
  rep_times_scalar = "rep(c(1, 2), times = 3)",
  rep_length_out = "rep(c(1, 2, 3), length.out = 7)",
  seq_by = "seq(1, 10, by = 3)",
  seq_length_out = "seq(0, 1, length.out = 5)",
  seq_from_to = "seq(2, 6)",
  seq_one_argument = "seq(5)",
  seq_along_vector = "seq_along(c(4, 5, 6))",
  head_default = "head(1:10)",
  head_negative = "head(1:10, -3)",
  head_data_frame = "head(dat.bcg, 3)",
  tail_default = "tail(1:10)",
  tail_negative = "tail(1:10, -3)",
  tail_data_frame = "tail(dat.bcg, 3)",
  # -- length, names, is.na, list
  length_vector = "length(c(1, 5, 9))",
  length_data_frame = "length(dat.bcg)",
  length_null = "length(NULL)",
  names_data_frame = "names(dat.bcg)",
  names_list = "names(list(a = 1, b = 2))",
  names_unnamed = "names(c(1, 2))",
  names_assigned = "x <- c(1, 2, 3); names(x) <- c('a', 'b', 'c'); x",
  names_assigned_data_frame = "d <- data.frame(p = 1:2, q = 3:4); names(d) <- c('u', 'v'); d",
  isna_vector = "is.na(c(1, NA, 3))",
  isna_data_frame = "is.na(data.frame(a = c(1, NA), b = c(NA, 2)))",
  list_named = "list(a = 1, b = 'x', c = c(1, 2))",
  list_unnamed = "list(1, 'a')",
  # -- matrices
  matrix_by_column = "matrix(1:6, nrow = 2)",
  matrix_by_row = "matrix(1:6, nrow = 2, byrow = TRUE)",
  matrix_ncol_only = "matrix(1:6, ncol = 3)",
  matrix_recycled = "matrix(0, 2, 2)",
  matrix_dimnames = "matrix(1:4, 2, dimnames = list(c('r1', 'r2'), c('c1', 'c2')))",
  cbind_vectors = "cbind(c(1, 2), c(3, 4))",
  cbind_named = "cbind(a = 1:2, b = 3:4)",
  cbind_matrix_and_vector = "cbind(matrix(1:4, 2), c(9, 9))",
  rbind_vectors = "rbind(c(1, 2), c(3, 4))",
  rbind_named = "rbind(a = 1:2, b = 3:4)",
  rbind_matrix_and_vector = "rbind(matrix(1:4, 2), c(9, 9))",
  diag_size = "diag(3)",
  diag_of_vector = "diag(c(1, 2, 3))",
  diag_of_matrix = "diag(matrix(1:9, 3))",
  as_matrix_data_frame = "as.matrix(data.frame(a = 1:2, b = 3:4))",
  as_matrix_vector = "as.matrix(1:3)",
  as_dataframe_matrix = "as.data.frame(matrix(1:4, 2))",
  as_dataframe_list = "as.data.frame(list(a = 1:2, b = c('x', 'y')))",
  dataframe_columns = "data.frame(a = 1:3, b = c('x', 'y', 'z'))",
  dataframe_recycled = "data.frame(a = 1:2, b = 3)",
  # -- statistics
  var_vector = "var(c(1, 2, 3, 4, 9))",
  var_matrix = "var(cbind(1:5, c(2, 1, 4, 3, 6)))",
  cor_vectors = "cor(1:5, c(2, 1, 4, 3, 6))",
  cor_matrix = "cor(cbind(1:5, c(2, 1, 4, 3, 6), c(5, 3, 4, 1, 2)))",
  # -- text, splitting, applying
  paste_default = "paste('a', 1:3)",
  paste_sep = "paste('a', 1:3, sep = '-')",
  paste_collapse = "paste(c('a', 'b'), collapse = '+')",
  paste_numbers = "paste(c(1.5, 2, 3.25))",
  split_vector = "split(dat.bcg$tpos, dat.bcg$alloc)",
  split_data_frame = "split(dat.bcg, dat.bcg$alloc)",
  lapply_named_list = "lapply(list(a = 1:3, b = 4:6), sum)",
  lapply_vector = "lapply(1:3, function(i) i^2)",
  # -- conversion and tests of type
  as_numeric_text = "as.numeric('3.5')",
  as_numeric_logical = "as.numeric(c(TRUE, FALSE))",
  as_numeric_texts = "as.numeric(c('1', '2'))",
  class_numeric = "class(1)",
  class_matrix = "class(matrix(1:4, 2))",
  class_data_frame = "class(dat.bcg)",
  inherits_data_frame = "inherits(dat.bcg, 'data.frame')",
  inherits_number = "inherits(1, 'numeric')",
  attr_set = "x <- 1:3; attr(x, 'foo') <- 'bar'; attr(x, 'foo')",
  attr_missing = "attr(1:3, 'foo')",
  # -- operators, conditions, checks
  equal_numbers = "c(1, 2, 3) == c(1, 5, 3)",
  equal_number_and_text = "c(1, 2) == c('1', 'b')",
  stopifnot_true = "stopifnot(TRUE, c(TRUE, TRUE)); 1",
  stopifnot_false = "stopifnot(c(TRUE, FALSE))",
  recycled_operands = "c(1, 2, 3, 4) + c(10, 20)",
  recycled_column = "dat.bcg$tpos[1:2] + 1",
  if_vector_condition = "if (c(TRUE, FALSE)[1]) 1 else 2",
  if_length_zero = "if (logical(0)) 1 else 2",
  capture_output_print = "capture.output(print(1:3))",
  # -- the transpiler: values and assignment targets
  assign_plain = "x <- 3; x + 1",
  assign_equals = "x = 4; x * 2",
  assign_arrow_right = "5 -> x; x",
  assign_nested_target = "d <- data.frame(a = 1:3); d$b <- d$a * 2; d",
  assign_element_of_column = "d <- data.frame(a = c(1, 2, 3)); d$a[2] <- 9; d",
  assign_index = "x <- c(1, 2, 3); x[2] <- 10; x",
  assign_index_matrix = "m <- matrix(0, 2, 2); m[1, 2] <- 5; m",
  assign_index2 = "l <- list(a = 1, b = 2); l[['b']] <- 7; l",
  assign_dollar_list = "l <- list(a = 1); l$z <- 'new'; l",
  assign_names_replacement = "x <- 1:3; names(x) <- c('a', 'b', 'c'); x",
  assign_attr_replacement = "x <- 1:3; attr(x, 'k') <- 'v'; attr(x, 'k')",
  assign_superassign = "x <- 1; f <- function() { x <<- 5 }; f(); x",
  assign_value_of_replacement = "x <- c(1, 2); (x[1] <- 8)",
  index_data_frame_rows = "dat.bcg[2:3, c('author', 'year')]",
  index_negative = "c(5, 6, 7, 8)[-1]",
  index_logical = "c(5, 6, 7, 8)[c(TRUE, FALSE, TRUE, FALSE)]",
  index_two_brackets = "list(a = 1:3, b = 'z')[['a']]",
  dollar_column = "dat.bcg$year[1:3]",
  # -- the transpiler: operators and control flow
  operator_arithmetic = "(2 + 3) * 4 - 6 / 3 + 2^3",
  operator_modulo = "c(7, 8, 9) %% 4",
  operator_integer_division = "c(7, 8, 9) %/% 4",
  operator_logic = "c(TRUE, FALSE) & c(TRUE, TRUE) | c(FALSE, FALSE)",
  operator_not = "!c(TRUE, FALSE)",
  operator_unary_minus = "-c(1, -2)",
  operator_comparison = "c(1, 2, 3) >= 2",
  operator_colon = "2:6",
  operator_in = "c(1, 5) %in% c(5, 6)",
  if_else_value = "x <- 3; if (x > 2) 'big' else 'small'",
  if_without_else = "x <- 1; y <- 0; if (x > 2) y <- 1; y",
  for_loop = "s <- 0; for (i in 1:4) s <- s + i; s",
  for_loop_block = "s <- c(); for (i in 1:3) { s <- c(s, i * 2) }; s",
  while_loop = "i <- 0; while (i < 3) i <- i + 1; i",
  function_no_defaults = "f <- function(a, b) a + b; f(1, 2)",
  function_with_defaults = "f <- function(a, b = 10) a + b; f(1)",
  function_named_call = "f <- function(a, b = 10) a - b; f(b = 1, a = 5)",
  function_closure = "make <- function(k) function(x) x * k; make(3)(4)",
  function_lazy_default = "f <- function(a, b = a * 2) b; f(4)",
  block_value = "{ 1; 2; 3 }",
  # -- the transpiler: calls into metafor
  call_nse_column = "res <- rma(yi, vi, data = escalc(measure = 'RR', ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)); round(res$beta, 4)",
  call_with = "with(dat.bcg, sum(tpos))",
  call_reserved_name = "T <- 3; F <- 2; T + F",
  call_formula_model = "res <- rma(yi ~ ablat, vi, data = escalc(measure = 'RR', ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)); round(coef(res), 4)",
  formula_one_side = "~ ablat",
  formula_two_sides = "yi ~ ablat + year",
  formula_interaction = "yi ~ ablat * alloc",
  formula_function_call = "yi ~ log(ablat) + I(year^2)",
  formula_no_intercept = "yi ~ 0 + ablat"
)

fresh_env <- function() {
  env <- new.env(parent = globalenv())
  utils::data("dat.bcg", package = "metadat", envir = env)
  env
}

record <- function(id, text) {
  exprs <- parse(text = text, keep.source = FALSE)
  env <- fresh_env()
  set.seed(1234)
  out <- list(id = id, r_text = text, units = lapply(as.list(exprs), ast))
  res <- tryCatch({
    value <- NULL
    # eval is safe here: the texts are the fixed list above, written in this file
    for (e in as.list(exprs)) value <- withVisible(suppressWarnings(eval(e, env)))$value
    list(value = value)
  }, error = function(e) list(error = conditionMessage(e)))
  if (!is.null(res$error)) {
    out$status <- "error"
    out$error <- res$error
    out$value <- list(k = "null")
  } else {
    out$status <- "ok"
    out$value <- node_of(res$value)
    if (inherits(res$value, "formula")) out$deparsed <- paste(deparse(res$value), collapse = " ")
  }
  out
}

records <- lapply(names(snippets), function(id) record(id, snippets[[id]]))
names(records) <- names(snippets)
doc <- list(meta = list(R = R.version.string, R_major = R.version$major, R_minor = R.version$minor, metafor = installed$metafor), snippets = records)
dir.create(file.path(root, "src", "tests", "reference", "audit"), showWarnings = FALSE, recursive = TRUE)
write_json(doc, file.path(root, "src", "tests", "reference", "audit", "snippets.json"))
