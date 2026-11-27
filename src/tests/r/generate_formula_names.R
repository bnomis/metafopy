# Reference column names and design matrices from model.matrix() for formulas that write interaction terms in several orders, with
# numeric, character and I() terms, so that metafopy's naming rule is checked against R rather than against names written by hand.
# Writes src/tests/reference/formula_names.json: for each formula, the column names and the matrix by column.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/generate_formula_names.R

suppressPackageStartupMessages({
  library(jsonlite)
})

options(useFancyQuotes = FALSE)

dat <- data.frame(
  x = c(1.5, 2.25, 3, 4.75, 5.5, 7),
  z = c(0.5, -1, 2, 0.25, -3, 1.75),
  year = c(1980, 1985, 1990, 1995, 2000, 2005),
  deltype = c("covert", "general", "overt", "covert", "general", "overt"),
  outcome = c("AL", "PD", "AL", "PD", "AL", "PD"),
  stringsAsFactors = FALSE
)

formulas <- c(
  "~ x",
  "~ factor(deltype) + year",
  "~ deltype + year",
  "~ I(x^2)",
  "~ x*outcome",
  "~ outcome*x",
  "~ x:outcome",
  "~ outcome:x",
  "~ 0 + outcome + outcome:x",
  "~ 0 + outcome + I(year - 1985):outcome",
  "~ outcome:I(year - 1985)",
  "~ x + z + z:x",
  "~ z:x + x + z",
  "~ x:z",
  "~ factor(deltype):x",
  "~ deltype + outcome + outcome:deltype",
  "~ x + z + outcome + outcome:z:x",
  "~ x - 1"
)

out <- list()
for (f in formulas) {
  mm <- model.matrix(as.formula(f), data = dat)
  out[[f]] <- list(
    names = colnames(mm),
    columns = lapply(seq_len(ncol(mm)), function(j) as.numeric(mm[, j]))
  )
}

out$meta <- list(R = R.version.string)
dir.create("src/tests/reference", showWarnings = FALSE, recursive = TRUE)
write_json(out, "src/tests/reference/formula_names.json", auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
cat("wrote src/tests/reference/formula_names.json\n")
