# Writes the versions of R and of the installed packages: Rscript versions.R <out.json>
args <- commandArgs(trailingOnly = TRUE)
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
source(file.path(here, "common.R"))
inst <- installed.packages(lib.loc = .libPaths())
vers <- as.list(setNames(inst[, "Version"], inst[, "Package"]))
vers <- vers[!(names(vers) %in% rownames(installed.packages(priority = "base")))]
vers$R <- paste(R.version$major, R.version$minor, sep = ".")
write_json(vers[order(names(vers))], args[1])
