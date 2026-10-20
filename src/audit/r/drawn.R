# Notes the text the plotting functions draw (shared by runcases.R and runscripts.R): `drawn` holds the strings since it was last reset.
#
# The text a plotting function draws, so that labels (axis titles, legends, the heading of the columns of forest()) are compared too: the graphics
# functions metafor calls are replaced, in the environment its namespace imports them from, by ones that note the strings they are given.
drawn <- character()
note_text <- function(x) {
  x <- tryCatch(as.character(unlist(x)), error = function(e) character())
  x <- trimws(gsub("[[:space:]]+", " ", x))
  drawn <<- c(drawn, x[!is.na(x) & grepl("[A-Za-z]", x)])
}
spy <- function(name, defn, picks) {
  orig <- get(name, envir = asNamespace("graphics"))
  force(defn)
  force(picks)  # the loop below changes `spec` before the wrappers run
  function(...) {
    args <- list(...)
    named <- tryCatch(match.call(defn, as.call(c(list(as.name("f")), args))), error = function(e) NULL)
    if (!is.null(named)) for (p in picks) if (!is.null(named[[p]])) note_text(named[[p]])
    do.call(orig, args)
  }
}
# the places a call can find these functions: metafor's imports, stats' imports (qqnorm()), and graphics itself (plot.density() and hist() call title() and
# plot.default() from there)
homes <- list(parent.env(asNamespace("metafor")), parent.env(asNamespace("stats")), asNamespace("graphics"), as.environment("package:graphics"))
for (spec in list(
  list("text", graphics:::text.default, "labels"), list("mtext", graphics::mtext, "text"), list("axis", graphics::axis, "labels"),
  list("title", graphics::title, c("main", "sub", "xlab", "ylab")), list("plot", graphics:::plot.default, c("main", "sub", "xlab", "ylab")),
  list("plot.default", graphics:::plot.default, c("main", "sub", "xlab", "ylab")), list("legend", graphics::legend, "legend"))) {
  spied <- spy(spec[[1]], spec[[2]], spec[[3]])
  for (home in homes) {
    if (exists(spec[[1]], envir = home, inherits = FALSE)) {
      if (bindingIsLocked(spec[[1]], home)) unlockBinding(spec[[1]], home)
      assign(spec[[1]], spied, envir = home)
    }
  }
}
