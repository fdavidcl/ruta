makeNetwork <- function(...) {
  net <- if (nargs() == 1 && class(...) == rutaLayer)
    list(...)
  else if (all(sapply(list(...), function(x) class(x) == rutaNetwork)))
    c(...)
  else
    stop("Error: Not a valid network")

  class(net) <- rutaNetwork
  net
}

makeLayer <- function(type, units, activation) {
  stopifnot(
    is.numeric(units),
    length(units) == 1,
    is.character(activation),
    length(activation) == 1
  )
  layer <- list(type = type, units = as.integer(units), activation = activation)
  class(layer) <- rutaLayer
  makeNetwork(layer)
}

#' @export
"+.rutaNetwork" <- function(...) {
  makeNetwork(...)
}

#' @export
dense <- function(units, activation = "linear") {
  makeLayer("dense", units, activation = activation)
}

#' @export
input <- function() {
  makeLayer("input", -1, "linear")
}

#' @export
output <- function(activation = "linear") {
  makeLayer("output", -1, activation)
}

#' @export
print.rutaNetwork <- function(x, ...) {
  cat("Ruta network. Structure:\n")
  ind <- " "

  for (layer in x) {
    cat(ind, layer$type)
    if (!(layer$type %in% c("input", "output")))
      cat("(", layer$units, " units)", sep = "")
    cat(" -", layer$activation, "\n")
  }
}
