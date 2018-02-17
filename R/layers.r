make_layer <- function(type, units, activation) {
  stopifnot(
    is.numeric(units),
    length(units) == 1,
    is.character(activation),
    length(activation) == 1
  )
  layer <- structure(
    list(
      type = type,
      units = as.integer(units),
      activation = activation
    ),
    class = ruta_layer
  )
  make_network(layer)
}

#' Create a fully-connected layer
#'
#' @param units Number of units
#' @param activation Optional, string indicating activation function (linear by default)
#' @export
dense <- function(units, activation = "linear") {
  make_layer("dense", units, activation = activation)
}

#' Create an input layer
#'
#' @export
input <- function() {
  make_layer("input", -1, "linear")
}

#' Create an output layer
#'
#' @param activation Optional, string indicating activation function (linear by default)
#' @export
output <- function(activation = "linear") {
  make_layer("output", -1, activation)
}
