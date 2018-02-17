new_layer <- function(type, units, activation) {
  # length check
  stopifnot(
    is_scalar_vector(type),
    is_scalar_vector(units),
    is_scalar_vector(activation)
  )
  # type coercing
  type <- as.character(type)
  units <- as.integer(units)
  activation <- as.character(activation)

  structure(
    list(
      type = type,
      units = units,
      activation = activation,
      regularizers = list()
    ),
    class = ruta_layer
  )
}

# coercion to ruta_network
as.ruta_network.ruta_layer <- function(object) {
  new_network(object)
}

make_atomic_network <- function(type, units, activation) {
  as.ruta_network(
    new_layer(type, units, activation)
  )
}

#' Create a fully-connected layer
#'
#' @param units Number of units
#' @param activation Optional, string indicating activation function (linear by default)
#' @export
dense <- function(units, activation = "linear") {
  make_atomic_network("dense", units, activation = activation)
}

#' Create an input layer
#'
#' @export
input <- function() {
  make_atomic_network("input", -1, "linear")
}

#' Create an output layer
#'
#' @param activation Optional, string indicating activation function (linear by default)
#' @export
output <- function(activation = "linear") {
  make_atomic_network("output", -1, activation)
}
