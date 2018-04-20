#' Layer wrapper constructor
#'
#' Constructor function for layers. You shouldn't generally need to use this. Instead, consider
#' using individual functions such as \code{\link{dense}}.
#'
#' @param cl Character string specifying class of layer (e.g. \code{"ruta_layer_dense"}), which
#'   will be used to call the corresponding methods
#' @param units Amount of units in the layer
#' @param activation Character string specifying activation function
#' @return A construct with class \code{"ruta_layer"}
#'
#' @examples
#' my_layer <- new_layer("dense", 30, "tanh")
#'
#' # Equivalent:
#' my_layer <- dense(30, "tanh")[[1]]
#' @import purrr
#' @export
new_layer <- function(cl, units, activation) {
  # length check
  stopifnot(
    is_scalar_vector(cl),
    is_scalar_vector(units),
    is_scalar_vector(activation)
  )
  # type coercion
  type <- as.character(cl)
  units <- as.integer(units)
  activation <- as.character(activation)

  structure(
    list(
      units = units,
      activation = activation
    ),
    class = c(cl, ruta_layer)
  )
}

#' @rdname as_network
#' @export
as_network.ruta_layer <- function(object) {
  new_network(object)
}

to_keras.ruta_layer_input <- function(x, input_shape, ...) {
  keras::layer_input(shape = input_shape)
}
to_keras.ruta_layer_dense <- function(x, input_shape, model = keras::keras_model_sequential(), ...) {
  if (x$units < 0) {
    x$units <- input_shape
  }

  act_reg = if (!is.null(x$activity_regularizer))
    to_keras(x$activity_regularizer, activation = x$activation)
  else
    NULL

  kern_reg = if (!is.null(x$kernel_regularizer))
    to_keras(x$kernel_regularizer)
  else
    NULL

  keras::layer_dense(
    model,
    units = x$units,
    activation = x$activation,
    activity_regularizer = act_reg,
    kernel_regularizer = kern_reg,
    name = x$name,
    ...
  )
}

make_atomic_network <- function(type, units, activation) {
  as_network(
    new_layer(type, units, activation)
  )
}

#' Create a fully-connected neural layer
#'
#' Wrapper for a dense/fully-connected layer.
#' @param units Number of units
#' @param activation Optional, string indicating activation function (linear by default)
#' @return A construct with class \code{"ruta_layer"}
#' @examples
#' dense(30, "tanh")
#' @family neural layers
#' @export
dense <- function(units, activation = "linear") {
  make_atomic_network(ruta_layer_dense, units, activation = activation)
}

#' Create an input layer
#'
#' This layer acts as a placeholder for input data. The number of units is not
#' needed as it is deduced from the data during training.
#' @return A construct with class \code{"ruta_layer"}
#' @family neural layers
#' @export
input <- function() {
  make_atomic_network(ruta_layer_input, -1, "linear")
}

#' Create an output layer
#'
#' This layer acts as a placeholder for the output layer in an autoencoder. The
#' number of units is not needed as it is deduced from the data during training.
#' @param activation Optional, string indicating activation function (linear by default)
#' @return A construct with class \code{"ruta_layer"}
#' @family neural layers
#' @export
output <- function(activation = "linear") {
  make_atomic_network(ruta_layer_dense, -1, activation)
}
