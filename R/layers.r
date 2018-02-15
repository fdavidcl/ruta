makeNetwork <- function(...) {
  args <- list(...)

  net <- if (every(args, ~ class(.) == rutaLayer)) {
    args
  } else if (every(args, ~ class(.) == rutaNetwork)) {
    args %>% map(unclass) %>% do.call(c, .)
  } else {
    stop("Error: Not a valid network")
  }

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
  layer <- structure(
    list(
      type = type,
      units = as.integer(units),
      activation = activation
    ),
    class = rutaLayer
  )
  makeNetwork(layer)
}

#' Add layers to a network/Join networks
#'
#' @rdname join-networks
#' @param e1 First network
#' @param e2 Second network
#' @return Network combination
#' @examples
#' network = input() + dense(30) + output("sigmoid")
#' another = c(input(), dense(30), dense(3), dense(30), output())
#' @export
"+.rutaNetwork" <- function(e1, e2) {
  if (class(e1) != class(e2)) {
    e1 <- makeNetwork(e1)
    e2 <- makeNetwork(e2)
  }

  makeNetwork(e1, e2)
}

#' @rdname join-networks
#' @param ... networks to be concatenated
#' @export
c.rutaNetwork <- function(...) {
  list(...) %>% reduce(`+`)
}

#' Access subnetworks of a network
#'
#' @param net A \code{"rutaNetwork"} object
#' @param index An integer vector of indices of layers to be extracted
#' @return A \code{"rutaNetwork"} object containing the specified layers.
#' @examples
#' (input() + dense(30))[2]
#' long = input() + dense(1000) + dense(100) + dense(1000) + output()
#' short = long[c(1, 3, 5)]
#' @export
"[.rutaNetwork" <- function(net, index) {
  reduce(index, function(acc, nxt) acc + net[[nxt]], .init = makeNetwork())
}

#' Create a fully-connected layer
#'
#' @param units Number of units
#' @param activation Optional, string indicating activation function (linear by default)
#' @export
dense <- function(units, activation = "linear") {
  makeLayer("dense", units, activation = activation)
}

#' Create an input layer
#'
#' @export
input <- function() {
  makeLayer("input", -1, "linear")
}

#' Create an output layer
#'
#' @param activation Optional, string indicating activation function (linear by default)
#' @export
output <- function(activation = "linear") {
  makeLayer("output", -1, activation)
}

#' Inspect a neural network
#'
#' @param x A \code{"rutaNetwork"} object
#' @param ... Additional parameters, currently ignored
#' @return Invisibly returns the same object passed as parameter
#' @export
print.rutaNetwork <- function(x, ...) {
  cat("Ruta network. Structure:\n")
  ind <- " "

  for (layer in x) {
    cat(ind, layer$type)
    if (!(layer$type %in% c("input", "output"))) {
      cat("(", layer$units, " units)", sep = "")
    }
    cat(" -", layer$activation, "\n")
  }

  invisible(x)
}

encodingIndex <- function(net) {
  ceiling(length(net) / 2)
}

#' Build a Keras network
#'
#' @param x A \code{"rutaNetwork"} object
#' @param input_shape The length of each input vector (number of input attributes)
#' @return A list of Keras Tensor objects with an attribute \code{"encoding"}
#' indicating the index of the encoding layer
#' @import purrr
#' @export
toKeras.rutaNetwork <- function(x, input_shape) {
  network <- NULL

  keras_lf <- list(
    input = keras::layer_input,
    dense = keras::layer_dense,
    output = keras::layer_dense
  )

  network <- keras_lf$input(shape = input_shape)
  net_list <- list()

  for (layer in x) {
    if (layer$units < 0) {
      layer$units <- input_shape
    }

    if (layer$type == "input") {
      network
    } else {
      act_reg <- if (!is.null(layer$activity_regularizer))
        toKeras(layer$activity_regularizer)
      else
        NULL

      network <- network %>% keras_lf[[layer$type]](
        units = layer$units,
        activation = layer$activation,
        activity_regularizer = act_reg
      )
    }

    net_list[[length(net_list) + 1]] <- network
  }

  net_list %>% structure(encoding = encodingIndex(x))
}
