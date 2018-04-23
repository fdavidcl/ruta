#' Sequential network constructor
#'
#' Constructor function for networks composed of several sequentially placed
#' layers. You shouldn't generally need to use this. Instead, consider
#' concatenating several layers with \code{\link{+.ruta_network}}.
#'
#' @param ... Zero or more objects of class \code{"ruta_layer"}
#' @return A construct with class \code{"ruta_network"}
#'
#' @examples
#' my_network <- new_network(
#'   new_layer("input", 784, "linear"),
#'   new_layer("dense",  32, "tanh"),
#'   new_layer("dense", 784, "sigmoid")
#' )
#'
#' # Instead, consider using
#' my_network <- input() + dense(32, "tanh") + output("sigmoid")
#' @import purrr
#' @export
new_network <- function(...) {
  args <- list(...)

  # type check
  stopifnot(
    every(args, ~ class(.) == ruta_layer)
  )

  structure(
    args,
    class = ruta_network,
    encoding = encoding_index(args)
  )
}

#' @rdname as_network
#' @export
as_network.ruta_network <- function(object) {
  object
}

#' @rdname as_network
#' @export
as_network.numeric <- function(object) {
  as_network(as.integer(object))
}

#' @rdname as_network
#' @examples
#' net <- as_network(c(784, 1000, 32))
#' @import purrr
#' @export
as_network.integer <- function(object) {
  object <- c(object, rev(object)[-1])
  hidden <- object %>% map(dense) %>% reduce(`+`)
  input() + hidden + output()
}

#' Add layers to a network/Join networks
#'
#' @rdname join-networks
#' @param e1 First network
#' @param e2 Second network
#' @return Network combination
#' @examples
#' network <- input() + dense(30) + output("sigmoid")
#' another <- c(input(), dense(30), dense(3), dense(30), output())
#' @export
"+.ruta_network" <- function(e1, e2) {
  c(e1, e2)
}

#' @rdname join-networks
#' @param ... networks or layers to be concatenated
#' @import purrr
#' @export
c.ruta_network <- function(...) {
  args <- list(...) %>% map(as_network) %>% flatten
  do.call(new_network, args)
}

#' Access subnetworks of a network
#'
#' @param net A \code{"ruta_network"} object
#' @param index An integer vector of indices of layers to be extracted
#' @return A \code{"ruta_network"} object containing the specified layers.
#' @examples
#' (input() + dense(30))[2]
#' long <- input() + dense(1000) + dense(100) + dense(1000) + output()
#' short <- long[c(1, 3, 5)]
#' @import purrr
#' @export
"[.ruta_network" <- function(net, index) {
  reduce(
    index,
    function(acc, nxt) acc + net[[nxt]],
    .init = new_network()
  )
}

#' Get the index of the encoding
#'
#' Calculates the index of the middle layer of an encoder-decoder network.
#' @param net A network of class \code{"ruta_network"}
#' @return Index of the middle layer
encoding_index <- function(net) {
  ceiling(length(net) / 2)
}

#' Inspect a neural network
#'
#' @param x A \code{"ruta_network"} object
#' @param ... Additional parameters, currently ignored
#' @return Invisibly returns the same object passed as parameter
#' @export
print.ruta_network <- function(x, ...) {
  cat("Ruta network. Structure:\n")
  ind <- " "

  for (layer in x) {
    cat(ind, gsub("ruta_layer_", "", class(layer)[1]))
    if (layer$units > 0) {
      cat("(", layer$units, " units)", sep = "")
    }
    cat(" -", layer$activation, "\n")
  }

  invisible(x)
}

#' Build a Keras network
#'
#' @param x A \code{"ruta_network"} object
#' @param input_shape The length of each input vector (number of input attributes)
#' @return A list of Keras Tensor objects with an attribute \code{"encoding"}
#' indicating the index of the encoding layer
#' @import purrr
#' @export
to_keras.ruta_network <- function(x, input_shape) {
  network <- NULL
  net_list <- list()

  x[[x %@% "encoding"]]$name = "encoded"

  for (layer in x) {
    network <- to_keras(layer, input_shape, model = network)
    net_list[[length(net_list) + 1]] <- network
  }

  keras::keras_model(inputs = net_list[[1]], outputs = net_list[[length(net_list)]])
}
