#' Weight decay
#'
#' @description A wrapper that describes a weight decay regularization of the
#' encoding layer
#'
#' @param decay Numeric value indicating the amount of decay
#'
#' @return A regularizer object containing the set parameters
weight_decay <- function(decay = 0.02) {
  structure(
    list(decay = decay),
    class = c(ruta_regularizer, ruta_weight_decay)
  )
}

#' Add weight decay to any autoencoder
#'
#' @description Adds a weight decay regularization to the encoding layer of a
#' given autoencoder
#'
#' @param learner The \code{"ruta_autoencoder"} object
#' @param decay Numeric value indicating the amount of decay
#'
#' @return An autoencoder object which contains the weight decay
#' @import purrr
#' @export
add_weight_decay <- function(learner, decay = 0.02) {
  # apply this regularizer only to the encoding?
  learner$network[[learner$network %@% "encoding"]]$kernel_regularizer <- weight_decay(decay)

  learner
}

#' Obtain a Keras weight decay
#'
#' @description Builds the Keras regularizer corresponding to the weight decay
#'
#' @param x A \code{"ruta_weight_decay"} object
#' @param ... Rest of parameters, ignored
#' @export
to_keras.ruta_weight_decay <- function(x, ...) {
  keras::regularizer_l2(l = x$decay)
}
