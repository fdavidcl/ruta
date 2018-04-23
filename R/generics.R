#' @rdname train.ruta_autoencoder
#' @export
train <- function(learner, ...) {
  UseMethod("train")
}

as_loss <- function(x) UseMethod("as_loss")

#' Coercion to ruta_network
#'
#' Generic function to coerce objects into networks.
#' @param object Object to be converted into a network
#' @return A \code{"ruta_network"} construct
#' @export
as_network <- function(object) UseMethod("as_network")

to_keras <- function(x, ...) UseMethod("to_keras")

generate <- function(learner, ...) UseMethod("generate")
