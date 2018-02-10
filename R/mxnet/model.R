#' Train a learner object.
#'
#' \code{train} creates a trained model for a given learner.
#'
#' This is a generic function, will find a specific method for each learner class.
#'
#' @param x Learner (possibly a \code{"rutaLearner"} object).
#' @param ... Parameters passed to the specific method.
#' @return A trained model.
#' @export
train <- function(x, ...)
  UseMethod("train")

#' Get outputs from any layer from a trained model and new data.
#'
#' @param model A \code{"rutaModel"} object.
#' @param task A \code{"rutaTask"} object.
#' @param layerInput An integer indicating the index of the layer in which data
#'   will be injected.
#' @param layerOutput An integer indicating the index of the layer to be
#'   obtained.
#' @param ... Custom parameters for internal prediction function.
#' @return A matrix containing layer outputs for each instance in the given task.
#' @export
ruta.layerOutputs <- function(model, task, layerInput = 1, layerOutput, ...) {
  UseMethod("ruta.layerOutputs")
}

#' Get weights from any layer from a trained model.
#'
#' @param model A \code{"rutaModel"} object.
#' @param layer An integer indicating the index of the layer for weights to be
#'   extracted from.
#' @return A matrix containing layer weights.
#' @export
ruta.getWeights <- function(model, layer) {
  UseMethod("ruta.getWeights")
}
