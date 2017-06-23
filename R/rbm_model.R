#' Train a Restricted Boltzmann Machine.
#' @param x A \code{"rutaRBM"} object. This is the learner that will be
#'   trained.
#' @param task A \code{"rutaUnsupervisedTask"} object. It contains the data that
#'   will be used to train the RBM. Class information, if present, will
#'   not be used.
#' @param ... Additional parameters for the training function
#' @return A \code{"rutaModel"} object containing the trained model.
#' @import deepnet
#' @export
train.rutaRBM <- function(x, task, ...) {
  data <- taskToDeepnet(task)

  rbm <- rbm.train(data, hidden = x$parameters$hidden, ...)

  model <-
    list(
      internal = rbm,
      backend = x$backend,
      parameters = list(...),
      learner = x
    )
  class(model) <- c(rutaModel, rutaRBMModel)
  model
}

#' Get outputs from any layer from a trained model and new data.
#'
#' @param model A \code{"rutaRBMModel"} object trained with an RBM learner.
#' @param task A \code{"rutaTask"} object.
#' @param layerInput Just 1 or 2 (default = 1).
#' @param layerOutput The remaining layer (optional).
#' @param ... Custom parameters for internal prediction function.
#' @return A matrix containing layer outputs for each instance in the given task.
#' @import deepnet
#' @export
ruta.layerOutputs.rutaRBMModel <- function(model, task, layerInput = 1, layerOutput = 2 - layerInput, ...) {
  if (layerInput == 1) {
    rbm.up(model$internal, taskToDeepnet(task))
  } else {
    rbm.down(model$internal, taskToDeepnet(task))
  }
}

#' Get weights from any layer from a trained model.
#'
#' @param model A \code{"rutaRBMModel"} object.
#' @param layer An integer indicating the index of the layer for weights to be
#'   extracted from.
#' @return A matrix containing layer weights.
#' @export
ruta.getWeights.rutaRBMModel <- function(model, layer) {
  if (layer == 1) {
    model$internal$W
  } else {
    t(model$internal$W)
  }
}

taskToDeepnet <- function(task) {
  data = data.matrix(task$data)
  if (is.null(task$cl))
    data
  else
    data[,-task$cl]
}
