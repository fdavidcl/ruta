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
