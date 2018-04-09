#' Create a robust autoencoder
#'
#' @description A robust autoencoder uses a special objective function,
#' correntropy, a localized similarity measure which makes it less sensitive
#' to noise in data.
#'
#' @param network Layer construct of class \code{"ruta_network"}
#' @param sigma Sigma parameter in the kernel used for correntropy
#'
#' @return A construct of class \code{"ruta_autoencoder"}
#' @export
autoencoder_robust <- function(network, sigma) {
  autoencoder(network, correntropy(sigma))
}

#' Correntropy loss
#'
#' @description A wrapper for the correntropy loss function
#'
#' @param sigma Sigma parameter in the kernel
#'
#' @return A \code{"ruta_loss"} object
#' @export
correntropy <- function(sigma) {
  structure(
    list(sigma = sigma),
    class = c(ruta_loss, ruta_correntropy)
  )
}

#' Add robust behavior to any autoencoder
#'
#' @description Converts an autoencoder into a robust one by assigning a
#' correntropy loss to it. Notice that this will replace the previous loss
#' function
#'
#' @param learner The \code{"ruta_autoencoder"} object
#' @param sigma Sigma parameter in the kernel used for correntropy
#'
#' @return An autoencoder object which contains the correntropy loss
#' @export
make_robust <- function(learner, sigma) {
  # message("This will replace the previous loss function")
  learner$loss <- correntropy(sigma)
}

#' Obtain a Keras correntropy loss
#'
#' @description Builds the Keras loss function corresponding to the object received
#'
#' @param x A \code{"ruta_correntropy"} object
#' @param ... Rest of parameters, ignored
#' @export
to_keras.ruta_correntropy <- function(x, ...) {
  sigma <- x$sigma

  robust_kernel <- function(alpha) {
    1 / (sqrt(2 * pi * sigma)) *
      keras::k_exp(- keras::k_square(alpha) / (2 * sigma * sigma))
  }

  # correntropy loss
  function(y_true, y_pred) {
    - keras::k_sum(robust_kernel(y_pred - y_true))
  }
}
