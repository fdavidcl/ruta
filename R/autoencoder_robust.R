#' Create a robust autoencoder
#'
#' A robust autoencoder uses a special objective function,
#' correntropy, a localized similarity measure which makes it less sensitive
#' to noise in data. Correntropy specifically measures the probability density
#' that two events are equal, and is less affected by outliers than the mean
#' squared error.
#'
#' @param network Layer construct of class \code{"ruta_network"}
#' @param sigma Sigma parameter in the kernel used for correntropy
#'
#' @return A construct of class \code{"ruta_autoencoder"}
#'
#' @references
#' - [Robust feature learning by stacked autoencoder with maximum correntropy criterion](https://ieeexplore.ieee.org/abstract/document/6854900/)
#'
#' @family autoencoder variants
#' @export
autoencoder_robust <- function(network, sigma = 0.2) {
  autoencoder(network, correntropy(sigma))
}

#' Correntropy loss
#'
#' A wrapper for the correntropy loss function
#'
#' @param sigma Sigma parameter in the kernel
#'
#' @return A \code{"ruta_loss"} object
#' @seealso `\link{autoencoder_robust}`
#' @family loss functions
#' @export
correntropy <- function(sigma = 0.2) {
  structure(
    list(sigma = sigma),
    class = c(ruta_loss_correntropy, ruta_loss)
  )
}

#' Add robust behavior to any autoencoder
#'
#' Converts an autoencoder into a robust one by assigning a
#' correntropy loss to it. Notice that this will replace the previous loss
#' function
#'
#' @param learner The \code{"ruta_autoencoder"} object
#' @param sigma Sigma parameter in the kernel used for correntropy
#'
#' @return An autoencoder object which contains the correntropy loss
#' @seealso `\link{autoencoder_robust}`
#' @export
make_robust <- function(learner, sigma = 0.2) {
  learner$loss <- correntropy(sigma)
  learner
}

#' Detect whether an autoencoder is robust
#' @param learner A \code{"ruta_autoencoder"} object
#' @return Logical value indicating if a correntropy loss was found
#' @seealso `\link{correntropy}`, `\link{autoencoder_robust}`, `\link{make_robust}`
#' @export
is_robust <- function(learner) {
  ruta_loss_correntropy %in% class(learner$loss)
}


#' @rdname to_keras.ruta_loss_named
#' @references
#' - Correntropy loss: [Robust feature learning by stacked autoencoder with maximum correntropy criterion](https://ieeexplore.ieee.org/abstract/document/6854900/)
#'
#' @export
to_keras.ruta_loss_correntropy <- function(x, ...) {
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
