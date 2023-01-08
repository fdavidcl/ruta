#' Create a denoising autoencoder
#'
#' @description A denoising autoencoder trains with noisy data in order to
#' create a model able to reduce noise in reconstructions from input data
#'
#' @param network Layer construct of class \code{"ruta_network"}
#' @param loss Loss function to be optimized
#' @param noise_type Type of data corruption which will be used to train the
#'   autoencoder, as a character string. Available types:
#'   - `"zeros"` Randomly set components to zero (`\link{noise_zeros}`)
#'   - `"ones"` Randomly set components to one (`\link{noise_ones}`)
#'   - `"saltpepper"` Randomly set components to zero or one (`\link{noise_saltpepper}`)
#'   - `"gaussian"` Randomly offset each component of an input as drawn from
#'     Gaussian distributions with the same variance (additive Gaussian noise,
#'     `\link{noise_gaussian}`)
#'   - `"cauchy"` Randomly offset each component of an input as drawn from
#'     Cauchy distributions with the same scale (additive Cauchy noise,
#'     `\link{noise_cauchy}`)
#' @param ... Extra parameters to customize the noisy filter:
#'   - `p` The probability that each instance in the input data which will be
#'     altered by random noise (for `"zeros"`, `"ones"` and `"saltpepper"`)
#'   - `var` or `sd` The variance or standard deviation of the Gaussian
#'     distribution from which additive noise will be drawn (for `"gaussian"`,
#'     only one of those parameters is necessary)
#'   - `scale` For the Cauchy distribution
#'
#' @return A construct of class \code{"ruta_autoencoder"}
#'
#' @references
#' - [Extracting and composing robust features with denoising autoencoders](https://dl.acm.org/doi/10.1145/1390156.1390294)
#'
#' @family autoencoder variants
#' @export
autoencoder_denoising <- function(network, loss = "mean_squared_error", noise_type = "zeros", ...) {
  autoencoder(network, loss) |>
    make_denoising(noise_type, ...)
}

#' Add denoising behavior to any autoencoder
#'
#' @description Converts an autoencoder into a denoising one by adding a filter
#' for the input data
#'
#' @param learner The \code{"ruta_autoencoder"} object
#' @param noise_type Type of data corruption which will be used to train the
#'   autoencoder, as a character string. See `\link{autoencoder_denoising}` for
#'   details
#' @param ... Extra parameters to customize the noisy filter. See
#'  `\link{autoencoder_denoising}` for details
#'
#' @return An autoencoder object which contains the noisy filter
#' @seealso `\link{autoencoder_denoising}`
#' @export
make_denoising <- function(learner, noise_type = "zeros", ...) {
  learner$filter <- noise(noise_type, ...)
  learner
}


#' Detect whether an autoencoder is denoising
#' @param learner A \code{"ruta_autoencoder"} object
#' @return Logical value indicating if a noise generator was found
#' @seealso `\link{noise}`, `\link{autoencoder_denoising}`, `\link{make_denoising}`
#' @export
is_denoising <- function(learner) {
  (!is.null(learner$filter)) & (ruta_noise %in% class(learner$filter))
}

