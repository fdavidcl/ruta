# This file contains excerpts of code from Keras examples demonstrating how to
# build a variational autoencoder with Keras.
# Source: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
# Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

#' Build a variational autoencoder
#'
#' A variational autoencoder assumes that a latent, unobserved random variable produces
#' the observed data and attempts to approximate its distribution. This function
#' constructs a wrapper for a variational autoencoder using a Gaussian
#' distribution as the prior of the latent space.
#'
#' @param network Network architecture as a `"ruta_network"` object (or coercible)
#' @param loss Reconstruction error to be combined with KL divergence in order to compute
#'   the variational loss
#' @param auto_transform_network Boolean: convert the encoding layer into a variational block if none is found?
#'
#' @return A construct of class \code{"ruta_autoencoder"}
#'
#' @import purrr
#' @examples
#' network <-
#'   input() +
#'   dense(256, "elu") +
#'   variational_block(3) +
#'   dense(256, "elu") +
#'   output("sigmoid")
#'
#' learner <- autoencoder_variational(network, loss = "binary_crossentropy")
#' @references
#' - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
#' - [Under the Hood of the Variational Autoencoder (in Prose and Code)](http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html)
#' - [Keras example: Variational autoencoder](https://keras.rstudio.com/articles/examples/variational_autoencoder.html)
#'
#' @family autoencoder variants
#' @export
autoencoder_variational <- function(network, loss = "binary_crossentropy", auto_transform_network = TRUE) {
  network <- as_network(network)

  if (detect_index(network, ~ ruta_layer_variational %in% class(.)) == 0) {
    if (auto_transform_network) {
      message("Transforming encoding layer into variational block")
      encoding_units <- network[[network %@% "encoding"]]$units
      network[network %@% "encoding"] <- variational_block(encoding_units)
    } else {
      stop("Can't build a variational autoencoder without a variational block")
    }
  }

  new_autoencoder(network, loss_variational(loss), extra_class = ruta_autoencoder_variational)
}

#' Detect whether an autoencoder is variational
#' @param learner A \code{"ruta_autoencoder"} object
#' @return Logical value indicating if a variational loss was found
#' @seealso `\link{autoencoder_variational}`
#' @export
is_variational <- function(learner) {
  ruta_loss_variational %in% class(learner$loss)
}

#' Create a variational block of layers
#'
#' This variational block consists in two dense layers which take as input the previous layer
#' and a sampling layer. More specifically, these layers aim to represent the mean and the
#' log variance of the learned distribution in a variational autoencoder.
#' @param units Number of units
#' @return A construct with class \code{"ruta_layer"}
#' @examples
#' variational_block(3)
#' @family neural layers
#' @seealso `\link{autoencoder_variational}`
#' @export
variational_block <- function(units) {
  make_atomic_network(ruta_layer_variational, units = units)
}

#' Obtain a Keras block of layers for the variational autoencoder
#'
#' This block contains two dense layers representing the mean and log var of a Gaussian
#' distribution and a sampling layer.
#'
#' @param x The layer object
#' @param input_shape Number of features in training data
#' @param model Keras model where the layers will be added
#' @param ... Unused
#' @return A Layer object from Keras
#'
#' @references
#' - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
#' - [Under the Hood of the Variational Autoencoder (in Prose and Code)](http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html)
#' - [Keras example: Variational autoencoder](https://keras.rstudio.com/articles/examples/variational_autoencoder.html)
#' @export
to_keras.ruta_layer_variational <- function(x, input_shape, model = keras::keras_model_sequential(), ...) {
  epsilon_std <- 1.0
  latent_dim <- x$units
  z_mean <- keras::layer_dense(model, latent_dim, name = "z_mean")
  z_log_var <- keras::layer_dense(model, latent_dim, name = "z_log_var")

  sampling <- function(arg){
    z_mean <- arg[, 1:(latent_dim)]
    z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]

    epsilon <- keras::k_random_normal(
      shape = c(keras::k_shape(z_mean)[[1]]),
      mean = 0.,
      stddev = epsilon_std
    )

    z_mean + keras::k_exp(z_log_var/2)*epsilon
  }

  # "output_shape" isn't necessary with the TensorFlow backend
  keras::layer_concatenate(list(z_mean, z_log_var)) %>%
    keras::layer_lambda(sampling, name = "sampling")
}

#' @rdname to_keras.ruta_autoencoder
#' @import purrr
#' @export
to_keras.ruta_autoencoder_variational <- function(learner, input_shape) {
  to_keras.ruta_autoencoder(learner, input_shape, encoder_end = "z_mean", decoder_start = "sampling")
}

#' Variational loss
#'
#' Specifies an evaluation function adapted to the variational autoencoder. It combines
#' a base reconstruction error and the Kullback-Leibler divergence between the learned
#' distribution and the true latent posterior.
#' @param reconstruction_loss Another loss to be used as reconstruction error (e.g. "binary_crossentropy")
#' @return A \code{"ruta_loss"} object
#' @references
#' - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
#' - [Under the Hood of the Variational Autoencoder (in Prose and Code)](http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html)
#' - [Keras example: Variational autoencoder](https://keras.rstudio.com/articles/examples/variational_autoencoder.html)
#' @seealso `\link{autoencoder_variational}`
#' @family loss functions
#' @export
loss_variational <- function(reconstruction_loss) {
  structure(
    list(reconstruction_loss = reconstruction_loss),
    class = c(ruta_loss_variational, ruta_loss)
  )
}

#' @rdname to_keras.ruta_loss_named
#' @references
#' - Variational loss:
#'     - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
#'     - [Under the Hood of the Variational Autoencoder (in Prose and Code)](http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html)
#'     - [Keras example: Variational autoencoder](https://keras.rstudio.com/articles/examples/variational_autoencoder.html)
#' @export
to_keras.ruta_loss_variational <- function(loss, learner, ...) {
  keras_model <- learner$models$autoencoder
  original_dim <- 1. * keras_model$input_shape[[2]]
  reconstruction_loss <- loss$reconstruction_loss %>% as_loss() %>% to_keras()
  z_mean <- keras::get_layer(keras_model, name = "z_mean")
  z_log_var <- keras::get_layer(keras_model, name = "z_log_var")

  function(x, x_decoded_mean) {
    xent_loss <- original_dim * reconstruction_loss(x, x_decoded_mean)
    kl_loss <- 0.5 * keras::k_mean(keras::k_square(z_mean$output) + keras::k_exp(z_log_var$output) - 1 - z_log_var$output, axis = -1L)
    xent_loss + kl_loss
  }
}

#' @import purrr
#' @rdname generate
#' @param dimensions Indices of the dimensions over which the model will be sampled
#' @param from Lower limit on the values which will be passed to the inverse CDF of the prior
#' @param to Upper limit on the values which will be passed to the inverse CDF of the prior
#' @param side Number of steps to take in each traversed dimension
#' @param fixed_values Value used as parameter for the inverse CDF of all non-traversed dimensions
#' @param ... Unused
#' @seealso `\link{autoencoder_variational}`
#' @export
generate.ruta_autoencoder_variational <- function(learner, dimensions = c(1, 2), from = 0.05, to = 0.95, side = 10, fixed_values = 0.5, ...) {
  d <- learner$models$decoder$input_shape[[2]]
  md <- length(dimensions)

  # Values from the inverse CDF of the Gaussian distribution
  col <- seq(from = from, to = to, length.out = side) %>% qnorm()

  args <- rep(list(col), times = md)
  names(args) <- paste("D", dimensions)
  moving_dims <- cross_df(args)

  # TODO Allow for different fixed values in each constant dimension
  encoded <-
    fixed_values %>%
    rep(side ** md) %>%
    qnorm() %>%
    list() %>%
    rep(d) %>%
    data.frame()

  encoded[, dimensions] <- moving_dims
  encoded <- as.matrix(encoded)

  sampled <- decode(learner, encoded)
}
