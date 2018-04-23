# This file contains excerpts of code from Keras examples demonstrating how to
# build a variational autoencoder with Keras.
# Source: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
# Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

#' Build a variational autoencoder
#'
#' @param intermediate Intermediate size
#' @param latent_dim Latent space dimension
#' @param loss Reconstruction error to be combined with KL divergence in order to compute
#'   the variational loss
#' @import purrr
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

  new_autoencoder(network, variational_loss(loss), extra_class = ruta_autoencoder_variational)
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
#' @export
variational_block <- function(units) {
  make_atomic_network(ruta_layer_variational, units, "linear")
}

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

variational_loss <- function(base_loss) {
  structure(
    list(base_loss = base_loss),
    class = c(ruta_loss_variational, ruta_loss)
  )
}

to_keras.ruta_loss_variational <- function(loss, model) {
  original_dim <- 1. * model$input_shape[[2]]
  base_loss <- loss$base_loss %>% as_loss() %>% to_keras()
  z_mean <- keras::get_layer(model, name = "z_mean")
  z_log_var <- keras::get_layer(model, name = "z_log_var")

  function(x, x_decoded_mean) {
    # xent_loss <- original_dim * loss_binary_crossentropy(x, x_decoded_mean)
    xent_loss <- original_dim * base_loss(x, x_decoded_mean)
    kl_loss <- 0.5 * keras::k_mean(keras::k_square(z_mean$output) + keras::k_exp(z_log_var$output) - 1 - z_log_var$output, axis = -1L)
    xent_loss + kl_loss
  }
}

#' @import purrr
#' @export
generate.ruta_autoencoder_variational <- function(learner, dimensions = c(1, 2), from = 0.05, to = 0.95, side = 10, fixed_values = 0.5) {
  d <- learner$models$decoder$input_shape[[2]]
  md <- length(dimensions)
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

  sampled <- decode(model, encoded)
}
