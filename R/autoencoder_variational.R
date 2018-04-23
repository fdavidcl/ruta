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
#' @export
autoencoder_variational <- function(intermediate, latent_dim, loss = "binary_crossentropy") {
  structure(
    list(
      intermediate = intermediate,
      latent_dim = latent_dim,
      loss = variational_loss(loss)
    ),
    class = c(ruta_autoencoder_variational, ruta_autoencoder)
  )
}

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
  z <- keras::layer_concatenate(list(z_mean, z_log_var)) %>%
    keras::layer_lambda(sampling)

  list(
    latent = z_mean,
    sampled = z
  )
}

to_keras.ruta_autoencoder_variational <- function(learner, input_shape) {
  original_dim <- input_shape
  intermediate_dim <- learner$intermediate
  latent_dim <- learner$latent_dim

  x <- keras::layer_input(shape = c(original_dim))
  h <- x
  #for (intermediate_dim in intermediate) {
    h <- keras::layer_dense(h, intermediate_dim, activation = "selu")
  #}

  encodings <- to_keras(variational_block(latent_dim)[[1]], input_shape, model = h)
  z_mean <- encodings$latent
  z <- encodings$sampled

  # we instantiate these layers separately so as to reuse them later
  decoder_h <- keras::layer_dense(units = intermediate_dim, activation = "selu")
  decoder_mean <- keras::layer_dense(units = original_dim, activation = "sigmoid")
  h_decoded <- decoder_h(z)
  x_decoded_mean <- decoder_mean(h_decoded)

  # end-to-end autoencoder
  vae <- keras::keras_model(x, x_decoded_mean)

  # encoder, from inputs to latent space
  encoder <- keras::keras_model(x, z_mean)

  # generator, from latent space to reconstructed inputs
  decoder_input <- keras::layer_input(shape = latent_dim)
  h_decoded_2 <- decoder_h(decoder_input)
  x_decoded_mean_2 <- decoder_mean(h_decoded_2)
  generator <- keras::keras_model(decoder_input, x_decoded_mean_2)

  list(
    autoencoder = vae,
    encoder = encoder,
    decoder = generator
  )
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
  cat(d)
  md <- length(dimensions)
  col <- seq(from = from, to = to, length.out = side) %>% qnorm()

  args <- rep(list(col), times = md)
  names(args) <- paste("D", dimensions)
  moving_dims <- cross_df(args)

  fixed <- rep(fixed_values, times = side ** md) %>% qnorm()

  encoded <- data.frame(rep(list(fixed), d))
  encoded[, dimensions] <- moving_dims
  # encoded <- data.frame(col1, col2) %>% as.matrix()
  sampled <- model %>% decode(encoded)
}
