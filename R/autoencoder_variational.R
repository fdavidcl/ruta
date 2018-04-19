#' This file contains excerpts of code from Keras examples demonstrating how to
#' build a variational autoencoder with Keras.
#' Source: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
#' Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

autoencoder_variational <- function(intermediate, latent_dim) {
  structure(
    list(
      intermediate = intermediate,
      latent_dim = latent_dim,
      loss = variational_loss(0.5)
    ),
    class = c(ruta_autoencoder_variational, ruta_autoencoder)
  )
}

variational_block <- function(units) {
  make_atomic_network(ruta_layer_variational, units, "linear")
}

to_keras.ruta_layer_variational <- function(x, input_shape, model = keras::keras_model_sequential(), ...) {
  epsilon_std <- 1.0
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

    z_mean + k_exp(z_log_var/2)*epsilon
  }

  # "output_shape" isn't necessary with the TensorFlow backend
  z <- keras::layer_concatenate(list(z_mean, z_log_var)) %>%
    keras::layer_lambda(sampling)

}

to_keras.ruta_autoencoder_variational <- function(learner, input_shape) {
  original_dim <- input_shape
  intermediate_dim <- learner$intermediate
  latent_dim <- learner$latent_dim

  x <- layer_input(shape = c(original_dim))
  h <- x
  #for (intermediate_dim in intermediate) {
    h <- layer_dense(h, intermediate_dim, activation = "relu")
  #}

  z <- to_keras(variational_block(latent_dim), input_shape, h)

  # we instantiate these layers separately so as to reuse them later
  decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
  decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
  h_decoded <- decoder_h(z)
  x_decoded_mean <- decoder_mean(h_decoded)

  # end-to-end autoencoder
  vae <- keras_model(x, x_decoded_mean)

  # encoder, from inputs to latent space
  encoder <- keras_model(x, z_mean)

  # generator, from latent space to reconstructed inputs
  decoder_input <- layer_input(shape = latent_dim)
  h_decoded_2 <- decoder_h(decoder_input)
  x_decoded_mean_2 <- decoder_mean(h_decoded_2)
  generator <- keras_model(decoder_input, x_decoded_mean_2)

  list(
    autoencoder = vae,
    encoder = encoder,
    decoder = generator
  )

}

variational_loss <- function(kl_coeff) {
  structure(list(kl_coeff = kl_coeff), class = c(ruta_variational_loss, ruta_loss))
}

to_keras.ruta_variational_loss <- function(loss, model) {
  z_mean <- get_layer(model, name = "z_mean")
  z_log_var <- get_layer(model, name = "z_log_var")

  function(x, x_decoded_mean) {
    # xent_loss <- original_dim * loss_binary_crossentropy(x, x_decoded_mean)
    xent_loss <- loss_binary_crossentropy(x, x_decoded_mean)
    kl_loss <- 0.5 * k_mean(k_square(z_mean$output) + k_exp(z_log_var$output) - 1 - z_log_var$output, axis = -1L)
    xent_loss + kl_loss
  }
}
