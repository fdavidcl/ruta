autoencoder_robust <- function(network, sigma) {
  autoencoder(network, correntropy(sigma))
}

correntropy <- function(sigma) {
  structure(
    list(sigma = sigma),
    class = c(ruta_loss, ruta_correntropy)
  )
}

make_robust <- function(learner, sigma) {
  # message("This will replace the previous loss function")
  learner$loss <- correntropy(sigma)
}

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
