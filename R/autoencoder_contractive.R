#' Create a contractive autoencoder
#'
#' A contractive autoencoder adds a penalty term to the loss
#' function of a basic autoencoder which attempts to induce a contraction of
#' data in the latent space.
#'
#' @param network Layer construct of class \code{"ruta_network"}
#' @param loss Character string specifying the reconstruction error part of the loss function
#' @param weight Weight assigned to the contractive loss
#'
#' @return A construct of class \code{"ruta_autoencoder"}
#'
#' @references
#' - [A practical tutorial on autoencoders for nonlinear feature fusion](https://arxiv.org/abs/1801.01586)
#'
#' @family autoencoder variants
#' @export
autoencoder_contractive <- function(network, loss = "mean_squared_error", weight = 2e-4) {
  autoencoder(network, loss) %>%
    make_contractive(weight)
}

#' Contractive loss
#'
#' @description This is a wrapper for a loss which induces a contraction in the
#' latent space.
#'
#' @param reconstruction_loss Original reconstruction error to be combined with the
#' contractive loss (e.g. `"binary_crossentropy"`)
#' @param weight Weight assigned to the contractive loss
#' @return A loss object which can be converted into a Keras loss
#'
#' @seealso `\link{autoencoder_contractive}`
#' @family loss functions
#' @export
contraction <- function(reconstruction_loss = "mean_squared_error", weight = 2e-4) {
  structure(
    list(
      reconstruction_loss = reconstruction_loss,
      weight = weight
    ),
    class = c(ruta_loss_contraction, ruta_loss)
  )
}

#' Add contractive behavior to any autoencoder
#'
#' @description Converts an autoencoder into a contractive one by assigning a
#' contractive loss to it
#'
#' @param learner The \code{"ruta_autoencoder"} object
#' @param weight Weight assigned to the contractive loss
#'
#' @return An autoencoder object which contains the contractive loss
#'
#' @seealso `\link{autoencoder_contractive}`
#' @export
make_contractive <- function(learner, weight = 2e-4) {
  if (!(ruta_loss_contraction %in% class(learner$loss))) {
    learner$loss = contraction(learner$loss, weight)
  }

  learner
}

#' @rdname to_keras.ruta_loss_named
#' @references
#' - Contractive loss: \href{https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/}{Deriving Contractive Autoencoder and Implementing it in Keras}
#' @export
to_keras.ruta_loss_contraction <- function(x, keras_model, ...) {
  rec_err <- x$reconstruction_loss %>% as_loss() %>% to_keras()
  encoding_layer <- keras::get_layer(keras_model, name = "encoding")

  # derivative of the activation function -- only tanh for now
  act_der <- function(h) 1 - h * h

  # contractive loss
  function(y_true, y_pred) {
    reconstruction <- rec_err(y_true, y_pred)

    sum_wt2 <-
      keras::k_variable(value = keras::get_weights(encoding_layer)[[1]]) %>%
      keras::k_transpose() %>%
      keras::k_square() %>%
      keras::k_sum(axis = 2)

    dh2 <- act_der(encoding_layer$output) ** 2

    contractive <- x$weight * keras::k_sum(dh2 * sum_wt2, axis = 2)

    reconstruction + contractive
  }
}
