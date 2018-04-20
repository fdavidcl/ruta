#' Create a contractive autoencoder
#'
#' @description A contractive autoencoder adds a penalty term to the loss
#' function of a basic autoencoder which attempts to induce a contraction of
#' data in the latent space.
#'
#' @param network Layer construct of class \code{"ruta_network"}
#' @param loss Character string specifying the reconstruction error part of the loss function
#' @param weight Weight assigned to the contractive loss
#'
#' @return A construct of class \code{"ruta_autoencoder"}
#' @export
autoencoder_contractive <- function(network, loss, weight) {
  autoencoder(network, loss) %>%
    make_contractive(weight)
}

#' Contractive loss
#'
#' @description This is a wrapper for a loss which induces a contraction in the
#' latent space.
#'
#' @param rec_err Original reconstruction error to be combined with the
#' contractive loss
#' @param weight Weight assigned to the contractive loss
#'
#' @return A loss object which can be converted into a Keras loss
#' @export
contraction <- function(rec_err, weight) {
  structure(
    list(
      reconstruction = rec_err,
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
#' @export
make_contractive <- function(learner, weight) {
  if (!(ruta_contraction %in% class(learner$loss))) {
    learner$loss = contraction(learner$loss, weight)
  }

  learner
}

#' Obtain a Keras contractive loss
#'
#' @description Builds the Keras loss function corresponding to the object received
#'
#' @param x A \code{"ruta_loss_contraction"} object
#' @param keras_model The keras autoencoder which will use the loss function
#' @return A function which returns the contractive loss for given true and
#' predicted values
#' @param ... Rest of parameters, ignored
#' @export
to_keras.ruta_loss_contraction <- function(x, keras_model, ...) {
  rec_err <- x$reconstruction %>% as_loss() %>% to_keras()
  encoding_layer <- keras::get_layer(keras_model, name = "encoded")

  # derivative of the activation function -- only tanh for now
  dh <- function(h) 1 - h * h

  # contractive loss
  function(y_true, y_pred) {
    reconstruction <- rec_err(y_true, y_pred)
    #reconstruction <- rec_err(y_pred, y_true)

    hid =
      # n x h
      keras::k_variable(value = encoding_layer$get_weights()[[1]]) %>%
      keras::k_square() %>%
      # 1 x h
      keras::k_sum(axis = 1)

    contractive = x$weight * keras::k_sum(dh(encoding_layer$output) ** 2 * hid)
    reconstruction + contractive
  }
}
