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
  autoencoder(network, loss) |>
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
  if (!is_contractive(learner)) {
    learner$loss = contraction(learner$loss, weight)
  }

  learner
}

#' Detect whether an autoencoder is contractive
#' @param learner A \code{"ruta_autoencoder"} object
#' @return Logical value indicating if a contractive loss was found
#' @seealso `\link{contraction}`, `\link{autoencoder_contractive}`, `\link{make_contractive}`
#' @export
is_contractive <- function(learner) {
  ruta_loss_contraction %in% class(learner$loss)
}

#' @rdname to_keras.ruta_loss_named
#' @param learner The learner object including the keras model which will use the loss
#'   function
#' @references
#' - Contractive loss: \href{https://agustinus.kristia.de/techblog/2016/12/05/contractive-autoencoder/}{Deriving Contractive Autoencoder and Implementing it in Keras}
#' @export
to_keras.ruta_loss_contraction <- function(x, learner, ...) {
  keras_model <- learner$models$autoencoder
  stopifnot(keras_model != NULL)
  rec_err <- x$reconstruction_loss |> as_loss() |> to_keras()
  input_x <- keras_model$input
  encoding_h <- keras_model$get_layer("encoding")$output

  # contractive loss
  function(y_true, y_pred) {
    reconstruction <- rec_err(y_true, y_pred)

    ## Code attempting to comput the loss with TF does not work
    ## because TF tensors clash with Keras Tensors
    # contractive <-
    #   tape$jacobian(encoding_h, input_x) |>
    #   # Squared Frobenius norm: sum of squared values of the matrix
    #   tensorflow::tf$square() |>
    #   tensorflow::tf$reduce_sum(axis = list(2, 3))

    # Instead, use k_gradients to compute derivatives
    contractive <-
      keras::k_gradients(encoding_h, input_x) |>
      keras::k_square() |>
      keras::k_sum(axis = list(2, 3))

    reconstruction + x$weight * contractive
  }
}
