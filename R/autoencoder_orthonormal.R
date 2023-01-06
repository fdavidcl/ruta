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
#' @import purrr
#' @export
autoencoder_orthonormal <- function(network, loss = "mean_squared_error", weight = 1e-3) {
  autoencoder(network, loss) %>%
    make_orthonormal(weight)
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
orthonormal_loss <- function(reconstruction_loss = "mean_squared_error", weight = 1e-3) {
  structure(
    list(
      reconstruction_loss = reconstruction_loss,
      weight = weight
    ),
    class = c("ruta_loss_orthonormal", ruta_loss)
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
make_orthonormal <- function(learner, weight = 2e-4) {
  if (!is_orthonormal(learner)) {
    learner$loss = orthonormal_loss(learner$loss, weight)
  }

  learner
}

#' Detect whether an autoencoder is contractive
#' @param learner A \code{"ruta_autoencoder"} object
#' @return Logical value indicating if a contractive loss was found
#' @seealso `\link{contraction}`, `\link{autoencoder_contractive}`, `\link{make_contractive}`
#' @export
is_orthonormal <- function(learner) {
  "ruta_loss_orthonormal" %in% class(learner$loss)
}

#' @rdname to_keras.ruta_loss_named
#' @param learner The learner object including the keras model which will use the loss
#'   function
#' @references
#' - Contractive loss: \href{https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/}{Deriving Contractive Autoencoder and Implementing it in Keras}
#' @import purrr
#' @export
to_keras.ruta_loss_orthonormal <- function(x, learner, ...) {
  rec_err <- x$reconstruction_loss %>% as_loss() %>% to_keras()

  keras_model <- learner$models$autoencoder
  input_x <- keras::get_layer(keras_model, index = 0)$output
  encoding_h <- keras::get_layer(keras_model, name = "encoding")$output
  encoding_len <- learner$network[[learner$network %@% "encoding"]]$units
  # Identity matrix
  # shape = (encoding_size, encoding_size)
  id <- keras::k_eye(size = as.integer(encoding_len))

  # contractive loss
  orthonormal <- function(y_true, y_pred) {
    reconstruction <- rec_err(y_true, y_pred)

    # Compute the Jacobian matrix of the encoding with respect to the inputs
    # shape = (batch_size, encoding_size, input_size)
    reg <- jacobian(encoding_h, input_x) %>%
      # Matrix product Jf Jf^T
      # shape = (batch_size, encoding_size, encoding_size)
      keras::k_batch_dot(., ., axes = 3) %>%
      # (Jf Jf^T) - I
      `-`(keras::k_expand_dims(id, axis = 1)) %>%
      # Compute square values and sum the matrix for each instance in the batch
      keras::k_square() %>%
      keras::k_sum(axis = list(2, 3))

    reconstruction + x$weight * reg
  }
}
