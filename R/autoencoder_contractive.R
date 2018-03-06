autoencoder_contractive <- function(network, loss, weight) {
  autoencoder(network, loss) %>%
    make_contractive(weight)
}

contraction <- function(rec_err, weight) {
  structure(
    list(
      reconstruction = rec_err,
      weight = weight
    ),
    class = c(ruta_loss, ruta_contraction)
  )
}

make_contractive <- function(learner, weight) {
  # Save the original loss function in order to make this
  # function idempotent
  learner$reconstruction <- learner$reconstruction %||% learner$loss
  learner$loss = contraction(learner$reconstruction, weight)
  learner
}

to_keras.ruta_contraction <- function(x, keras_model, ...) {
  rec_err <- x$reconstruction %>% as_loss() %>% to_keras()

  # derivative of the activation function -- only tanh for now
  dh <- function(h) 1 - h * h

  # contractive loss
  function(y_pred, y_true) {
    reconstruction <- rec_err(y_true, y_pred)

    hid =
      keras::k_variable(value = keras::get_layer(keras_model, name = "encoded")$get_weights()[0]) %>%
      keras::k_transpose() %>%
      keras::k_square() %>%
      keras::k_sum(axis = 1)

    contractive = x$weight * keras::k_sum(
      dh(keras::get_layer(keras_model, name = "encoded")$output) ** 2 * hid)
    reconstruction + contractive
  }
}
