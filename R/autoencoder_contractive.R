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
  if (!(ruta_contraction %in% class(learner$loss))) {
    learner$loss = contraction(learner$loss, weight)
  }

  learner
}

to_keras.ruta_contraction <- function(x, keras_model, ...) {
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
