contraction <- function(rec_err, weight) {
  structure(
    list(
      reconstruction <- rec_err,
      weight = weight
    ),
    class = c(ruta_loss, ruta_contraction)
  )
}

to_keras.ruta_contraction <- function(x, keras_model) {
  contractive_loss <- function(y_pred, y_true) {
    reconstruction <- rec_err(y_true, y_pred)

    hid = keras::k_variable(value = keras_model$get_layer("encoded")$get_weights()[0]) %>%
      keras::k_transpose() %>%
      keras::k_square() %>%
      keras::k_sum(axis = 1)
    
    contractive = x$weight * keras::k_sum()
  }
}
