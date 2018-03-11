weight_decay <- function(decay = 0.02) {
  structure(
    list(decay = decay),
    class = c(ruta_regularizer, ruta_weight_decay)
  )
}

add_weight_decay <- function(learner, decay = 0.02) {
  # apply this regularizer only to the encoding?
  learner$network[[learner$network %@% "encoding"]]$kernel_regularizer <- weight_decay(decay)

  learner
}

to_keras.ruta_weight_decay <- function(x, ...) {
  keras::regularizer_l2(l = x$decay)
}
