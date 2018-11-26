add_custom_penalty <- function(learner, weight, func) {
  old_loss <- learner$loss
  learner$loss <- structure(list(
    reconstruction = old_loss,
    f = func
  ), class = "ruta_loss_penalty")
  learner
}

to_keras.ruta_loss_penalty <- function(loss, learner) {
  old_loss <- loss$reconstruction %>% as_loss() %>% to_keras()
  penalty <- loss$f(learner)

  function(y_true, y_pred) {
    rec_err <- old_loss(y_true, y_pred)
    rec_err + weight * penalty(y_true, y_pred)
  }
}

add_penalty_entropy <- function(learner, weight = 1) {
  encoding_layer <- learner$network[[learner$network %@% "encoding"]]

  if (!(encoding_layer$activation %in% c("tanh", "sigmoid", "softsign", "hard_sigmoid"))) {
    message("This regularization is better defined for bounded activation functions (with an infimum and a supremum) in the encoding layer. Performance could be affected by this.")
  }

  learner$network[[learner$network %@% "encoding"]]$activity_regularizer <- penalty_entropy(weight)

  learner
}

penalty_entropy <- function(weight) {
  structure(list(weight = weight), class = "ruta_penalty_entropy")
}

to_keras.ruta_penalty_entropy <- function(x, activation) {
  # This regularization only makes sense for bounded activation functions, but we
  # adapt it to any other activation by defining high value as > 1 and low value
  # as < -1
  low_v = switch(activation,
                 sigmoid = 0,
                 hard_sigmoid = 0,
                 relu = 0,
                 softplus = 0,
                 selu = - 1.7581,
                 -1
  )
  high_v = 1


  function(observed_activations) {
    observed <- observed_activations %>%
      keras::k_mean(axis = 1) %>%
      keras::k_clip(low_v + keras::k_epsilon(), high_v - keras::k_epsilon())

    # rescale means: what we want to calculate is the probability of a high value
    q_high <- (observed - low_v) / (high_v - low_v)

    # Max entropy = Min (1 - entropy)
    1 + keras::k_mean(
      q_high * keras::k_log(q_high) +
        (1 - q_high) * keras::k_log(1 - q_high)
    )
  }
}
