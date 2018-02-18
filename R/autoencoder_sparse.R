sparsity <- function(expected_value, weight) {
  structure(
    list(
      expected_value = expected_value,
      weight = weight
    ),
    class = c(ruta_regularizer, ruta_sparsity)
  )
}

make_sparse <- function(learner, expected_value, weight = 0.2) {
  # TODO warn when sparsity expected value and encoding activation function
  # don't match (e.g. -0.8 for sigmoid or relu)
  learner$layers[[encoding_index(learner$layers)]]$activity_regularizer <- sparsity(expected_value, weight)

  learner
}

is_sparse <- function(learner) {
  !is.null(learner$layers[[encoding_index(learner$layers)]]$activity_regularizer)
}

to_keras.ruta_sparsity <- function(x, ...) {
  expected = x$expected_value
  # TODO detect tanh instead / use KL for tanh
  rescale = FALSE
  if (expected < 0) {
    rescale = TRUE
    expected = (1 + expected) / 2
  }

  function(observed_activations) {
    if (rescale) {
      observed_activations = (1 + observed_activations) / 2
    }
    observed <- observed_activations %>%
      keras::k_mean(axis = 0) %>%
      keras::k_clip(keras::k_epsilon(), 1)

    keras::k_sum(expected * keras::k_log(expected / observed) +
                   (1 - expected) * keras::k_log((1 - expected) / (1 - observed)))
  }
}

