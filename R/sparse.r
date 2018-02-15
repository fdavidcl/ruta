sparsity <- function(expected_value, weight) {
  structure(
    list(
      expected_value = expected_value,
      weight = weight
    ),
    class = c(rutaRegularizer, rutaSparsity)
  )
}

#' @export
makeSparse <- function(learner, expected_value, weight = 0.2) {
  # TODO warn when sparsity expected value and encoding activation function
  # don't match (e.g. -0.8 for sigmoid or relu)
  learner$layers[[encodingIndex(learner$layers)]]$activity_regularizer <- sparsity(expected_value, weight)

  learner
}

isSparse <- function(learner) {
  !is.null(learner$layers[[encodingIndex(learner$layers)]]$activity_regularizer)
}

#' @export
toKeras.rutaSparsity <- function(x, ...) {
  expected = x$expected_value
  # TODO detect tanh instead / use KL for tanh
  rescale = FALSE
  if (expected < 0) {
    rescale = TRUE
    expected = (1 + expected) / 2
  }

  # Kullback-Leibler divergence for two Bernoulli distributions
  # with mean `expected` and `v` respectively
  kl <- function(v) {
    if (rescale) {
      v = (1 + v) / 2
    }

    # TODO check if k_any would work here
    keras::k_switch(
      keras::k_less_equal(expected / v, 0),
      0,
      keras::k_switch(
        keras::k_less_equal((1 - expected) / (1 - v), 0),
        0,
        expected * keras::k_log(expected / v) +
          (1 - expected) * keras::k_log((1 - expected) / (1 - v))
      )
    )
  }

  # Regularizer
  function(observed_activations) {
    # outputs of encoding layer will be shaped (batch_size, num_attributes)
    observed = keras::k_mean(observed_activations, axis = 0)
    allkl = keras::k_map_fn(kl, observed)
    sumkl = keras::k_sum(allkl)

    x$weight * sumkl
  }
}
