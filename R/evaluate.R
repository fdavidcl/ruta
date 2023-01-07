#' Custom evaluation metrics
#'
#' Create a different evaluation metric from a valid Keras metric
#'
#' @param evaluate_f Must be either a metric function defined by Keras (e.g.
#'   `keras::metric_binary_crossentropy`) or a valid function for Keras to
#'   create a performance metric (see `\link[keras]{metric_binary_accuracy}`
#'   for details)
#' @return A function which can be called with parameters `learner` and `data`
#'   just like the ones defined in `\link[ruta]{evaluate}`.
#' @seealso `\link[ruta]{evaluate}`
#' @export
evaluation_metric <- function(evaluate_f) function(learner, data, ...) {
  k_model <- learner$models$autoencoder

  keras::compile(
    k_model,
    optimizer = "sgd",
    loss = to_keras(learner$loss, learner),
    metrics = evaluate_f
  )
  keras::evaluate(k_model, x = data, y = data, ...)
}

#' Evaluation metrics
#'
#' Performance evaluation metrics for autoencoders
#'
#' @param learner A trained learner object
#' @param data Test data for evaluation
#' @param ... Additional parameters passed to `keras::\link[keras]{evaluate}`.
#' @return A named list with the autoencoder training loss and evaluation metric for the
#'   given data
#' @examples
#' x <- as.matrix(sample(iris[, 1:4]))
#' x_train <- x[1:100, ]
#' x_test <- x[101:150, ]
#'
#' \donttest{
#' autoencoder(2) |>
#'   train(x_train) |>
#'   evaluate_mean_squared_error(x_test)
#' }
#'
#' @seealso `\link{evaluation_metric}`
#' @rdname evaluate
#' @export
evaluate_mean_squared_error <- evaluation_metric(keras::metric_mean_squared_error)

#' @rdname evaluate
#' @export
evaluate_mean_absolute_error <- evaluation_metric(keras::metric_mean_absolute_error)

#' @rdname evaluate
#' @export
evaluate_binary_crossentropy <- evaluation_metric(keras::metric_binary_crossentropy)

#' @rdname evaluate
#' @export
evaluate_binary_accuracy <- evaluation_metric(keras::metric_binary_accuracy)

#' @rdname evaluate
#' @export
evaluate_kullback_leibler_divergence <- evaluation_metric(keras::metric_kullback_leibler_divergence)
