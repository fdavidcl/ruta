#' @import purrr
runif_matrix <- function(data) {
  dims <- dim(data)
  dims %>%
    prod() %>%
    stats::runif() %>%
    array(dim = dims)
}

new_noise <- function(cl, ...) {
  structure(
    list(...),
    class = c(cl, ruta_noise, ruta_filter)
  )
}

#' Noise generator
#'
#' Delegates on noise classes to generate noise of some type
#' @param type Type of noise, as a character string
#' @param ... Parameters for each noise class
#' @export
noise <- function(type, ...) {
  noise_f <- switch(tolower(type),
                    zeros = noise_zeros,
                    ones = noise_ones,
                    saltpepper = noise_saltpepper,
                    gaussian = noise_gaussian,
                    cauchy = noise_cauchy,
                    NULL
  )

  if (is.null(noise_f)) {
    stop("Invalid noise type selected")
  }

  noise_f(...)
}

#' Filter to add zero noise
#'
#' A data filter which replaces some values with zeros
#'
#' @param p Probability that a feature in an instance is set to zero
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_zeros <- function(p = 0.05) {
  new_noise(ruta_noise_zeros, p = p)
}

#' @rdname apply_filter
#' @export
apply_filter.ruta_noise_zeros <- function(filter, data, ...) {
  multiplier <- as.integer(runif_matrix(data) > filter$p)
  data * multiplier
}

#' Filter to add ones noise
#'
#' A data filter which replaces some values with ones
#'
#' @param p Probability that a feature in an instance is set to one
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_ones <- function(p = 0.05) {
  new_noise(ruta_noise_ones, p = p)
}

#' @rdname apply_filter
#' @export
apply_filter.ruta_noise_ones <- function(filter, data, ...) {
  term <- runif_matrix(data)
  data[term < filter$p] <- 1
  data
}

#' Filter to add salt-and-pepper noise
#'
#' A data filter which replaces some values with zeros or ones
#'
#' @param p Probability that a feature in an instance is set to zero or one
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_saltpepper <- function(p = 0.05) {
  new_noise(ruta_noise_saltpepper, p = p)
}

#' @rdname apply_filter
#' @export
apply_filter.ruta_noise_saltpepper <- function(filter, data, ...) {
  saltpepper <- runif_matrix(data)
  zero_mask <- saltpepper < filter$p/2
  one_mask <- saltpepper > (1 - filter$p/2)

  data[zero_mask] <- 0
  data[one_mask] <- 1
  data
}

#' Additive Gaussian noise
#'
#' A data filter which adds Gaussian noise to instances
#'
#' @param sd Standard deviation for the Gaussian distribution
#' @param var Variance of the Gaussian distribution (optional, only used
#'  if `sd` is not provided)
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_gaussian <- function(sd = NULL, var = NULL) {
  if (is.null(sd)) {
    sd <- if (is.null(var))
      0.1
    else
      sqrt(var)
  }

  new_noise(ruta_noise_gaussian, sd = sd)
}

#' @rdname apply_filter
#' @import purrr
#' @export
apply_filter.ruta_noise_gaussian <- function(filter, data, ...) {

  dims <- dim(data)
  term <-
    dims %>%
    prod() %>%
    stats::rnorm(sd = filter$sd) %>%
    array(dim = dims)

  data + term
}

#' Additive Cauchy noise
#'
#' A data filter which adds noise from a Cauchy distribution to instances
#'
#' @param scale Scale for the Cauchy distribution
#' @return Object which can be applied to data with `\link{apply_filter}`
#' @family noise generators
#' @export
noise_cauchy <- function(scale = 0.005) {
  new_noise(ruta_noise_cauchy, scale = scale)
}

#' @rdname apply_filter
#' @import purrr
#' @export
apply_filter.ruta_noise_cauchy <- function(filter, data, ...) {
  dims <- dim(data)
  term <-
    dims %>%
    prod() %>%
    stats::rcauchy(scale = filter$scale) %>%
    array(dim = dims)

  data + term
}

#' @import R.utils
#' @param data
#' @param batch_size
to_keras.ruta_filter <- function(x, data, batch_size, ...) {
  limit <- dim(data)[1]
  order <- sample.int(limit)
  start <- 1
  function() {
    if (start + batch_size > limit) {
      idx <- order[start:limit]
      order <- sample.int(limit)
      start <- 1
    } else {
      idx <- order[start:(start + batch_size - 1)]
      start <- start + batch_size
    }
    original <- R.utils::extract(data, "1" = idx)
    noisy <- apply_filter(x, original)
    list(noisy, original)
  }
}
