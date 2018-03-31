autoencoder_denoising <- function(network, loss, ratio = 0.05) {
  autoencoder(network, loss) %>%
    make_denoising(ratio)
}

# noise_input <- function(x, ratio = 0.05) {
#   noisy = x + keras::k_random_uniform(x$shape, minval = 0, maxval = 0.5 / (1 - ratio))
#   keras::k_clip(noisy, 0, 1)
# }

make_denoising <- function(learner, ratio = 0.05) {
  learner$filter <- structure(
    list(ratio = ratio),
    class = c(ruta_filter, ruta_noise)
  )

  learner
}

## Random setting to 0 or 1,
## assumes data is normalized :(
apply_filter.ruta_noise <- function(filter, data) {
  # noisy = data + keras::k_random_uniform(k_shape(data), minval = 0, maxval = 0.5 / (1 - filter$ratio), dtype = "float64")
  # keras::k_clip(noisy, 0, 1)
  # k_get_value(noisy) ?
  # how to train with this ?

  noisy <- data + matrix(
    data %>% dim %>% prod %>% runif(min = 0, max = 0.5 / (1 - filter$ratio)),
    nrow = dim(data)[1]
  )
  pmin(noisy, 1)
}
