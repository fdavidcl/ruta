autoencoder_denoising <- function(network, loss, ratio) {
  autoencoder(network, loss) %>%
    make_denoising(ratio)
}

noise_input <- function(x, ratio = 0.05) {
  noisy = x + keras::k_random_uniform(x$shape, minval = 0, maxval = 0.5 / (1 - ratio))
  keras::k_clip(noisy, 0, 1)
}
