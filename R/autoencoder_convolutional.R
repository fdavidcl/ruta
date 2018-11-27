
example_convolutional <- function() {
plot_square <- function(square, ...) {
  image(t(square)[,28:1], xaxt = "n", yaxt = "n", col = gray((255:0)/255), ...)
}

plot_sample <- function(digits_test, digits_dec, sample) {
  sample_size <- length(sample)
  layout(
    matrix(c(1:(2 * sample_size)), byrow = F, nrow = 2)
  )

  for (i in sample) {
    par(mar = c(0,0,0,0) + 1)
    plot_square(digits_test[i,,,1 ])
    plot_square(digits_dec[i,,,1 ])
  }
}
mnist <- dataset_mnist()
x_train <- mnist$train$x / 255.0
x_test <- mnist$test$x / 255.0

net <- input() +
  layer_keras("reshape", target_shape = c(dim(x_train)[-1], 1)) +
  layer_keras("conv_2d", filters = 16, kernel_size = 3, activation = "relu", padding = "same") +
  layer_keras("max_pooling_2d") +
  layer_keras("conv_2d", filters = 8, kernel_size = 3, activation = "relu", padding = "same") +
  layer_keras("max_pooling_2d", name = "encoding") +
  layer_keras("conv_2d", filters = 8, kernel_size = 3, activation = "relu", padding = "same") +
  layer_keras("upsampling_2d") +
  layer_keras("conv_2d", filters = 16, kernel_size = 3, activation = "relu", padding = "same") +
  layer_keras("upsampling_2d") +
  layer_keras("conv_2d", filters = 1, kernel_size = 3, activation = "sigmoid", padding = "same") +
  layer_keras("reshape", target_shape = dim(x_train)[-1])

ae <- autoencoder(net, loss = "binary_crossentropy")

model <- ae %>% train(x_train, validation_data = x_test, epochs = 4)

evaluate_mean_squared_error(model, x_test)
enc <- model %>% encode(x_test)
decode <- model %>% reconstruct(x_test)
plot_sample(x_test, decode, 1:10)
}

example2 <- function() {

  mnist <- dataset_mnist()
  x_train <- mnist$train$x / 255.0
  x_test <- mnist$test$x / 255.0
  x_train <- array_reshape(x_train, c(dim(x_train), 1))
  x_test <- array_reshape(x_test, c(dim(x_test), 1))

  net <- input() +
    conv(16, 3, max_pooling = 2, activation = "relu") +
    conv(8, 3, max_pooling = 2, activation = "relu") +
    conv(8, 3, upsampling = 2, activation = "relu") +
    conv(16, 3, upsampling = 2, activation = "relu") +
    conv(1, 3, activation = "sigmoid")
  ae <- autoencoder(net, loss = "binary_crossentropy")

  model <- ae %>% train(x_train, validation_data = x_test, epochs = 4)

  evaluate_mean_squared_error(model, x_test)
  enc <- model %>% encode(x_test)
  decode <- model %>% reconstruct(x_test)
}
