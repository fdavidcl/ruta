library(magrittr)
library(keras)
library(ruta)

baseline <- list()
baseline$learner <- autoencoder(
  input() + dense(16, "relu") + output("sigmoid"),
  loss = "binary_crossentropy"
) #%>% add_penalty_entropy(weight = .2)

baseline$model <- train(
  baseline$learner,
  x_train,
  epochs = 10,
  optimizer = "rmsprop",
  batch_size = 32,
  metrics = list("mean_squared_error")
)

baseline$enc <- baseline$model %>% encode(x_test)
baseline$decode <- baseline$model %>% reconstruct(x_test)
plot_sample(x_test, baseline$enc, baseline$decode, 11:20)
evaluate_mean_squared_error(baseline$model, x_test) # => 0.00871023
