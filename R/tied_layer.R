DenseTied <- R6::R6Class("DenseTied",
  inherit = KerasLayer,

  public = list(
    output_dim = NULL,
    tied_to = NULL,
    tied_weights = NULL,
    activation = NULL,

    initialize = function(tied_to, activation) {
      self$tied_to <- tied_to
      self$tied_weights <- tied_to$weights
      self$output_dim <- self$tied_weights[[1]]$shape[0]
      self$activation <- activation
      if (is.character(self$activation)) {
        self$activation <- get(paste0("activation_", self$activation), pos = "package:keras")
      }
    },

    build = function(input_shape) {
      # self$kernel <- self$add_weight(
      #   name = 'kernel',
      #   shape = list(input_shape[[2]], self$output_dim),
      #   initializer = initializer_random_normal(),
      #   trainable = TRUE
      # )
    },

    call = function(x, mask = NULL) {
      # k_dot(x, self$kernel)
      # Return the transpose layer mapping using the explicit weight matrices
      output <- keras::k_dot(x - self$tied_weights[[2]], keras::k_transpose(self$tied_weights[[1]]))
      if (!is.null(self$activation)) {
        output <- self$activation(output)
      }
      output
    },

    compute_output_shape = function(input_shape) {
      list(input_shape[[1]], self$output_dim)
    }
  )
)

layer_dense_tied <- function(object, tied_to, name = NULL, activation = NULL) {
  create_layer(DenseTied, object, list(
    name = name,
    tied_to = tied_to,
    activation = activation,
    trainable = FALSE
  ))
}


mytest <- function() {
  test <- keras_model_sequential()
  encoder1 <-
    layer_dense(
      test,
      units = 100,
      input_shape = list(784)
    )
  encoder2 <-
    layer_dense(
      encoder1,
      units = 36,
      activation = "hard_sigmoid",
      name = "encoding"
    )
  decoder2 <-
    layer_dense_tied(encoder2, tied_to = encoder2)
  decoder1 <-
    layer_dense_tied(decoder2, tied_to = encoder1, activation = "hard_sigmoid")

  contractive_loss <-
    ruta:::contraction("binary_crossentropy") %>% ruta:::to_keras.ruta_loss_contraction(list(models = list(autoencoder = decoder)))
  compile(
    decoder1,
    optimizer = "rmsprop",
    loss = contractive_loss,
    metrics = list("mean_squared_error")
  )
  fit(decoder, x_train, x_train, batch_size = 32)
  decode <- predict(decoder, x_test)

  plot_sample(x_test, baseline$decode, decode, 11:20)
}
