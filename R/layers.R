#' Layer wrapper constructor
#'
#' Constructor function for layers. You shouldn't generally need to use this. Instead, consider
#' using individual functions such as \code{\link{dense}}.
#'
#' @param cl Character string specifying class of layer (e.g. \code{"ruta_layer_dense"}), which
#'   will be used to call the corresponding methods
#' @param ... Other parameters (usually `units`, `activation`)
#' @return A construct with class \code{"ruta_layer"}
#'
#' @examples
#' my_layer <- new_layer("dense", 30, "tanh")
#'
#' # Equivalent:
#' my_layer <- dense(30, "tanh")[[1]]
#' @export
new_layer <- function(cl, ...) {
  # length check
  stopifnot(is_scalar_vector(cl))

  # type coercion
  cl <- as.character(cl)

  structure(
    list(...),
    class = c(cl, ruta_layer)
  )
}

#' @rdname as_network
#' @export
as_network.ruta_layer <- function(x) {
  new_network(x)
}

make_atomic_network <- function(cl, ...) {
  as_network(
    new_layer(cl, ...)
  )
}

#' Create an input layer
#'
#' This layer acts as a placeholder for input data. The number of units is not
#' needed as it is deduced from the data during training.
#' @return A construct with class \code{"ruta_network"}
#' @family neural layers
#' @export
input <- function() {
  make_atomic_network(ruta_layer_input)
}

#' Convert Ruta layers onto Keras layers
#'
#' @param x The layer object
#' @param input_shape Number of features in training data
#' @param ... Unused
#' @return A Layer object from Keras
#' @export
to_keras.ruta_layer_input <- function(x, input_shape, ...) {
  keras::layer_input(shape = input_shape)
}

#' Create an output layer
#'
#' This layer acts as a placeholder for the output layer in an autoencoder. The
#' number of units is not needed as it is deduced from the data during training.
#' @param activation Optional, string indicating activation function (linear by default)
#' @return A construct with class \code{"ruta_network"}
#' @family neural layers
#' @export
output <- function(activation = "linear") {
  check_args_alt(formals(), environment())
  make_atomic_network(ruta_layer_dense, activation = activation)
}

#' Create a fully-connected neural layer
#'
#' Wrapper for a dense/fully-connected layer.
#' @param units Number of units
#' @param activation Optional, string indicating activation function (linear by default)
#' @return A construct with class \code{"ruta_network"}
#' @examples
#' dense(30, "tanh")
#' @family neural layers
#' @export
dense <- function(units, activation = arg_activation("linear")) {
  check_args_alt(formals(), environment())
  make_atomic_network(ruta_layer_dense, units = units, activation = activation)
}

#' @param model Keras model where the layer will be added
#' @rdname to_keras.ruta_layer_input
#' @export
to_keras.ruta_layer_dense <- function(x, input_shape, model = keras::keras_model_sequential(), ...) {
  if (is.null(x$units)) {
    x$units <- input_shape
  }

  act_reg = if (!is.null(x$activity_regularizer))
    to_keras(x$activity_regularizer, activation = x$activation)
  else
    NULL

  kern_reg = if (!is.null(x$kernel_regularizer))
    to_keras(x$kernel_regularizer)
  else
    NULL

  kern_ini <- list(...)$kernel_initializer

  if (is.null(kern_ini)) {
    kern_ini <-
      if (x$activation == "selu")
        "lecun_normal"
      else
        "glorot_uniform"
  }


  keras::layer_dense(
    model,
    units = x$units,
    activity_regularizer = act_reg,
    kernel_regularizer = kern_reg,
    name = if (is.null(x$name))
      NULL
    else
      paste0("pre_", x$name),
    kernel_initializer = kern_ini,
    ...
  ) %>%
    keras::layer_activation(activation = x$activation, name = x$name)

}


#' Create a convolutional layer
#'
#' Wrapper for a convolutional layer. The dimensions of the convolution operation are
#' inferred from the shape of the input data. This shape must follow the pattern
#' \code{(batch_shape, x, [y, [z, ]], channel)} where dimensions \code{y} and \code{z}
#' are optional, and \code{channel} will be either \code{1} for grayscale images or
#' generally \code{3} for colored ones.
#' @param filters Number of filters learned by the layer
#' @param kernel_size Integer or list of integers indicating the size of the weight
#'  matrices to be convolved with the image
#' @param padding One of "valid" or "same" (case-insensitive). See
#'  \code{\link[keras]{layer_conv_2d}} for more details
#' @param max_pooling \code{NULL} or an integer indicating the reduction ratio for a max
#'  pooling operation after the convolution
#' @param average_pooling \code{NULL} or an integer indicating the reduction ratio for
#'  an average pooling operation after the convolution
#' @param upsampling \code{NULL} or an integer indicating the augmentation ratio for an
#'  upsampling operation after the convolution
#' @param activation Optional, string indicating activation function (linear by default)
#' @return A construct with class \code{"ruta_network"}
#' @examples
#' # Sample convolutional autoencoder
#' net <- input() +
#'  conv(16, 3, max_pooling = 2, activation = "relu") +
#'  conv(8, 3, max_pooling = 2, activation = "relu") +
#'  conv(8, 3, upsampling = 2, activation = "relu") +
#'  conv(16, 3, upsampling = 2, activation = "relu") +
#'  conv(1, 3, activation = "sigmoid")
#' @family neural layers
#' @export
conv <- function(filters, kernel_size, padding = "same", max_pooling = NULL, average_pooling = NULL, upsampling = NULL, activation = "linear") {
  if (sum(map_lgl(list(max_pooling, average_pooling, upsampling), is.null)) < 2) {
    warning("More than one pooling or upsampling operation has been selected in this layer.")
  }

  make_atomic_network(
    ruta_layer_conv,
    filters = filters,
    kernel_size = kernel_size,
    padding = padding,
    activation = activation,
    max_pooling = max_pooling,
    average_pooling = average_pooling,
    upsampling = upsampling
  )
}

#' @rdname to_keras.ruta_layer_input
#' @export
to_keras.ruta_layer_conv <- function(x, input_shape, model = keras::keras_model_sequential(), ...) {
  dm <- model$shape$ndims - 2 # shape minus batch size and channel dimension

  if (dm == 0) {
    stop("Not enough dimensions provided for a convolutional operation. Required shape: (batch_size, dim1, [dim2, [dim3, ]], channel). Provided shape: (", paste0(model$shape, collapse = ", "), ")")
  }
  if (dm > 3) {
    stop("Too many dimensions provided for a convolutional operation. Required shape: (batch_size, dim1, [dim2, [dim3, ]], channel). Provided shape: (", paste0(model$shape, collapse = ", "), ")")
  }

  act_reg = if (!is.null(x$activity_regularizer))
    to_keras(x$activity_regularizer, activation = x$activation)
  else
    NULL

  kern_reg = if (!is.null(x$kernel_regularizer))
    to_keras(x$kernel_regularizer)
  else
    NULL

  kern_ini <- list(...)$kernel_initializer

  if (is.null(kern_ini)) {
    kern_ini <-
      if (x$activation == "selu")
        "lecun_normal"
    else
      "glorot_uniform"
  }

  layer_f <- switch (dm,
                     keras::layer_conv_1d,
                     keras::layer_conv_2d,
                     keras::layer_conv_3d)
  tensor <- layer_f(model,
          filters = x$filters,
          kernel_size = x$kernel_size,
          activity_regularizer = act_reg,
          kernel_regularizer = kern_reg,
          name = if (is.null(x$name))
            NULL
          else
            paste0("pre_", x$name),
          kernel_initializer = kern_ini,
          padding = x$padding,
          ...
  ) %>%
    keras::layer_activation(activation = x$activation, name = if (is.null(x$max_pooling) && is.null(x$average_pooling) && is.null(x$upsampling)) x$name else NULL)

  if (!is.null(x$max_pooling)) {
    switch(dm,
           keras::layer_max_pooling_1d,
           keras::layer_max_pooling_2d,
           keras::layer_max_pooling_3d)(tensor, pool_size = x$max_pooling, name = x$name)
  } else if (!is.null(x$average_pooling)) {
    switch(dm,
           keras::layer_average_pooling_1d,
           keras::layer_average_pooling_2d,
           keras::layer_average_pooling_3d)(tensor, pool_size = x$average_pooling, name = x$name)
  } else if (!is.null(x$upsampling)) {
    switch(dm,
           keras::layer_upsampling_1d,
           keras::layer_upsampling_2d,
           keras::layer_upsampling_3d)(tensor, size = x$upsampling, name = x$name)
  } else {
    tensor
  }
}

#' Custom layer from Keras
#'
#' Gets any layer available in Keras with the specified parameters
#'
#' @param type The name of the layer, e.g. `"activity_regularization"` for a
#'   `keras::layer_activity_regularization` object
#' @param ... Named parameters for the Keras layer constructor
#' @return A wrapper for the specified layer, which can be combined with other Ruta
#'   layers
#' @family neural layers
#' @export
layer_keras <- function(type, ...) {
  make_atomic_network(ruta_layer_custom, type = type, params = list(...))
}

#' Dropout layer
#'
#' Randomly sets a fraction `rate` of input units to 0 at each update during training
#' time, which helps prevent overfitting.
#' @param rate The fraction of affected units
#' @return A construct of class `"ruta_network"`
#' @family neural layers
#' @export
dropout <- function(rate = 0.5) {
  layer_keras("dropout", rate = rate)
}

#' @rdname to_keras.ruta_layer_input
#' @export
to_keras.ruta_layer_custom <- function(x, input_shape, model = keras::keras_model_sequential(), ...) {
  layer_f = get_keras_object(x$type, "layer")
  args = c(list(object = model), x$params)
  do.call(layer_f, args)
}
