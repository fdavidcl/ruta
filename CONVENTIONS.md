# Conventions for coding ruta

## Naming

- Functions, variables and arguments will be named using snake_case.

    Example:

    ```r
    to_keras <- function(x, ...) UseMethod("to_keras")
    to_keras.ruta_network <- function(x, input_shape) {
      # ...
    }
    ```

- *Constructor* functions are named starting with nouns, e.g. `autoencoder_robust` or `contraction`. In particular, all autoencoder variants start with `autoencoder_`.
- *Modifier* functions (those which take an object and do things with it) are named with verbs, e.g. `make_denoising` or `train`.
- *Coercion* functions and methods (those which coerce a value to an object of some class) are named with `as_`, e.g. `as_network`.
- *Backend* methods are named with `to_`, e.g. `to_keras`. This will allow to support other backends in the future.

## Classes

S3 classes are defined as string constants in `R/classes.R`. The use snake_case notation just like other variables.

## Remaining

This package will attempt to adhere to the [tidyverse style guide](http://style.tidyverse.org/) as closely as possible.
