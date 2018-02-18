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

## Classes

S3 classes are defined as string constants in `R/classes.r`. The use snake_case notation just like other variables.

## Remaining

This package will attempt to adhere to the [tidyverse style guide](http://style.tidyverse.org/) as closely as possible.
