# Conventions for coding ruta

## Function and variable naming

- Functions and variables will be named using camelCase. Exported functions will be prepended by `ruta.` to avoid name conflicts with other packages. Methods, however, will use common names (`print`, `train`, etc.).
- Arguments for functions will use the dot as a separator.

Example:

```r
#' @export
ruta.makeLearner <- function(x, arg.two, arg.three) {}

aNonExportedFunction <- function(x, ...) {}
```

## Classes

S3 classes are defined in `R/classes.r`. The use camelCase notation just like other variables.

## Comments

Comments inside code will start with `##` (thank ESS for that).

Documentation comments start with `#'`.

## Remaining

Follow [Hadley's style guide](http://stat405.had.co.nz/r-style.html) for rest of conventions.
