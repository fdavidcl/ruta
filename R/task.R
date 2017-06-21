#' Create a representation for a learning task.
#' @param id A string. A name for the task.
#' @param data A dataset, will be casted to data.frame with \code{as.data.frame}.
#' @param cl The index of the column corresponding to the class, or \code{NULL} if the dataset has no class.
#' @return A generic task object with the provided data and parameters.
#' @export
ruta.makeTask <- function(id, data, cl = NULL) {
  if (missing(id))
    id <- substitute(data)

  task <- list(id = id,
               data = as.data.frame(data),
               cl = cl)
  class(task) <- c(rutaTask)
  task
}

#' \code{print} method for tasks.
#' @param x \code{"rutaTask"} object
#' @param ... Ignored
#' @export
print.rutaTask <- function(x, ...) {
  type <- if (rutaUnsupervisedTask %in% class(x))
    "unsupervised"
  else
    "other"

  class <- if (!is.null(x$cl))
    paste0("Yes (", x$cl, ")")
  else
    "No"

  cat("# ruta Task: ", x$id, "\n",
      "# Type: ", type, "\n",
      "# Instances: ", length(x$data[[1]]), "\n",
      "# Features: ", length(x$data), "\n",
      "# Has class: ", class, "\n",
      sep = "")
}

taskToMxnet <- function(task) {
  if (is.null(task$cl))
    t(data.matrix(task$data))
  else
    t(data.matrix(task$data[-task$cl]))
}
