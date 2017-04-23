#' @export
ruta.makeUnsupervisedTask <- function(id, data, cl = NULL) {
  if (missing(id))
    id <- substitute(data)

  task <- list(id = id,
               data = as.data.frame(data),
               cl = cl)
  class(task) <- c(rutaTask, rutaUnsupervisedTask)
  task
}

print.rutaTask <- function(task) {
  type <- if (rutaUnsupervisedTask %in% class(task))
    "unsupervised"
  else
    "other"

  class <- if (!is.null(task$cl))
    paste0("Yes (", task$cl, ")")
  else
    "No"

  cat("# ruta Task: ", task$id, "\n",
      "# Type: ", type, "\n",
      "# Instances: ", length(task$data[[1]]), "\n",
      "# Features: ", length(task$data), "\n",
      "# Has class: ", class, "\n",
      sep = "")
}

taskToMxnet <- function(task) {
  if (is.null(task$cl))
    t(data.matrix(task$data))
  else
    t(data.matrix(task$data[-task$cl]))
}
