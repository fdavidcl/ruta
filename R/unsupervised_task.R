#' Create a representation for an unsupervised learning task.Create a representation for a learning task.
#' @param id A string. A name for the task.
#' @param data A dataset, will be casted to data.frame with \code{as.data.frame}.
#' @param cl The index of the column corresponding to the class, or \code{NULL} if the dataset has no class.
#' @return An unsupervised task object with the provided data and parameters.
#' @export
ruta.makeUnsupervisedTask <- function(id, data, cl = NULL) {
  task <- ruta.makeTask(id, data, cl)
  class(task) <- c(class(task), rutaUnsupervisedTask)
  task
}
