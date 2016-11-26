# id: string, data: data.frame, target: string, positive: string (?)
ruta.makeTask <- function(id, data) {
    if (missing(id))
        id <- substitute(data)
    task <- list(id = id, data = as.data.frame(data))
    class(dataset) <- "ruta.task"
    task
}
