# type : string, ...
ruta.makeLearner <- function(cl, id = cl, ...) {
    # Just find the appropriate function
    fname <- paste0("ruta::ruta.makeLearner.", type)
    if (exists(fname) && class(get(fname)) == "function") {
        get(fname)(...)
    } else {
        stop(paste0("No corresponding function found for ", type, " learner type"))
    }
}

ruta.train <- function(learner, task, subset) {
    if (class(learner) == "character")
        learner <- ruta.makeLearner(learner)

    if (!("ruta.learner" %in% class(learner)))
        stop("'learner' parameter is not of class 'ruta.learner'")

    
}
