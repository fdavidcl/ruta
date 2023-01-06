library(ruta)
#===============================================================================
mx.model.FeedForward.partial <-
  function(symbol, X, y=NULL, ctx=NULL, begin.round=1,
           num.round=10, optimizer="sgd",
           initializer=mxnet:::mx.init.uniform(0.01),
           eval.data=NULL, eval.metric=NULL,
           epoch.end.callback=NULL, batch.end.callback=NULL,
           array.batch.size=128, array.layout="auto",
           kvstore="local",
           verbose=TRUE,
           arg.params=NULL, aux.params=NULL,
           allow.extra.params = FALSE
           ...) {
    if (is.array(X) || is.matrix(X)) {
      if (array.layout == "auto") {
        array.layout <- mxnet:::mx.model.select.layout.train(X, y)
      }
      if (array.layout == "rowmajor") {
        X <- t(X)
      }
    }
    X <- mxnet:::mx.model.init.iter(X, y, batch.size=array.batch.size, is.train=TRUE)
    if (!X$iter.next()) {
      X$reset()
      if (!X$iter.next()) stop("Empty input")
    }
    input.shape <- dim((X$value())$data)
    params <- mxnet:::mx.model.init.params(symbol, input.shape, initializer, mxnet:::mx.cpu())
    if (!is.null(arg.params)) params$arg.params <- arg.params
    if (!is.null(aux.params)) params$aux.params <- aux.params
    if (is.null(ctx)) ctx <- mxnet:::mx.ctx.default()
    if (is.mxnet:::mx.context(ctx)) {
      ctx <- list(ctx)
    }
    if (!is.list(ctx)) stop("ctx must be mxnet:::mx.context or list of mxnet:::mx.context")
    if (is.character(optimizer)) {
      ndim <- length(input.shape)
      batchsize = input.shape[[ndim]]
      optimizer <- mxnet:::mx.opt.create(optimizer, rescale.grad=(1/batchsize), ...)
    }
    if (!is.null(eval.data) && !is.list(eval.data) && !is.mxnet:::mx.dataiter(eval.data)) {
      stop("The validation set should be either a mxnet:::mx.io.DataIter or a R list")
    }
    if (is.list(eval.data)) {
      if (is.null(eval.data$data) || is.null(eval.data$label)){
        stop("Please provide the validation set as list(data=R.array, label=R.array)")
      }
      if (is.array(eval.data$data) || is.matrix(eval.data$data)) {
        if (array.layout == "auto") {
          array.layout <- mxnet:::mx.model.select.layout.train(eval.data$data, eval.data$label)
        }
        if (array.layout == "rowmajor") {
          eval.data$data <- t(eval.data$data)
        }
      }
      eval.data <- mxnet:::mx.model.init.iter(eval.data$data, eval.data$label, batch.size=array.batch.size, is.train = TRUE)
    }
    kvstore <- mxnet:::mx.model.create.kvstore(kvstore, params$arg.params, length(ctx), verbose=verbose)
    model <- mxnet:::mx.model.train(symbol, ctx, input.shape,
                            params$arg.params, params$aux.params,
                            begin.round, num.round, optimizer=optimizer,
                            train.data=X, eval.data=eval.data,
                            metric=eval.metric,
                            epoch.end.callback=epoch.end.callback,
                            batch.end.callback=batch.end.callback,
                            kvstore=kvstore,
                            verbose=verbose)
    return (model)
  }
#===============================================================================
bindExec <- function(symbol, ctx, grad.req = "null", ...) {
  if (!is.MXSymbol(symbol))
    stop("symbol need to be MXSymbol")

  slist <- symbol$infer.shape(list(...))

  if (is.null(slist)) {
    stop("Need more shape information to decide the shapes of arguments")
  }
  arg.arrays <- sapply(slist$arg.shapes, function(shape) {
    mxnet:::mx.nd.zeros(shape, ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  aux.arrays <- sapply(slist$aux.shapes, function(shape) {
    mxnet:::mx.nd.zeros(shape, ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  grad.reqs <- lapply(names(slist$arg.shapes), function(nm) {
    if (!mxnet:::mx.util.str.endswith(nm, "label") && !mxnet:::mx.util.str.endswith(nm, "data")) {
      grad.req
    } else {
      "null"
    }
  })
  exec <- mxnet:::mx.symbol.bind(symbol, ctx,
                 arg.arrays=arg.arrays,
                 aux.arrays=aux.arrays,
                 grad.reqs = grad.reqs)
}
predictInternal <- function(model, X, ctx=NULL, layer, array.batch.size=128, array.layout="auto") {
  # initialization stuff I probably don't care about ---------------------------
  if (is.null(ctx)) ctx <- mxnet:::mx.ctx.default()
  if (is.array(X) || is.matrix(X)) {
    if (array.layout == "auto") {
      array.layout <- mxnet:::mx.model.select.layout.predict(X, model)
    }
    if (array.layout == "rowmajor") {
      X <- t(X)
    }
  }
  # end initialization ---------------------------------------------------------
  # iterator creation ----------------------------------------------------------
  ## X iterates through the batches of input data
  X <- mxnet:::mx.model.init.iter(X, NULL, batch.size=array.batch.size, is.train=FALSE)
  X$reset()
  if (!X$iter.next()) stop("Cannot predict on empty iterator")
  dlist = X$value()
  # end iterator creation ------------------------------------------------------
  # executor creation ----------------------------------------------------------
  ## mx.simple.bind defined in https://github.com/dmlc/mxnet/blob/master/R-package/R/executor.R#L5
  ## internally calls mx.symbol.bind, defined in https://github.com/dmlc/mxnet/blob/master/R-package/src/executor.cc#L191
  ### see also: https://github.com/dmlc/mxnet/blob/e7514fe1b3265aaf15870b124bb6ed0edd82fa76/R-package/demo/basic_executor.R
  # pexec <- mxnet:::mx.simple.bind(model$symbol, ctx=ctx, data=dim(dlist$data), grad.req="null")
  internals <- model$symbol$get.internals()
  pexec <- mxnet:::mx.simple.bind(internals[[1 + 3 * layer]], ctx=ctx, data=dim(dlist$data), grad.req="null")
  # end executor creation ------------------------------------------------------
  # set up arg arrays ----------------------------------------------------------
  internalArgParams <- model$arg.params[1:(2 * layer)]
  # print(names(internalArgParams))

  mxnet:::mx.exec.update.arg.arrays(pexec, internalArgParams, match.name=T)
  mxnet:::mx.exec.update.aux.arrays(pexec, model$aux.params, match.name=T)
  # end set up arg arrays ------------------------------------------------------
  # the rest is left untouched -------------------------------------------------
  packer <- mxnet:::mx.nd.arraypacker()
  X$reset()
  while (X$iter.next()) {
    dlist = X$value()
    mxnet:::mx.exec.update.arg.arrays(pexec, list(data=dlist$data), match.name=T)
    mxnet:::mx.exec.forward(pexec, is.train=FALSE)
    out.pred <- mxnet:::mx.nd.copyto(pexec$ref.outputs[[1]], mxnet:::mx.cpu())
    padded <- X$num.pad()
    oshape <- dim(out.pred)
    ndim <- length(oshape)
    packer$push(mxnet:::mx.nd.slice(out.pred, 0, oshape[[ndim]] - padded))
  }
  X$reset()
  return(packer$get())
}
#===============================================================================
ae = ruta.makeLearner("autoencoder", hidden = c(4, 3, 2, 3, 4), activation = "tanh")
tiris = ruta.makeUnsupervisedTask(data = iris, cl = 5)
trainX = t(data.matrix(tiris$data[-tiris$cl]))

model = train(ae, tiris, epochs = 20)

# int = ae$nn$get.internals()
int = model$symbol$get.internals()

n = mx.symbol.FullyConnected(data = a, name = "innermost", num_hidden = 3)
na= mx.symbol.Activation(data = n, act.type = "tanh", name = "innermostact")

predictInternal(model, trainX[1:4,1:2], layer = 1, array.layout = "colmajor", array.batch.size = 1)
predict(model, trainX[1:4,1:2], array.layout = "colmajor", array.batch.size = 1)
#===============================================================================
library(mxnet)
a = mx.symbol.Variable("data")
c = mx.symbol.sum(data = a, keepdims = F, name = "sum")
d = mx.symbol.arccos(data = c, name = "arccos")
reg = mx.symbol.SoftmaxOutput(data = d, name = "sm")

inputs = t(data.matrix(data.frame(a = c(1, 2, -1), b = c(0, -2, 0))))
outputs = t(data.matrix(data.frame(out = c(0, 1.57, 3.14))))

mx.model.FeedForward.create(reg, X = inputs, y = outputs, array.layout = "colmajor", eval.metric = mx.metric.rmse, learning.rate = 0.01)

arra = mx.nd.array(as.array(c(1, 2)), mx.cpu())
arrb = mx.nd.array(as.array(c(0, -2)), mx.cpu())
exec = mxnet:::mx.symbol.bind(
  symbol=d,
  ctx=mx.cpu(),
  arg.arrays = list(inputa = arra, inputb = arrb),
  aux.arrays = list(),
  grad.reqs = list("null", "null"))

mx.exec.forward(exec)
out = as.array(exec$outputs[[1]])
print(out)

internals = exec$get.internals()
