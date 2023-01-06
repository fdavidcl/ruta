library(ruta)
## wdbc dataset
library(foreign)
wdbc = read.arff("../wdbc.arff")

ae = ruta.makeLearner("autoencoder", hidden = c(30, 2, 30))
task = ruta.makeUnsupervisedTask(data = wdbc, cl = 1)

model = train(ae, task,
              epochs = 50,
              learning.rate = 0.01,
              momentum = 0.9,
              optimizer = "adagrad")

dfeat <- ruta.layerOutputs(model, task, layerOutput = 2, array.layout = "colmajor")
dfeat

## toy dataset
invent = data.frame(
  class = c(0, 0, 1, 0, 0, 1, 0, 1, 1, 1),
  f1    = c(0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
  f2    = c(1, 1, 0, 0, 1, 1, 0, 0, 1, 1),
  f3    = c(1, 1, 1, 1, 0, 0, 0, 0, 1, 1)
)

## resultado interesante: el autoencoder decide intercambiar los unos y ceros en la capa
## primera, y recuperarlos en la de salida
## ae = ruta.makeLearner("autoencoder", hidden = c(3, 3), activation = "sigmoid")
## entrenado con momentum 0 y learning.rate 0.3
ae = ruta.makeLearner("autoencoder", hidden = c(3, 2, 3))
task = ruta.makeUnsupervisedTask(data = invent, cl = 1)

model = train(ae, task,
              epochs = 1000,
              learning.rate = 0.2,
              momentum = 0,
              optimizer = "adagrad")

dfeat <- ruta.layerOutputs(model, task, layerOutput = 2, array.layout = "colmajor")
dfeat

## toy dataset 2
toy2 = data.frame(
  f1    = c(1, 2, 0, 9, 4, 1, 3, 5, 6, 1),
  f2    = c(1, 2, 0, 9, 4, 1, 3, 5, 6, 1),
  f3    = c(1, 2, 0, 9, 4, 1, 3, 5, 6, 1)
)
ae = ruta.makeLearner("autoencoder", hidden = c(3, 1, 3))
task = ruta.makeUnsupervisedTask(data = toy2)

model = train(ae, task,
              epochs = 100,
              learning.rate = 0.2,
              momentum = 0,
              optimizer = "adagrad")

dfeat <- ruta.layerOutputs(model, task, layerOutput = 3, array.layout = "colmajor")
dfeat

## toy dataset 3: successor
toy3 = data.frame(
  f1    = c(1, 2, 0, 9,  4, 1, 3, 5, 6, 1),
  f2    = c(2, 3, 1, 10, 5, 2, 4, 6, 7, 2),
  f3    = c(3, 4, 2, 11, 6, 3, 5, 7, 8, 3)
)
ae = ruta.makeLearner("autoencoder", hidden = c(3, 1, 3), activation = "sigmoid")
task = ruta.makeUnsupervisedTask(data = toy3)

model = train(ae, task,
              epochs = 500,
              learning.rate = 0.1,
              momentum = 1,
              optimizer = "adagrad")

dfeat <- ruta.layerOutputs(model, task, layerOutput = 3, array.layout = "colmajor")
dfeat

## toy dataset 4: sum
toy4 = data.frame(
  f1    = c(1, 2, 0, 9,  4, 1, 3, 4,  6, 2),
  f2    = c(2, 3, 1, 1,  5, 4, 4, 6,  1, 2),
  f3    = c(3, 5, 1, 10, 9, 5, 7, 10, 7, 4)
)
suma_test = data.frame(
  # test con ejemplos que el autoencoder no conoce
  f1    = c(1, 4, 1),
  f2    = c(2, 4, 1),
  f3    = c(3, 8, 2)
)
ae = ruta.makeLearner("autoencoder", hidden = c(3, 2, 3), sparse = T, activation = "sigmoid")
task = ruta.makeUnsupervisedTask(data = toy4)
mxnet::mx.model.FeedForward.create(
  symbol = ae$nn,
  X = t(task$data),
  y = t(task$data),
  num.round = 500,
  eval.metric = mx.metric.rmse,
  array.layout = "colmajor",
  optimizer = "sgd",
  aux.params = list(
    "aelayer1kl_moving_avg" = mxnet::mx.nd.ones(3) * 0.5,
    "aelayer2kl_moving_avg" = mxnet::mx.nd.ones(2) * 0.5,
    "aelayer3kl_moving_avg" = mxnet::mx.nd.ones(3) * 0.5
  )
)
model = train(ae, task,
              epochs = 500,
              learning.rate = 0.1,
              momentum = 1,
              optimizer = "adagrad")

dfeat <- ruta.layerOutputs(model, task, layerOutput = 3, array.layout = "colmajor")
dfeat
# parece que aprende que la 3ª columna es la suma de las otras dos, incluso cuando
# la codificación es en una variable:
dfeat[,3] - dfeat[,1] - dfeat[,2]

# ha aprendido a codificar la suma? (spoiler: probablemente)
test_task = ruta.makeUnsupervisedTask(data = suma_test)
dfeat <- ruta.layerOutputs(model, test_task, layerOutput = 3, array.layout = "colmajor")
round(dfeat)

## toy dataset 5: product (non-linear dependency)
toy5 = data.frame(
  f1    = c(1, 2, 0, 9,  4, 1,  3,  4, 6, 2),
  f2    = c(2, 3, 1, 1,  5, 4,  4,  6, 1, 2),
  f3    = c(2, 6, 0, 9, 20, 4, 12, 24, 6, 4)
)
prod_test = data.frame(
  # test con ejemplos que el autoencoder no conoce
  f1    = c(1,  4, 1, 3, 4),
  f2    = c(2,  4, 1, 3, 2),
  f3    = c(2, 16, 1, 9, 8)
)
ae = ruta.makeLearner("autoencoder", hidden = c(3, 2, 3))
task = ruta.makeUnsupervisedTask(data = toy5)

model = train(ae, task,
              epochs = 500,
              learning.rate = 0.2,
              momentum = 1,
              optimizer = "adagrad")

dfeat <- ruta.layerOutputs(model, task, layerOutput = 3, array.layout = "colmajor")
dfeat

# ha aprendido a codificar el producto? (spoiler: puede?)
test_task = ruta.makeUnsupervisedTask(data = prod_test)
dfeat <- ruta.layerOutputs(model, test_task, layerOutput = 3, array.layout = "colmajor")
dfeat
