context("test-layers.R")

test_that("layers can be joined", {
  expect_equal(length(input() + dense(1)), 2)
  expect_equal(length(input() + (dense(1) + output())), 3)
})

test_that("networks can be subsetted", {
  expect_equal(length((input() + dense(1))[1]), 1)
  expect_equal(length((input() + dense(1) + output())[2:3]), 2)
})

test_that("layers have correct classes", {
  expect_s3_class(dense(1), ruta_network)
  expect_s3_class(dense(1)[1], ruta_network)
  expect_s3_class((dense(1) + dense(1))[2], ruta_network)
})

test_that("networks convert to keras networks", {
  net <- input() + dense(1) + output()
  knet <- to_keras(net, 3)

  net2 <- input() + dense(2) + dense(1) + dense(2) + output()
  knet2 <- to_keras(net2, 3)

  expect_s3_class(knet, "keras.engine.training.Model")
  expect_s3_class(knet2, "keras.engine.training.Model")
  expect_equal(length(knet$layers), length(net))
  expect_equal(length(knet2$layers), length(net2))
})
