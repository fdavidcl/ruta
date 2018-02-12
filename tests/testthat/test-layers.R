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
  expect_s3_class(dense(1), rutaNetwork)
  expect_s3_class(dense(1)[1], rutaNetwork)
  expect_s3_class((dense(1) + dense(1))[2], rutaNetwork)
})

test_that("networks convert to keras networks", {
  net = input() + dense(1) + output()
  net2 = input() + dense(2) + dense(1) + dense(2) + output()
  expect_equal(length(toKeras(net, 3)), length(net))
  expect_equal(length(toKeras(net2, 3)), length(net2))
})
