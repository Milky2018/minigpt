///|
fn assert_close(actual : Double, expected : Double) -> Unit {
  if (actual - expected).abs() >= 1.0e-6 {
    abort("values are not close enough")
  }
}

///|
fn assert_double_array_eq(
  actual : Array[Double],
  expected : Array[Double],
) -> Unit {
  if actual.length() != expected.length() {
    abort("array lengths differ")
  }
  for i in 0..<actual.length() {
    assert_close(actual[i], expected[i])
  }
}

///|
test "embedding wraps tensor gather rows" {
  let weight = @tensor.Tensor::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
  let ids = @tensor.TokenIds([1, 2], [2])
  let out = embedding(weight, ids)
  @test.assert_eq(out.shape(), [2, 2])
  assert_double_array_eq(out.data(), [3.0, 4.0, 5.0, 6.0])
}

///|
test "cross_entropy wraps tensor loss" {
  let logits = @tensor.Tensor::from_array([0.0, 1.0, 2.0], [1, 3])
  let targets = @tensor.TokenIds([2], [1])
  let loss = cross_entropy(logits, targets)
  assert_close(loss.data()[0], 0.40760596444438046)
}
