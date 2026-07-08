///|
fn assert_close(actual : Double, expected : Double) -> Unit {
  if (actual - expected).abs() >= 1.0e-6 {
    abort("values are not close enough")
  }
}

///|
fn assert_array_close(actual : Array[Double], expected : Array[Double]) -> Unit {
  if actual.length() != expected.length() {
    abort("array lengths differ")
  }
  for i in 0..<actual.length() {
    assert_close(actual[i], expected[i])
  }
}

///|
test "sgd_step updates parameters with gradients" {
  let ctx = @tensor.AutogradContext::new()
  let p = @tensor.Tensor::parameter(ctx, [1.0, 2.0], [2])
  (p * @tensor.Tensor::from_array([2.0, -3.0], [2])).sum().backward()
  sgd_step([p], 0.1)
  assert_array_close(p.data(), [0.8, 2.3])
}

///|
test "AdamW step updates parameters and counts steps" {
  let ctx = @tensor.AutogradContext::new()
  let p = @tensor.Tensor::parameter(ctx, [1.0], [1])
  (p * @tensor.Tensor::from_array([2.0], [1])).sum().backward()
  let opt = AdamW::new_with_options([p], 0.1, 0.9, 0.999, 1.0e-8, 0.0)
  opt.step()
  @test.assert_eq(opt.step_count(), 1)
  assert_close(p.data()[0], 0.9000000005)
}
