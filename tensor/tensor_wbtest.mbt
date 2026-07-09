///|
fn assert_array_close(
  actual : Array[Double],
  expected : Array[Double],
) -> Unit raise {
  @test.assert_eq(actual.length(), expected.length())
  for i in 0..<actual.length() {
    assert_true((actual[i] - expected[i]).abs() < 1.0e-9)
  }
}

///|
test "tensor validates shape and stores copies" {
  let t = Tensor::from_array([1.0, 2.0, 3.0, 4.0], [2, 2])
  @test.assert_eq(t.shape(), [2, 2])
  assert_array_close(t.data(), [1.0, 2.0, 3.0, 4.0])
}

///|
test "token ids validate shape" {
  let ids = TokenIds([1, 2, 3, 4], [2, 2])
  @test.assert_eq(ids.shape(), [2, 2])
  @test.assert_eq(ids.data(), [1, 2, 3, 4])
}

///|
test "elementwise add broadcasts values" {
  let a = Tensor::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let b = Tensor::from_array([10.0, 20.0, 30.0], [3])
  let c = a + b
  @test.assert_eq(c.shape(), [2, 3])
  assert_array_close(c.data(), [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
}

///|
test "elementwise mul broadcasts gradients" {
  let ctx = AutogradContext()
  let a = Tensor::parameter(ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let b = Tensor::parameter(ctx, [10.0, 20.0, 30.0], [3])
  let loss = (a * b).sum()
  loss.backward()
  match a.grad() {
    Some(g) =>
      assert_array_close(g.data(), [10.0, 20.0, 30.0, 10.0, 20.0, 30.0])
    None => abort("expected a gradient")
  }
  match b.grad() {
    Some(g) => assert_array_close(g.data(), [5.0, 7.0, 9.0])
    None => abort("expected b gradient")
  }
}

///|
test "add sub div neg and mean produce gradients" {
  let ctx = AutogradContext()
  let x = Tensor::parameter(ctx, [2.0, 4.0], [2])
  let y = Tensor::parameter(ctx, [1.0, 2.0], [2])
  let loss = ((x + y - -y) / Tensor::scalar(2.0)).mean()
  loss.backward()
  match x.grad() {
    Some(g) => assert_array_close(g.data(), [0.25, 0.25])
    None => abort("expected x gradient")
  }
  match y.grad() {
    Some(g) => assert_array_close(g.data(), [0.5, 0.5])
    None => abort("expected y gradient")
  }
}

///|
test "reshape preserves values and propagates gradients" {
  let ctx = AutogradContext()
  let x = Tensor::parameter(ctx, [1.0, 2.0, 3.0, 4.0], [2, 2])
  let y = x.reshape([4])
  @test.assert_eq(y.shape(), [4])
  assert_array_close(y.data(), [1.0, 2.0, 3.0, 4.0])
  y.sum().backward()
  match x.grad() {
    Some(g) => assert_array_close(g.data(), [1.0, 1.0, 1.0, 1.0])
    None => abort("expected x gradient")
  }
}

///|
test "clear graph preserves parameter node ids" {
  let ctx = AutogradContext()
  let x = Tensor::parameter(ctx, [2.0], [1])
  let _tmp = x + Tensor::scalar(1.0)
  let y = Tensor::parameter(ctx, [3.0], [1])
  ctx.clear_graph()
  (y * Tensor::scalar(2.0)).sum().backward()
  match y.grad() {
    Some(g) => assert_array_close(g.data(), [2.0])
    None => abort("expected y gradient")
  }
}
