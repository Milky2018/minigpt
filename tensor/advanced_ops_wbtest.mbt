///|
test "matmul values and gradients" {
  let ctx = AutogradContext()
  let a = Tensor::parameter(ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let b = Tensor::parameter(ctx, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
  let c = a.matmul(b)
  @test.assert_eq(c.shape(), [2, 2])
  assert_array_close(c.data(), [58.0, 64.0, 139.0, 154.0])
  c.sum().backward()
  match a.grad() {
    Some(g) =>
      assert_array_close(g.data(), [15.0, 19.0, 23.0, 15.0, 19.0, 23.0])
    None => abort("expected a gradient")
  }
  match b.grad() {
    Some(g) => assert_array_close(g.data(), [5.0, 5.0, 7.0, 7.0, 9.0, 9.0])
    None => abort("expected b gradient")
  }
}

///|
test "swap axes and transpose2d propagate gradients" {
  let ctx = AutogradContext()
  let x = Tensor::parameter(ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let y = x.transpose2d()
  @test.assert_eq(y.shape(), [3, 2])
  assert_array_close(y.data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
  y.sum().backward()
  match x.grad() {
    Some(g) => assert_array_close(g.data(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    None => abort("expected x gradient")
  }
}

///|
test "exp log relu values and gradients" {
  let ctx = AutogradContext()
  let x = Tensor::parameter(ctx, [0.5, 2.0], [2])
  let y = x.log().exp().relu().sum()
  y.backward()
  match x.grad() {
    Some(g) => assert_array_close(g.data(), [1.0, 1.0])
    None => abort("expected x gradient")
  }
}

///|
test "gelu passes gradient check" {
  assert_true(check_grad([-1.0, -0.2, 0.4, 1.3], [4], fn(t) { t.gelu().sum() }))
}

///|
test "softmax normalizes rows and passes gradient check" {
  let x = Tensor::from_array([1.0, 2.0, 3.0, 2.0], [2, 2])
  let y = x.softmax(1)
  @test.assert_eq(y.shape(), [2, 2])
  let d = @math.exp(1.0) + @math.exp(2.0)
  assert_array_close(y.data(), [
    @math.exp(1.0) / d,
    @math.exp(2.0) / d,
    @math.exp(3.0) / (@math.exp(3.0) + @math.exp(2.0)),
    @math.exp(2.0) / (@math.exp(3.0) + @math.exp(2.0)),
  ])
  assert_true(
    check_grad([0.2, -0.4, 0.7, 1.3], [2, 2], fn(t) {
      (t.softmax(1) * Tensor::from_array([0.3, -0.2, 0.5, 0.1], [2, 2])).sum()
    }),
  )
}

///|
test "layer_norm normalizes last dimension and propagates gradients" {
  let ctx = AutogradContext()
  let x = Tensor::parameter(ctx, [1.0, 3.0, 2.0, 4.0], [2, 2])
  let weight = Tensor::parameter(ctx, [2.0, 3.0], [2])
  let bias = Tensor::parameter(ctx, [0.5, -0.5], [2])
  let y = x.layer_norm(weight, bias, 0.0)
  @test.assert_eq(y.shape(), [2, 2])
  assert_array_close(y.data(), [-1.5, 2.5, -1.5, 2.5])
  y.sum().backward()
  match weight.grad() {
    Some(g) => assert_array_close(g.data(), [-2.0, 2.0])
    None => abort("expected weight gradient")
  }
  match bias.grad() {
    Some(g) => assert_array_close(g.data(), [2.0, 2.0])
    None => abort("expected bias gradient")
  }
  match x.grad() {
    Some(g) => assert_array_close(g.data(), [0.0, 0.0, 0.0, 0.0])
    None => abort("expected input gradient")
  }
}

///|
test "layer_norm passes gradient check" {
  assert_true(
    check_grad([0.2, -0.4, 0.7, 1.3, 0.1, -0.8], [2, 3], fn(t) {
      t
      .layer_norm(
        Tensor::from_array([1.0, 1.5, 0.5], [3]),
        Tensor::from_array([0.1, -0.2, 0.3], [3]),
        1.0e-5,
      )
      .sum()
    }),
  )
}

///|
test "matmul passes gradient check" {
  assert_true(
    check_grad([1.0, 2.0, 3.0, 4.0], [2, 2], fn(t) {
      t.matmul(Tensor::from_array([2.0, -1.0, 0.5, 3.0], [2, 2])).sum()
    }),
  )
}

///|
test "randn creates parameter tensor" {
  let ctx = AutogradContext()
  let rng = @random.Rand::new()
  let t = Tensor::randn(ctx, [2, 3], rng)
  @test.assert_eq(t.shape(), [2, 3])
  @test.assert_eq(t.data().length(), 6)
  assert_true(t.requires_grad())
}
