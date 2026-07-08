///|
test "gather_rows values and gradients" {
  let ctx = AutogradContext::new()
  let weight = Tensor::parameter(ctx, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
  let ids = TokenIds::new([2, 0, 2], [3])
  let out = weight.gather_rows(ids)
  assert_int_array(out.shape(), [3, 2])
  assert_array_close(out.data(), [5.0, 6.0, 1.0, 2.0, 5.0, 6.0])
  out.sum().backward()
  match weight.grad() {
    Some(g) => assert_array_close(g.data(), [1.0, 1.0, 0.0, 0.0, 2.0, 2.0])
    None => abort("expected weight gradient")
  }
}

///|
test "cross_entropy value and gradients" {
  let ctx = AutogradContext::new()
  let logits = Tensor::parameter(ctx, [1.0, 2.0, 3.0, 3.0], [2, 2])
  let targets = TokenIds::new([1, 0], [2])
  let loss = logits.cross_entropy(targets)
  let expected = (
      @math.ln(@math.exp(1.0) + @math.exp(2.0)) -
      2.0 +
      @math.ln(@math.exp(3.0) + @math.exp(3.0)) -
      3.0
    ) /
    2.0
  assert_array_close(loss.data(), [expected])
  loss.backward()
  let d = @math.exp(1.0) + @math.exp(2.0)
  match logits.grad() {
    Some(g) =>
      assert_array_close(g.data(), [
        @math.exp(1.0) / d / 2.0,
        (@math.exp(2.0) / d - 1.0) / 2.0,
        -0.25,
        0.25,
      ])
    None => abort("expected logits gradient")
  }
}

///|
test "cross_entropy passes gradient check" {
  assert_true(
    check_grad([0.2, -0.4, 0.7, 1.3, 0.1, -0.8], [2, 3], fn(t) {
      t.cross_entropy(TokenIds::new([2, 0], [2]))
    }),
  )
}

///|
test "set_data_in_place updates parameter data" {
  let ctx = AutogradContext::new()
  let p = Tensor::parameter(ctx, [1.0, 2.0], [2])
  p.set_data_in_place([3.0, 4.0])
  assert_array_close(p.data(), [3.0, 4.0])
}
