///|
pub fn Tensor::randn(
  ctx : AutogradContext,
  shape : Array[Int],
  rng : @random.Rand,
) -> Tensor {
  let n = shape_size(shape)
  let data = Array::make(n, 0.0)
  let mut i = 0
  while i < n {
    let u1 = rng.double().max(1.0e-12)
    let u2 = rng.double()
    let r = (-2.0 * @math.ln(u1)).sqrt()
    let theta = 2.0 * @math.PI * u2
    data[i] = r * @math.cos(theta)
    if i + 1 < n {
      data[i + 1] = r * @math.sin(theta)
    }
    i += 2
  }
  Tensor::parameter(ctx, data, shape)
}
