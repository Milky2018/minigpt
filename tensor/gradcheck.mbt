///|
pub fn check_grad(
  values : Array[Double],
  shape : Array[Int],
  f : (Tensor) -> Tensor,
  eps? : Double,
  tol? : Double,
) -> Bool {
  let eps = eps.unwrap_or(1.0e-6)
  let tol = tol.unwrap_or(1.0e-4)
  let ctx = AutogradContext()
  let x = Tensor::parameter(ctx, values, shape)
  let y = f(x)
  y.backward()
  let analytic = match x.grad() {
    Some(g) => g.data()
    None => return false
  }
  for i in 0..<values.length() {
    let plus = values.copy()
    let minus = values.copy()
    plus[i] += eps
    minus[i] -= eps
    let yp = f(Tensor::from_array(plus, shape)).data()[0]
    let ym = f(Tensor::from_array(minus, shape)).data()[0]
    let numeric = (yp - ym) / (2.0 * eps)
    if (analytic[i] - numeric).abs() > tol {
      return false
    }
  }
  true
}
