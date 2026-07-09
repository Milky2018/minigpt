///|
pub fn Tensor::exp(self : Tensor) -> Tensor {
  let out = self.data.map(x => @math.exp(x))
  unary_tensor(self, out, self.shape, fn() {
    ExpBackward::{ input_data: self.data }
  })
}

///|
pub fn Tensor::log(self : Tensor) -> Tensor {
  let out = self.data.map(x => @math.ln(x))
  unary_tensor(self, out, self.shape, fn() {
    LogBackward::{ input_data: self.data }
  })
}

///|
pub fn Tensor::relu(self : Tensor) -> Tensor {
  let out = self.data.map(x => if x > 0.0 { x } else { 0.0 })
  unary_tensor(self, out, self.shape, fn() {
    ReluBackward::{ input_data: self.data }
  })
}

///|
const INV_SQRT_2 : Double = 0.7071067811865475

///|
const INV_SQRT_2PI : Double = 0.3989422804014327

///|
fn erf_approx(x : Double) -> Double {
  let sign = if x < 0.0 { -1.0 } else { 1.0 }
  let ax = x.abs()
  let t = 1.0 / (1.0 + 0.3275911 * ax)
  let y = 1.0 -
    (
      (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592
    ) *
    t *
    @math.exp(-ax * ax)
  sign * y
}

///|
fn normal_cdf(x : Double) -> Double {
  0.5 * (1.0 + erf_approx(x * INV_SQRT_2))
}

///|
fn normal_pdf(x : Double) -> Double {
  INV_SQRT_2PI * @math.exp(-0.5 * x * x)
}

///|
fn gelu_value(x : Double) -> Double {
  x * normal_cdf(x)
}

///|
pub fn Tensor::gelu(self : Tensor) -> Tensor {
  let out = self.data.map(x => gelu_value(x))
  unary_tensor(self, out, self.shape, fn() {
    GeluBackward::{ input_data: self.data }
  })
}
