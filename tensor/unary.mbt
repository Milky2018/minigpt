///|
pub fn Tensor::exp(self : Tensor) -> Tensor {
  let out = self.data.map(x => @math.exp(x))
  unary_tensor(self, out, self.shape, parent => {
    Exp(parent, self.data, self.shape)
  })
}

///|
pub fn Tensor::log(self : Tensor) -> Tensor {
  let out = self.data.map(x => @math.ln(x))
  unary_tensor(self, out, self.shape, parent => {
    Log(parent, self.data, self.shape)
  })
}

///|
pub fn Tensor::relu(self : Tensor) -> Tensor {
  let out = self.data.map(x => if x > 0.0 { x } else { 0.0 })
  unary_tensor(self, out, self.shape, parent => {
    Relu(parent, self.data, self.shape)
  })
}

///|
fn gelu_value(x : Double) -> Double {
  let k = 0.7978845608028654
  let c = 0.044715
  0.5 * x * (1.0 + @math.tanh(k * (x + c * x * x * x)))
}

///|
pub fn Tensor::gelu(self : Tensor) -> Tensor {
  let out = self.data.map(x => gelu_value(x))
  unary_tensor(self, out, self.shape, parent => {
    Gelu(parent, self.data, self.shape)
  })
}
