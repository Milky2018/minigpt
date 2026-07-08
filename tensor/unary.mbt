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
