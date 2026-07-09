///|
impl BackwardOp for AddBackward with fn backward(_self, grad, out_shape, inputs) {
  [
    { slot: 0, data: reduce_to_shape(grad, out_shape, inputs[0].shape) },
    { slot: 1, data: reduce_to_shape(grad, out_shape, inputs[1].shape) },
  ]
}

///|
impl BackwardOp for SubBackward with fn backward(_self, grad, out_shape, inputs) {
  [
    { slot: 0, data: reduce_to_shape(grad, out_shape, inputs[0].shape) },
    {
      slot: 1,
      data: reduce_to_shape(grad, out_shape, inputs[1].shape).map(x => -x),
    },
  ]
}

///|
impl BackwardOp for MulBackward with fn backward(self, grad, out_shape, inputs) {
  let lhs_shape = inputs[0].shape
  let rhs_shape = inputs[1].shape
  let lhs_grad = Array::make(shape_size(out_shape), 0.0)
  let rhs_grad = Array::make(shape_size(out_shape), 0.0)
  for i in 0..<grad.length() {
    let out_idx = unravel_index(i, out_shape)
    let li = broadcast_source_index(out_idx, lhs_shape)
    let ri = broadcast_source_index(out_idx, rhs_shape)
    lhs_grad[i] = grad[i] * self.rhs_data[ri]
    rhs_grad[i] = grad[i] * self.lhs_data[li]
  }
  [
    { slot: 0, data: reduce_to_shape(lhs_grad, out_shape, lhs_shape) },
    { slot: 1, data: reduce_to_shape(rhs_grad, out_shape, rhs_shape) },
  ]
}

///|
impl BackwardOp for DivBackward with fn backward(self, grad, out_shape, inputs) {
  let lhs_shape = inputs[0].shape
  let rhs_shape = inputs[1].shape
  let lhs_grad = Array::make(shape_size(out_shape), 0.0)
  let rhs_grad = Array::make(shape_size(out_shape), 0.0)
  for i in 0..<grad.length() {
    let out_idx = unravel_index(i, out_shape)
    let li = broadcast_source_index(out_idx, lhs_shape)
    let ri = broadcast_source_index(out_idx, rhs_shape)
    lhs_grad[i] = grad[i] / self.rhs_data[ri]
    rhs_grad[i] = grad[i] *
      -(self.lhs_data[li] / (self.rhs_data[ri] * self.rhs_data[ri]))
  }
  [
    { slot: 0, data: reduce_to_shape(lhs_grad, out_shape, lhs_shape) },
    { slot: 1, data: reduce_to_shape(rhs_grad, out_shape, rhs_shape) },
  ]
}

///|
impl BackwardOp for NegBackward with fn backward(_self, grad, out_shape, inputs) {
  [
    {
      slot: 0,
      data: reduce_to_shape(grad.map(x => -x), out_shape, inputs[0].shape),
    },
  ]
}

///|
impl BackwardOp for ExpBackward with fn backward(self, grad, out_shape, inputs) {
  let g = Array::make(grad.length(), 0.0)
  for i in 0..<g.length() {
    g[i] = grad[i] * @math.exp(self.input_data[i])
  }
  [{ slot: 0, data: reduce_to_shape(g, out_shape, inputs[0].shape) }]
}

///|
impl BackwardOp for LogBackward with fn backward(self, grad, out_shape, inputs) {
  let g = Array::make(grad.length(), 0.0)
  for i in 0..<g.length() {
    g[i] = grad[i] / self.input_data[i]
  }
  [{ slot: 0, data: reduce_to_shape(g, out_shape, inputs[0].shape) }]
}

///|
impl BackwardOp for ReluBackward with fn backward(self, grad, out_shape, inputs) {
  let g = Array::make(grad.length(), 0.0)
  for i in 0..<g.length() {
    g[i] = if self.input_data[i] > 0.0 { grad[i] } else { 0.0 }
  }
  [{ slot: 0, data: reduce_to_shape(g, out_shape, inputs[0].shape) }]
}

///|
impl BackwardOp for GeluBackward with fn backward(self, grad, out_shape, inputs) {
  let g = Array::make(grad.length(), 0.0)
  for i in 0..<g.length() {
    let x = self.input_data[i]
    let derivative = normal_cdf(x) + x * normal_pdf(x)
    g[i] = grad[i] * derivative
  }
  [{ slot: 0, data: reduce_to_shape(g, out_shape, inputs[0].shape) }]
}
