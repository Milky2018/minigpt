///|
impl BackwardOp for MatmulBackward with fn backward(
  self,
  grad,
  out_shape,
  inputs,
) {
  let lhs_shape = inputs[0].shape
  let rhs_shape = inputs[1].shape
  let k = lhs_shape[lhs_shape.length() - 1]
  let lhs_grad = Array::make(shape_size(lhs_shape), 0.0)
  let rhs_grad = Array::make(shape_size(rhs_shape), 0.0)
  for i in 0..<grad.length() {
    let out_idx = unravel_index(i, out_shape)
    let row = out_idx[out_idx.length() - 2]
    let col = out_idx[out_idx.length() - 1]
    for inner in 0..<k {
      let lhs_idx = out_idx[0:out_idx.length() - 2].to_owned()
      lhs_idx.push(row)
      lhs_idx.push(inner)
      let rhs_idx = out_idx[0:out_idx.length() - 2].to_owned()
      rhs_idx.push(inner)
      rhs_idx.push(col)
      let li = broadcast_source_index(lhs_idx, lhs_shape)
      let ri = broadcast_source_index(rhs_idx, rhs_shape)
      lhs_grad[li] += grad[i] * self.rhs_data[ri]
      rhs_grad[ri] += grad[i] * self.lhs_data[li]
    }
  }
  [{ slot: 0, data: lhs_grad }, { slot: 1, data: rhs_grad }]
}

///|
impl BackwardOp for SoftmaxBackward with fn backward(
  self,
  grad,
  out_shape,
  _inputs,
) {
  let g = Array::make(grad.length(), 0.0)
  let outer = shape_size(out_shape[0:self.axis].to_owned())
  let dim = out_shape[self.axis]
  let inner = shape_size(out_shape[self.axis + 1:].to_owned())
  for o in 0..<outer {
    for i in 0..<inner {
      let base = o * dim * inner + i
      let mut dot = 0.0
      for d in 0..<dim {
        dot += grad[base + d * inner] * self.output_data[base + d * inner]
      }
      for d in 0..<dim {
        let idx = base + d * inner
        g[idx] = self.output_data[idx] * (grad[idx] - dot)
      }
    }
  }
  [{ slot: 0, data: g }]
}

///|
impl BackwardOp for LayerNormBackward with fn backward(
  self,
  grad,
  _out_shape,
  inputs,
) {
  let input_shape = inputs[0].shape
  let hidden = input_shape[input_shape.length() - 1]
  let rows = shape_size(input_shape) / hidden
  let input_grad = Array::make(shape_size(input_shape), 0.0)
  let weight_grad = Array::make(hidden, 0.0)
  let bias_grad = Array::make(hidden, 0.0)
  for row in 0..<rows {
    let base = row * hidden
    let mut mean = 0.0
    for i in 0..<hidden {
      mean += self.input_data[base + i]
    }
    mean = mean / hidden.to_double()
    let mut variance = 0.0
    for i in 0..<hidden {
      let centered = self.input_data[base + i] - mean
      variance += centered * centered
    }
    variance = variance / hidden.to_double()
    let inv_std = 1.0 / (variance + self.eps).sqrt()
    let xhat = Array::make(hidden, 0.0)
    let dxhat = Array::make(hidden, 0.0)
    let mut sum_dxhat = 0.0
    let mut sum_dxhat_xhat = 0.0
    for i in 0..<hidden {
      xhat[i] = (self.input_data[base + i] - mean) * inv_std
      dxhat[i] = grad[base + i] * self.weight_data[i]
      sum_dxhat += dxhat[i]
      sum_dxhat_xhat += dxhat[i] * xhat[i]
      weight_grad[i] += grad[base + i] * xhat[i]
      bias_grad[i] += grad[base + i]
    }
    for i in 0..<hidden {
      input_grad[base + i] = inv_std *
        (hidden.to_double() * dxhat[i] - sum_dxhat - xhat[i] * sum_dxhat_xhat) /
        hidden.to_double()
    }
  }
  [
    { slot: 0, data: input_grad },
    { slot: 1, data: weight_grad },
    { slot: 2, data: bias_grad },
  ]
}
