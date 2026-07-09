///|
impl BackwardOp for SumBackward with fn backward(
  _self,
  grad,
  _out_shape,
  inputs,
) {
  [{ slot: 0, data: Array::make(shape_size(inputs[0].shape), grad[0]) }]
}

///|
impl BackwardOp for ReshapeBackward with fn backward(
  _self,
  grad,
  _out_shape,
  inputs,
) {
  [{ slot: 0, data: Tensor::from_array(grad, inputs[0].shape).data() }]
}

///|
impl BackwardOp for SwapAxesBackward with fn backward(
  self,
  grad,
  out_shape,
  inputs,
) {
  let parent_shape = inputs[0].shape
  let parent_grad = Array::make(shape_size(parent_shape), 0.0)
  for i in 0..<grad.length() {
    let out_idx = unravel_index(i, out_shape)
    let in_idx = copy_ints(out_idx)
    in_idx[self.axis_a] = out_idx[self.axis_b]
    in_idx[self.axis_b] = out_idx[self.axis_a]
    parent_grad[ravel_index(in_idx, parent_shape)] = grad[i]
  }
  [{ slot: 0, data: parent_grad }]
}
