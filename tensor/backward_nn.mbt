///|
impl BackwardOp for GatherRowsBackward with fn backward(
  self,
  grad,
  _out_shape,
  inputs,
) {
  let weight_shape = inputs[0].shape
  let width = weight_shape[1]
  let weight_grad = Array::make(shape_size(weight_shape), 0.0)
  for i in 0..<self.ids.length() {
    let row = self.ids[i]
    for d in 0..<width {
      weight_grad[row * width + d] += grad[i * width + d]
    }
  }
  [{ slot: 0, data: weight_grad }]
}

///|
impl BackwardOp for CrossEntropyBackward with fn backward(
  self,
  grad,
  _out_shape,
  inputs,
) {
  let logits_shape = inputs[0].shape
  let vocab = logits_shape[logits_shape.length() - 1]
  let scale = grad[0] / self.targets.length().to_double()
  let logits_grad = self.probs.copy()
  for i in 0..<self.targets.length() {
    logits_grad[i * vocab + self.targets[i]] -= 1.0
  }
  for i in 0..<logits_grad.length() {
    logits_grad[i] *= scale
  }
  [{ slot: 0, data: logits_grad }]
}
