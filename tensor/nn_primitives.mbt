///|
pub fn Tensor::gather_rows(self : Tensor, ids : TokenIds) -> Tensor {
  if self.shape.length() != 2 {
    abort("gather_rows requires a rank-2 weight tensor")
  }
  let vocab = self.shape[0]
  let width = self.shape[1]
  let out_shape = ids.shape()
  out_shape.push(width)
  let out = Array::make(shape_size(out_shape), 0.0)
  let id_data = ids.data()
  for i in 0..<id_data.length() {
    let row = id_data[i]
    if row < 0 || row >= vocab {
      abort("token id out of vocabulary range")
    }
    for d in 0..<width {
      out[i * width + d] = self.data[row * width + d]
    }
  }
  if self.requires_grad {
    match (self.context, self.node_id) {
      (Some(ctx), Some(id)) => {
        let node_id = ctx.push_node(
          out_shape,
          GatherRows(Some(id), id_data, self.shape),
        )
        {
          data: out,
          shape: out_shape,
          requires_grad: true,
          context: Some(ctx),
          node_id: Some(node_id),
        }
      }
      _ => abort("differentiable tensor operation requires an autograd context")
    }
  } else {
    Tensor::from_array(out, out_shape)
  }
}

///|
pub fn Tensor::cross_entropy(self : Tensor, targets : TokenIds) -> Tensor {
  if self.shape.length() < 1 {
    abort("cross_entropy requires logits with a vocabulary axis")
  }
  let vocab = self.shape[self.shape.length() - 1]
  if vocab <= 0 {
    abort("cross_entropy requires a non-empty vocabulary axis")
  }
  let target_data = targets.data()
  let expected_target_shape = self.shape[0:self.shape.length() - 1].to_owned()
  if targets.shape() != expected_target_shape {
    abort("cross_entropy target shape must match logits without the last axis")
  }
  let count = target_data.length()
  if count == 0 {
    abort("cross_entropy requires at least one target")
  }
  let probs = Array::make(self.data.length(), 0.0)
  let mut loss = 0.0
  for i in 0..<count {
    let target = target_data[i]
    if target < 0 || target >= vocab {
      abort("target id out of vocabulary range")
    }
    let base = i * vocab
    let mut max_logit = self.data[base]
    for v in 1..<vocab {
      max_logit = max_logit.max(self.data[base + v])
    }
    let mut sum_exp = 0.0
    for v in 0..<vocab {
      let value = @math.exp(self.data[base + v] - max_logit)
      probs[base + v] = value
      sum_exp += value
    }
    for v in 0..<vocab {
      probs[base + v] = probs[base + v] / sum_exp
    }
    loss += @math.ln(sum_exp) + max_logit - self.data[base + target]
  }
  loss = loss / count.to_double()
  if self.requires_grad {
    match (self.context, self.node_id) {
      (Some(ctx), Some(id)) => {
        let node_id = ctx.push_node(
          [],
          CrossEntropy(Some(id), probs, self.shape, target_data),
        )
        {
          data: [loss],
          shape: [],
          requires_grad: true,
          context: Some(ctx),
          node_id: Some(node_id),
        }
      }
      _ => abort("differentiable tensor operation requires an autograd context")
    }
  } else {
    Tensor::scalar(loss)
  }
}
