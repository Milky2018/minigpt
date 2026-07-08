///|
fn layer_norm_context(
  x : Tensor,
  weight : Tensor,
  bias : Tensor,
) -> AutogradContext? {
  match (result_context(x, weight), bias.context) {
    (Some(ctx), Some(bctx)) => {
      require_same_context(ctx, bctx)
      Some(ctx)
    }
    (Some(ctx), None) => Some(ctx)
    (None, Some(ctx)) => Some(ctx)
    (None, None) => None
  }
}

///|
pub fn Tensor::layer_norm(
  self : Tensor,
  weight : Tensor,
  bias : Tensor,
  eps : Double,
) -> Tensor {
  if eps < 0.0 {
    abort("layer_norm eps must not be negative")
  }
  if self.shape.length() == 0 {
    abort("layer_norm requires at least one dimension")
  }
  let hidden = self.shape[self.shape.length() - 1]
  if weight.shape != [hidden] || bias.shape != [hidden] {
    abort("layer_norm weight and bias must match the last dimension")
  }
  let rows = self.data.length() / hidden
  let out = Array::make(self.data.length(), 0.0)
  for row in 0..<rows {
    let base = row * hidden
    let mut mean = 0.0
    for i in 0..<hidden {
      mean += self.data[base + i]
    }
    mean = mean / hidden.to_double()
    let mut variance = 0.0
    for i in 0..<hidden {
      let centered = self.data[base + i] - mean
      variance += centered * centered
    }
    variance = variance / hidden.to_double()
    let inv_std = 1.0 / (variance + eps).sqrt()
    for i in 0..<hidden {
      let normalized = (self.data[base + i] - mean) * inv_std
      out[base + i] = normalized * weight.data[i] + bias.data[i]
    }
  }
  let needs_grad = self.requires_grad ||
    weight.requires_grad ||
    bias.requires_grad
  if needs_grad {
    match layer_norm_context(self, weight, bias) {
      Some(ctx) => {
        let node_id = ctx.push_node(
          self.shape,
          LayerNorm(
            grad_parent_id(self),
            self.data,
            self.shape,
            grad_parent_id(weight),
            weight.data,
            grad_parent_id(bias),
            eps,
          ),
        )
        {
          data: out,
          shape: copy_ints(self.shape),
          requires_grad: true,
          context: Some(ctx),
          node_id: Some(node_id),
        }
      }
      None =>
        abort("differentiable tensor operation requires an autograd context")
    }
  } else {
    Tensor::from_array(out, self.shape)
  }
}
