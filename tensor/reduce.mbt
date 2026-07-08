///|
pub fn Tensor::sum(self : Tensor) -> Tensor {
  let mut value = 0.0
  for x in self.data {
    value += x
  }
  if self.requires_grad {
    match (self.context, self.node_id) {
      (Some(ctx), Some(id)) => {
        let node_id = ctx.push_node([], Sum(Some(id), self.shape))
        {
          data: [value],
          shape: [],
          requires_grad: true,
          context: Some(ctx),
          node_id: Some(node_id),
        }
      }
      _ => abort("differentiable tensor operation requires an autograd context")
    }
  } else {
    Tensor::scalar(value)
  }
}

///|
pub fn Tensor::mean(self : Tensor) -> Tensor {
  self.sum() / Tensor::scalar(self.numel().to_double())
}
