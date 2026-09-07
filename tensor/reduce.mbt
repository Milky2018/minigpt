///|
pub fn Tensor::sum(self : Tensor) -> Tensor {
  let mut value = 0.0
  for x in self.data {
    value += x
  }
  if self.requires_grad {
    match (self.context, self.node_ref) {
      (Some(ctx), Some(id)) => {
        let node_ref = ctx.push_tape_node(
          [],
          [{ target: Some(id), shape: copy_ints(self.shape), }],
          SumBackward::{ },
        )
        {
          data: [value],
          shape: [],
          requires_grad: true,
          context: Some(ctx),
          node_ref: Some(node_ref),
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
