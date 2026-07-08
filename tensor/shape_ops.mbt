///|
pub fn Tensor::reshape(self : Tensor, shape : Array[Int]) -> Tensor {
  if shape_size(shape) != self.data.length() {
    abort("reshape must preserve tensor element count")
  }
  if self.requires_grad {
    match (self.context, self.node_id) {
      (Some(ctx), Some(id)) => {
        let node_id = ctx.push_node(shape, Reshape(Some(id), self.shape))
        {
          data: copy_doubles(self.data),
          shape: copy_ints(shape),
          requires_grad: true,
          context: Some(ctx),
          node_id: Some(node_id),
        }
      }
      _ => abort("differentiable tensor operation requires an autograd context")
    }
  } else {
    Tensor::from_array(self.data, shape)
  }
}
