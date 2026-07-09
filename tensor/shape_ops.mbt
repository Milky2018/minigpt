///|
pub fn Tensor::reshape(self : Tensor, shape : Array[Int]) -> Tensor {
  if shape_size(shape) != self.data.length() {
    abort("reshape must preserve tensor element count")
  }
  if self.requires_grad {
    match (self.context, self.node_ref) {
      (Some(ctx), Some(id)) => {
        let node_ref = ctx.push_tape_node(
          shape,
          [{ target: Some(id), shape: copy_ints(self.shape) }],
          ReshapeBackward::{  },
        )
        {
          data: copy_doubles(self.data),
          shape: copy_ints(shape),
          requires_grad: true,
          context: Some(ctx),
          node_ref: Some(node_ref),
        }
      }
      _ => abort("differentiable tensor operation requires an autograd context")
    }
  } else {
    Tensor::from_array(self.data, shape)
  }
}

///|
pub fn Tensor::swap_axes(self : Tensor, axis_a : Int, axis_b : Int) -> Tensor {
  let rank = self.shape.length()
  let a = normalize_axis(axis_a, rank)
  let b = normalize_axis(axis_b, rank)
  let out_shape = copy_ints(self.shape)
  out_shape[a] = self.shape[b]
  out_shape[b] = self.shape[a]
  let out = Array::make(self.data.length(), 0.0)
  for i in 0..<out.length() {
    let out_idx = unravel_index(i, out_shape)
    let in_idx = copy_ints(out_idx)
    in_idx[a] = out_idx[b]
    in_idx[b] = out_idx[a]
    out[i] = self.data[ravel_index(in_idx, self.shape)]
  }
  unary_tensor(self, out, out_shape, fn() {
    SwapAxesBackward::{ axis_a: a, axis_b: b }
  })
}

///|
pub fn Tensor::transpose2d(self : Tensor) -> Tensor {
  if self.shape.length() != 2 {
    abort("transpose2d requires a rank-2 tensor")
  }
  self.swap_axes(0, 1)
}
