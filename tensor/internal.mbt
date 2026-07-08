///|
fn normalize_axis(axis : Int, rank : Int) -> Int {
  let actual = if axis < 0 { rank + axis } else { axis }
  if actual < 0 || actual >= rank {
    abort("tensor axis is out of range")
  }
  actual
}

///|
fn grad_parent_id(t : Tensor) -> Int? {
  if t.requires_grad {
    match t.node_id {
      Some(id) => Some(id)
      None => abort("requires_grad tensor is missing an autograd node")
    }
  } else {
    None
  }
}

///|
fn unary_tensor(
  input : Tensor,
  out_data : Array[Double],
  out_shape : Array[Int],
  kind : (Int?) -> NodeKind,
) -> Tensor {
  if input.requires_grad {
    match input.context {
      Some(ctx) => {
        let parent = grad_parent_id(input)
        let node_id = ctx.push_node(out_shape, kind(parent))
        {
          data: out_data,
          shape: copy_ints(out_shape),
          requires_grad: true,
          context: Some(ctx),
          node_id: Some(node_id),
        }
      }
      None =>
        abort("differentiable tensor operation requires an autograd context")
    }
  } else {
    Tensor::from_array(out_data, out_shape)
  }
}
