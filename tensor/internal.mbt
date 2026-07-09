///|
fn normalize_axis(axis : Int, rank : Int) -> Int {
  let actual = if axis < 0 { rank + axis } else { axis }
  if actual < 0 || actual >= rank {
    abort("tensor axis is out of range")
  }
  actual
}

///|
fn grad_parent_ref(t : Tensor) -> NodeRef? {
  if t.requires_grad {
    match t.node_ref {
      Some(node_ref) => Some(node_ref)
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
  kind : (NodeRef?) -> NodeKind,
) -> Tensor {
  if input.requires_grad {
    match input.context {
      Some(ctx) => {
        let parent = grad_parent_ref(input)
        let node_ref = ctx.push_tape_node(out_shape, kind(parent))
        {
          data: out_data,
          shape: copy_ints(out_shape),
          requires_grad: true,
          context: Some(ctx),
          node_ref: Some(node_ref),
        }
      }
      None =>
        abort("differentiable tensor operation requires an autograd context")
    }
  } else {
    Tensor::from_array(out_data, out_shape)
  }
}
