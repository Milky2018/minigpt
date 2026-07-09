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
fn tape_input(t : Tensor) -> TapeInput {
  { target: grad_parent_ref(t), shape: copy_ints(t.shape) }
}

///|
fn copy_tape_inputs(inputs : Array[TapeInput]) -> Array[TapeInput] {
  let copied : Array[TapeInput] = []
  for input in inputs {
    copied.push({ target: input.target, shape: copy_ints(input.shape) })
  }
  copied
}

///|
fn unary_tensor(
  input : Tensor,
  out_data : Array[Double],
  out_shape : Array[Int],
  op : () -> &BackwardOp,
) -> Tensor {
  if input.requires_grad {
    match input.context {
      Some(ctx) => {
        let node_ref = ctx.push_tape_node(out_shape, [tape_input(input)], op())
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
