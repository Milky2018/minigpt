///|
fn backward_node(ctx : AutogradContext, id : Int, grad : Array[Double]) -> Unit {
  let node = ctx.state.val.tape[id]
  let input_grads = node.op.backward(grad, node.shape, node.inputs)
  for input_grad in input_grads {
    if input_grad.slot < 0 || input_grad.slot >= node.inputs.length() {
      abort("backward op returned an invalid input slot")
    }
    let input = node.inputs[input_grad.slot]
    if input_grad.data.length() != shape_size(input.shape) {
      abort("backward op returned a gradient with the wrong shape")
    }
    match input.target {
      Some(target) => ctx.add_grad(target, input_grad.data)
      None => ()
    }
  }
}
