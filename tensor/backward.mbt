///|
fn backward_node(ctx : AutogradContext, id : Int, grad : Array[Double]) -> Unit {
  let node = ctx.node(id)
  match node.kind {
    Cleared => ()
    Parameter => ()
    Add(aid, a_shape, bid, b_shape) => {
      match aid {
        Some(parent) =>
          ctx.add_grad(parent, reduce_to_shape(grad, node.shape, a_shape))
        None => ()
      }
      match bid {
        Some(parent) =>
          ctx.add_grad(parent, reduce_to_shape(grad, node.shape, b_shape))
        None => ()
      }
    }
    Sub(aid, a_shape, bid, b_shape) => {
      match aid {
        Some(parent) =>
          ctx.add_grad(parent, reduce_to_shape(grad, node.shape, a_shape))
        None => ()
      }
      match bid {
        Some(parent) => {
          let b_grad = reduce_to_shape(grad, node.shape, b_shape).map(x => -x)
          ctx.add_grad(parent, b_grad)
        }
        None => ()
      }
    }
    Mul(aid, a_data, a_shape, bid, b_data, b_shape) => {
      let ga = Array::make(shape_size(node.shape), 0.0)
      let gb = Array::make(shape_size(node.shape), 0.0)
      for i in 0..<grad.length() {
        let out_idx = unravel_index(i, node.shape)
        let ai = broadcast_source_index(out_idx, a_shape)
        let bi = broadcast_source_index(out_idx, b_shape)
        ga[i] = grad[i] * b_data[bi]
        gb[i] = grad[i] * a_data[ai]
      }
      match aid {
        Some(parent) =>
          ctx.add_grad(parent, reduce_to_shape(ga, node.shape, a_shape))
        None => ()
      }
      match bid {
        Some(parent) =>
          ctx.add_grad(parent, reduce_to_shape(gb, node.shape, b_shape))
        None => ()
      }
    }
    Div(aid, a_data, a_shape, bid, b_data, b_shape) => {
      let ga = Array::make(shape_size(node.shape), 0.0)
      let gb = Array::make(shape_size(node.shape), 0.0)
      for i in 0..<grad.length() {
        let out_idx = unravel_index(i, node.shape)
        let ai = broadcast_source_index(out_idx, a_shape)
        let bi = broadcast_source_index(out_idx, b_shape)
        ga[i] = grad[i] / b_data[bi]
        gb[i] = grad[i] * -(a_data[ai] / (b_data[bi] * b_data[bi]))
      }
      match aid {
        Some(parent) =>
          ctx.add_grad(parent, reduce_to_shape(ga, node.shape, a_shape))
        None => ()
      }
      match bid {
        Some(parent) =>
          ctx.add_grad(parent, reduce_to_shape(gb, node.shape, b_shape))
        None => ()
      }
    }
    Neg(parent, parent_shape) =>
      match parent {
        Some(id) => {
          let parent_grad = reduce_to_shape(
            grad.map(x => -x),
            node.shape,
            parent_shape,
          )
          ctx.add_grad(id, parent_grad)
        }
        None => ()
      }
    Sum(parent, parent_shape) =>
      match parent {
        Some(id) =>
          ctx.add_grad(id, Array::make(shape_size(parent_shape), grad[0]))
        None => ()
      }
    Reshape(parent, parent_shape) =>
      match parent {
        Some(id) =>
          ctx.add_grad(id, Tensor::from_array(grad, parent_shape).data())
        None => ()
      }
  }
}
