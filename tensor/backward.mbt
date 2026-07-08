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
    Exp(parent, input_data, input_shape) =>
      match parent {
        Some(id) => {
          let g = Array::make(grad.length(), 0.0)
          for i in 0..<g.length() {
            g[i] = grad[i] * @math.exp(input_data[i])
          }
          ctx.add_grad(id, reduce_to_shape(g, node.shape, input_shape))
        }
        None => ()
      }
    Log(parent, input_data, input_shape) =>
      match parent {
        Some(id) => {
          let g = Array::make(grad.length(), 0.0)
          for i in 0..<g.length() {
            g[i] = grad[i] / input_data[i]
          }
          ctx.add_grad(id, reduce_to_shape(g, node.shape, input_shape))
        }
        None => ()
      }
    Relu(parent, input_data, input_shape) =>
      match parent {
        Some(id) => {
          let g = Array::make(grad.length(), 0.0)
          for i in 0..<g.length() {
            g[i] = if input_data[i] > 0.0 { grad[i] } else { 0.0 }
          }
          ctx.add_grad(id, reduce_to_shape(g, node.shape, input_shape))
        }
        None => ()
      }
    SwapAxes(parent, parent_shape, axis_a, axis_b) =>
      match parent {
        Some(id) => {
          let parent_grad = Array::make(shape_size(parent_shape), 0.0)
          for i in 0..<grad.length() {
            let out_idx = unravel_index(i, node.shape)
            let in_idx = copy_ints(out_idx)
            in_idx[axis_a] = out_idx[axis_b]
            in_idx[axis_b] = out_idx[axis_a]
            parent_grad[ravel_index(in_idx, parent_shape)] = grad[i]
          }
          ctx.add_grad(id, parent_grad)
        }
        None => ()
      }
    Matmul(aid, a_data, a_shape, bid, b_data, b_shape) => {
      let k = a_shape[a_shape.length() - 1]
      let a_grad = Array::make(shape_size(a_shape), 0.0)
      let b_grad = Array::make(shape_size(b_shape), 0.0)
      for i in 0..<grad.length() {
        let out_idx = unravel_index(i, node.shape)
        let row = out_idx[out_idx.length() - 2]
        let col = out_idx[out_idx.length() - 1]
        for inner in 0..<k {
          let a_idx = out_idx[0:out_idx.length() - 2].to_owned()
          a_idx.push(row)
          a_idx.push(inner)
          let b_idx = out_idx[0:out_idx.length() - 2].to_owned()
          b_idx.push(inner)
          b_idx.push(col)
          let ai = broadcast_source_index(a_idx, a_shape)
          let bi = broadcast_source_index(b_idx, b_shape)
          a_grad[ai] += grad[i] * b_data[bi]
          b_grad[bi] += grad[i] * a_data[ai]
        }
      }
      match aid {
        Some(id) => ctx.add_grad(id, a_grad)
        None => ()
      }
      match bid {
        Some(id) => ctx.add_grad(id, b_grad)
        None => ()
      }
    }
    Softmax(parent, output_data, axis) =>
      match parent {
        Some(id) => {
          let g = Array::make(grad.length(), 0.0)
          let outer = shape_size(node.shape[0:axis].to_owned())
          let dim = node.shape[axis]
          let inner = shape_size(node.shape[axis + 1:].to_owned())
          for o in 0..<outer {
            for i in 0..<inner {
              let base = o * dim * inner + i
              let mut dot = 0.0
              for d in 0..<dim {
                dot += grad[base + d * inner] * output_data[base + d * inner]
              }
              for d in 0..<dim {
                let idx = base + d * inner
                g[idx] = output_data[idx] * (grad[idx] - dot)
              }
            }
          }
          ctx.add_grad(id, g)
        }
        None => ()
      }
    GatherRows(parent, ids, weight_shape) =>
      match parent {
        Some(id) => {
          let width = weight_shape[1]
          let weight_grad = Array::make(shape_size(weight_shape), 0.0)
          for i in 0..<ids.length() {
            let row = ids[i]
            for d in 0..<width {
              weight_grad[row * width + d] += grad[i * width + d]
            }
          }
          ctx.add_grad(id, weight_grad)
        }
        None => ()
      }
    CrossEntropy(parent, probs, logits_shape, targets) =>
      match parent {
        Some(id) => {
          let vocab = logits_shape[logits_shape.length() - 1]
          let scale = grad[0] / targets.length().to_double()
          let logits_grad = probs.copy()
          for i in 0..<targets.length() {
            logits_grad[i * vocab + targets[i]] -= 1.0
          }
          for i in 0..<logits_grad.length() {
            logits_grad[i] *= scale
          }
          ctx.add_grad(id, logits_grad)
        }
        None => ()
      }
  }
}
