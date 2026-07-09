///|
fn batch_shape_for_matmul(
  a_shape : Array[Int],
  b_shape : Array[Int],
) -> Array[Int] {
  if a_shape.length() < 2 || b_shape.length() < 2 {
    abort("matmul requires tensors with rank >= 2")
  }
  let a_batch = a_shape[0:a_shape.length() - 2].to_owned()
  let b_batch = b_shape[0:b_shape.length() - 2].to_owned()
  broadcast_shape(a_batch, b_batch)
}

///|
fn matmul_shape(a_shape : Array[Int], b_shape : Array[Int]) -> Array[Int] {
  let m = a_shape[a_shape.length() - 2]
  let k = a_shape[a_shape.length() - 1]
  let kb = b_shape[b_shape.length() - 2]
  let n = b_shape[b_shape.length() - 1]
  if k != kb {
    abort("matmul inner dimensions do not match")
  }
  let out = batch_shape_for_matmul(a_shape, b_shape)
  out.push(m)
  out.push(n)
  out
}

///|
fn matmul_data(a : Tensor, b : Tensor, out_shape : Array[Int]) -> Array[Double] {
  let out = Array::make(shape_size(out_shape), 0.0)
  let k = a.shape[a.shape.length() - 1]
  for i in 0..<out.length() {
    let out_idx = unravel_index(i, out_shape)
    let row = out_idx[out_idx.length() - 2]
    let col = out_idx[out_idx.length() - 1]
    let mut sum = 0.0
    for inner in 0..<k {
      let a_idx = out_idx[0:out_idx.length() - 2].to_owned()
      a_idx.push(row)
      a_idx.push(inner)
      let b_idx = out_idx[0:out_idx.length() - 2].to_owned()
      b_idx.push(inner)
      b_idx.push(col)
      let ai = broadcast_source_index(a_idx, a.shape)
      let bi = broadcast_source_index(b_idx, b.shape)
      sum += a.data[ai] * b.data[bi]
    }
    out[i] = sum
  }
  out
}

///|
pub fn Tensor::matmul(self : Tensor, that : Tensor) -> Tensor {
  let out_shape = matmul_shape(self.shape, that.shape)
  let out_data = matmul_data(self, that, out_shape)
  let needs_grad = self.requires_grad || that.requires_grad
  if needs_grad {
    match result_context(self, that) {
      Some(ctx) => {
        let node_ref = ctx.push_tape_node(
          out_shape,
          Matmul(
            grad_parent_ref(self),
            self.data,
            self.shape,
            grad_parent_ref(that),
            that.data,
            that.shape,
          ),
        )
        {
          data: out_data,
          shape: out_shape,
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
