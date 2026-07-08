///|
fn result_context(a : Tensor, b : Tensor) -> AutogradContext? {
  match (a.context, b.context) {
    (Some(ca), Some(cb)) => {
      require_same_context(ca, cb)
      Some(ca)
    }
    (Some(ca), None) => Some(ca)
    (None, Some(cb)) => Some(cb)
    (None, None) => None
  }
}

///|
fn binary_data(
  a : Tensor,
  b : Tensor,
  out_shape : Array[Int],
  f : (Double, Double) -> Double,
) -> Array[Double] {
  let out = Array::make(shape_size(out_shape), 0.0)
  for i in 0..<out.length() {
    let out_idx = unravel_index(i, out_shape)
    let ai = broadcast_source_index(out_idx, a.shape)
    let bi = broadcast_source_index(out_idx, b.shape)
    out[i] = f(a.data[ai], b.data[bi])
  }
  out
}

///|
fn binary_tensor(
  a : Tensor,
  b : Tensor,
  f : (Double, Double) -> Double,
  kind : (Int?, Int?, Array[Int]) -> NodeKind,
) -> Tensor {
  let out_shape = broadcast_shape(a.shape, b.shape)
  let out_data = binary_data(a, b, out_shape, f)
  let needs_grad = a.requires_grad || b.requires_grad
  if needs_grad {
    match result_context(a, b) {
      Some(ctx) => {
        let aid = if a.requires_grad {
          match a.node_id {
            Some(id) => Some(id)
            None => abort("requires_grad tensor is missing an autograd node")
          }
        } else {
          None
        }
        let bid = if b.requires_grad {
          match b.node_id {
            Some(id) => Some(id)
            None => abort("requires_grad tensor is missing an autograd node")
          }
        } else {
          None
        }
        let node_id = ctx.push_node(out_shape, kind(aid, bid, out_shape))
        {
          data: out_data,
          shape: out_shape,
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

///|
pub fn Tensor::add(self : Tensor, that : Tensor) -> Tensor {
  binary_tensor(self, that, fn(x, y) { x + y }, fn(aid, bid, _shape) {
    Add(aid, self.shape, bid, that.shape)
  })
}

///|
pub fn Tensor::sub(self : Tensor, that : Tensor) -> Tensor {
  binary_tensor(self, that, fn(x, y) { x - y }, fn(aid, bid, _shape) {
    Sub(aid, self.shape, bid, that.shape)
  })
}

///|
pub fn Tensor::mul(self : Tensor, that : Tensor) -> Tensor {
  binary_tensor(self, that, fn(x, y) { x * y }, fn(aid, bid, _shape) {
    Mul(aid, self.data, self.shape, bid, that.data, that.shape)
  })
}

///|
pub fn Tensor::div(self : Tensor, that : Tensor) -> Tensor {
  binary_tensor(self, that, fn(x, y) { x / y }, fn(aid, bid, _shape) {
    Div(aid, self.data, self.shape, bid, that.data, that.shape)
  })
}

///|
pub fn Tensor::neg(self : Tensor) -> Tensor {
  let out_data = self.data.map(x => -x)
  if self.requires_grad {
    match (self.context, self.node_id) {
      (Some(ctx), Some(id)) => {
        let node_id = ctx.push_node(self.shape, Neg(Some(id), self.shape))
        {
          data: out_data,
          shape: copy_ints(self.shape),
          requires_grad: true,
          context: Some(ctx),
          node_id: Some(node_id),
        }
      }
      _ => abort("differentiable tensor operation requires an autograd context")
    }
  } else {
    Tensor::from_array(out_data, self.shape)
  }
}

///|
pub impl Add for Tensor with fn add(self : Tensor, that : Tensor) -> Tensor {
  self.add(that)
}

///|
pub impl Sub for Tensor with fn sub(self : Tensor, that : Tensor) -> Tensor {
  self.sub(that)
}

///|
pub impl Mul for Tensor with fn mul(self : Tensor, that : Tensor) -> Tensor {
  self.mul(that)
}

///|
pub impl Div for Tensor with fn div(self : Tensor, that : Tensor) -> Tensor {
  self.div(that)
}

///|
pub impl Neg for Tensor with fn neg(self : Tensor) -> Tensor {
  self.neg()
}
