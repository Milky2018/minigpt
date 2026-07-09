///|
pub fn Tensor::from_array(data : Array[Double], shape : Array[Int]) -> Tensor {
  if data.length() != shape_size(shape) {
    abort("tensor data length does not match shape")
  }
  {
    data: copy_doubles(data),
    shape: copy_ints(shape),
    requires_grad: false,
    context: None,
    node_ref: None,
  }
}

///|
pub fn Tensor::scalar(value : Double) -> Tensor {
  Tensor::from_array([value], [])
}

///|
pub fn Tensor::zeros(shape : Array[Int]) -> Tensor {
  Tensor::from_array(Array::make(shape_size(shape), 0.0), shape)
}

///|
pub fn Tensor::ones(shape : Array[Int]) -> Tensor {
  Tensor::from_array(Array::make(shape_size(shape), 1.0), shape)
}

///|
pub fn Tensor::parameter(
  ctx : AutogradContext,
  data : Array[Double],
  shape : Array[Int],
) -> Tensor {
  if data.length() != shape_size(shape) {
    abort("tensor data length does not match shape")
  }
  let node_ref = ctx.push_parameter(shape)
  {
    data: copy_doubles(data),
    shape: copy_ints(shape),
    requires_grad: true,
    context: Some(ctx),
    node_ref: Some(node_ref),
  }
}

///|
pub fn Tensor::shape(self : Tensor) -> Array[Int] {
  copy_ints(self.shape)
}

///|
pub fn Tensor::data(self : Tensor) -> Array[Double] {
  copy_doubles(self.data)
}

///|
pub fn Tensor::requires_grad(self : Tensor) -> Bool {
  self.requires_grad
}

///|
pub fn Tensor::numel(self : Tensor) -> Int {
  self.data.length()
}

///|
pub fn Tensor::grad(self : Tensor) -> Tensor? {
  match (self.context, self.node_ref) {
    (Some(ctx), Some(node_ref)) =>
      match ctx.grad(node_ref) {
        Some(grad) => Some(Tensor::from_array(grad, self.shape))
        None => None
      }
    _ => None
  }
}

///|
pub fn Tensor::backward(self : Tensor) -> Unit {
  if shape_size(self.shape) != 1 {
    abort("backward requires a scalar tensor")
  }
  match (self.context, self.node_ref) {
    (Some(ctx), Some(node_ref)) => {
      ctx.zero_grad()
      ctx.add_grad(node_ref, [1.0])
      if ctx.state.val.tape.length() == 0 {
        return
      }
      let mut i = ctx.state.val.tape.length() - 1
      while i >= 0 {
        match ctx.state.val.tape[i].grad {
          Some(grad) => backward_node(ctx, i, grad)
          None => ()
        }
        if i == 0 {
          break
        }
        i -= 1
      }
    }
    _ => abort("backward requires a tensor attached to an autograd context")
  }
}

///|
pub fn TokenIds::TokenIds(data : Array[Int], shape : Array[Int]) -> TokenIds {
  if data.length() != shape_size(shape) {
    abort("token id data length does not match shape")
  }
  { data: data.copy(), shape: copy_ints(shape) }
}

///|
pub fn TokenIds::data(self : TokenIds) -> Array[Int] {
  self.data.copy()
}

///|
pub fn TokenIds::shape(self : TokenIds) -> Array[Int] {
  copy_ints(self.shape)
}
