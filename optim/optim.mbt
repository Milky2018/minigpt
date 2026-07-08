///|
pub fn sgd_step(params : Array[@tensor.Tensor], lr : Double) -> Unit {
  for param in params {
    match param.grad() {
      Some(grad_tensor) => {
        let data = param.data()
        let grad = grad_tensor.data()
        for i in 0..<data.length() {
          data[i] -= lr * grad[i]
        }
        param.set_data_in_place(data)
      }
      None => ()
    }
  }
}

///|
struct AdamWState {
  params : Array[@tensor.Tensor]
  m : Array[Array[Double]]
  v : Array[Array[Double]]
  mut lr : Double
  beta1 : Double
  beta2 : Double
  eps : Double
  weight_decays : Array[Double]
  mut step : Int
  mut beta1_power : Double
  mut beta2_power : Double
}

///|
pub struct AdamW {
  state : Ref[AdamWState]
}

///|
pub struct AdamWCheckpoint {
  m : Array[Array[Double]]
  v : Array[Array[Double]]
  lr : Double
  beta1 : Double
  beta2 : Double
  eps : Double
  weight_decays : Array[Double]
  step : Int
  beta1_power : Double
  beta2_power : Double
}

///|
fn copy_double_arrays(values : Array[Array[Double]]) -> Array[Array[Double]] {
  let copied : Array[Array[Double]] = []
  for value in values {
    copied.push(value.copy())
  }
  copied
}

///|
pub fn AdamW::new(params : Array[@tensor.Tensor], lr : Double) -> AdamW {
  AdamW::new_with_options(params, lr, 0.9, 0.999, 1.0e-8, 0.0)
}

///|
pub fn AdamW::new_with_options(
  params : Array[@tensor.Tensor],
  lr : Double,
  beta1 : Double,
  beta2 : Double,
  eps : Double,
  weight_decay : Double,
) -> AdamW {
  AdamW::new_with_parameter_weight_decays(
    params,
    lr,
    beta1,
    beta2,
    eps,
    Array::make(params.length(), weight_decay),
  )
}

///|
pub fn AdamW::new_with_parameter_weight_decays(
  params : Array[@tensor.Tensor],
  lr : Double,
  beta1 : Double,
  beta2 : Double,
  eps : Double,
  weight_decays : Array[Double],
) -> AdamW {
  if params.length() != weight_decays.length() {
    abort("AdamW weight decay count must match parameter count")
  }
  let m : Array[Array[Double]] = []
  let v : Array[Array[Double]] = []
  for param in params {
    m.push(Array::make(param.numel(), 0.0))
    v.push(Array::make(param.numel(), 0.0))
  }
  {
    state: {
      val: {
        params: params.copy(),
        m,
        v,
        lr,
        beta1,
        beta2,
        eps,
        weight_decays: weight_decays.copy(),
        step: 0,
        beta1_power: 1.0,
        beta2_power: 1.0,
      },
    },
  }
}

///|
pub fn AdamW::step(self : AdamW) -> Unit {
  self.step_with_gradient_scale(1.0)
}

///|
pub fn AdamW::set_learning_rate(self : AdamW, lr : Double) -> Unit {
  if lr <= 0.0 {
    abort("AdamW learning rate must be positive")
  }
  self.state.val.lr = lr
}

///|
pub fn AdamW::step_with_grad_clip(self : AdamW, max_norm : Double) -> Unit {
  if max_norm < 0.0 {
    abort("AdamW max_norm must not be negative")
  }
  let scale = self.gradient_clip_scale(max_norm)
  self.step_with_gradient_scale(scale)
}

///|
fn AdamW::gradient_clip_scale(self : AdamW, max_norm : Double) -> Double {
  if max_norm == 0.0 {
    return 1.0
  }
  let state = self.state.val
  let mut squared_sum = 0.0
  for param in state.params {
    match param.grad() {
      Some(grad_tensor) => {
        let grad = grad_tensor.data()
        for value in grad {
          squared_sum += value * value
        }
      }
      None => ()
    }
  }
  let norm = squared_sum.sqrt()
  if norm > max_norm {
    max_norm / (norm + 1.0e-6)
  } else {
    1.0
  }
}

///|
fn AdamW::step_with_gradient_scale(self : AdamW, grad_scale : Double) -> Unit {
  let state = self.state.val
  state.step += 1
  state.beta1_power *= state.beta1
  state.beta2_power *= state.beta2
  let bias1 = 1.0 - state.beta1_power
  let bias2 = 1.0 - state.beta2_power
  for pi in 0..<state.params.length() {
    let param = state.params[pi]
    match param.grad() {
      Some(grad_tensor) => {
        let data = param.data()
        let grad = grad_tensor.data()
        if data.length() != state.m[pi].length() {
          abort("AdamW state does not match parameter size")
        }
        for i in 0..<data.length() {
          let g = grad[i] * grad_scale
          state.m[pi][i] = state.beta1 * state.m[pi][i] +
            (1.0 - state.beta1) * g
          state.v[pi][i] = state.beta2 * state.v[pi][i] +
            (1.0 - state.beta2) * g * g
          let m_hat = state.m[pi][i] / bias1
          let v_hat = state.v[pi][i] / bias2
          data[i] = data[i] * (1.0 - state.lr * state.weight_decays[pi]) -
            state.lr * m_hat / (v_hat.sqrt() + state.eps)
        }
        param.set_data_in_place(data)
      }
      None => ()
    }
  }
}

///|
pub fn AdamW::step_count(self : AdamW) -> Int {
  self.state.val.step
}

///|
pub fn AdamW::checkpoint(self : AdamW) -> AdamWCheckpoint {
  let state = self.state.val
  {
    m: copy_double_arrays(state.m),
    v: copy_double_arrays(state.v),
    lr: state.lr,
    beta1: state.beta1,
    beta2: state.beta2,
    eps: state.eps,
    weight_decays: state.weight_decays.copy(),
    step: state.step,
    beta1_power: state.beta1_power,
    beta2_power: state.beta2_power,
  }
}
