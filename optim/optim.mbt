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
        param.assign_parameter_data(data)
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
  priv state : Ref[AdamWState]
}

///|
pub struct AdamWCheckpoint {
  priv m : Array[Array[Double]]
  priv v : Array[Array[Double]]
  priv lr : Double
  priv beta1 : Double
  priv beta2 : Double
  priv eps : Double
  priv weight_decays : Array[Double]
  priv step : Int
  priv beta1_power : Double
  priv beta2_power : Double
}

///|
pub struct AdamWConfig {
  priv lr : Double
  priv beta1 : Double
  priv beta2 : Double
  priv eps : Double
  priv weight_decay : Double
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
pub fn AdamWConfig::AdamWConfig(
  lr : Double,
  beta1? : Double = 0.9,
  beta2? : Double = 0.999,
  eps? : Double = 1.0e-8,
  weight_decay? : Double = 0.0,
) -> AdamWConfig {
  if lr <= 0.0 {
    abort("AdamW learning rate must be positive")
  }
  if beta1 < 0.0 || beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0 {
    abort("AdamW beta values must be in [0, 1)")
  }
  if eps <= 0.0 {
    abort("AdamW eps must be positive")
  }
  if weight_decay < 0.0 {
    abort("AdamW weight decay must not be negative")
  }
  { lr, beta1, beta2, eps, weight_decay }
}

///|
pub fn AdamWConfig::lr(self : AdamWConfig) -> Double {
  self.lr
}

///|
pub fn AdamWConfig::beta1(self : AdamWConfig) -> Double {
  self.beta1
}

///|
pub fn AdamWConfig::beta2(self : AdamWConfig) -> Double {
  self.beta2
}

///|
pub fn AdamWConfig::eps(self : AdamWConfig) -> Double {
  self.eps
}

///|
pub fn AdamWConfig::weight_decay(self : AdamWConfig) -> Double {
  self.weight_decay
}

///|
pub fn AdamW::AdamW(
  params : Array[@tensor.Tensor],
  config : AdamWConfig,
) -> AdamW {
  AdamW::with_parameter_weight_decays(
    params,
    config,
    Array::make(params.length(), config.weight_decay),
  )
}

///|
pub fn AdamW::with_parameter_weight_decays(
  params : Array[@tensor.Tensor],
  config : AdamWConfig,
  weight_decays : Array[Double],
) -> AdamW {
  if params.length() != weight_decays.length() {
    abort("AdamW weight decay count must match parameter count")
  }
  for weight_decay in weight_decays {
    if weight_decay < 0.0 {
      abort("AdamW weight decay must not be negative")
    }
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
        lr: config.lr,
        beta1: config.beta1,
        beta2: config.beta2,
        eps: config.eps,
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
        param.assign_parameter_data(data)
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

///|
pub fn AdamWCheckpoint::m(self : AdamWCheckpoint) -> Array[Array[Double]] {
  copy_double_arrays(self.m)
}

///|
pub fn AdamWCheckpoint::v(self : AdamWCheckpoint) -> Array[Array[Double]] {
  copy_double_arrays(self.v)
}

///|
pub fn AdamWCheckpoint::lr(self : AdamWCheckpoint) -> Double {
  self.lr
}

///|
pub fn AdamWCheckpoint::beta1(self : AdamWCheckpoint) -> Double {
  self.beta1
}

///|
pub fn AdamWCheckpoint::beta2(self : AdamWCheckpoint) -> Double {
  self.beta2
}

///|
pub fn AdamWCheckpoint::eps(self : AdamWCheckpoint) -> Double {
  self.eps
}

///|
pub fn AdamWCheckpoint::weight_decays(self : AdamWCheckpoint) -> Array[Double] {
  self.weight_decays.copy()
}

///|
pub fn AdamWCheckpoint::step(self : AdamWCheckpoint) -> Int {
  self.step
}

///|
pub fn AdamWCheckpoint::beta1_power(self : AdamWCheckpoint) -> Double {
  self.beta1_power
}

///|
pub fn AdamWCheckpoint::beta2_power(self : AdamWCheckpoint) -> Double {
  self.beta2_power
}
