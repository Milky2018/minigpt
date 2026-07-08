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
  lr : Double
  beta1 : Double
  beta2 : Double
  eps : Double
  weight_decay : Double
  mut step : Int
  mut beta1_power : Double
  mut beta2_power : Double
}

///|
pub struct AdamW {
  state : Ref[AdamWState]
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
        weight_decay,
        step: 0,
        beta1_power: 1.0,
        beta2_power: 1.0,
      },
    },
  }
}

///|
pub fn AdamW::step(self : AdamW) -> Unit {
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
          let g = grad[i]
          state.m[pi][i] = state.beta1 * state.m[pi][i] +
            (1.0 - state.beta1) * g
          state.v[pi][i] = state.beta2 * state.v[pi][i] +
            (1.0 - state.beta2) * g * g
          let m_hat = state.m[pi][i] / bias1
          let v_hat = state.v[pi][i] / bias2
          data[i] = data[i] * (1.0 - state.lr * state.weight_decay) -
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
