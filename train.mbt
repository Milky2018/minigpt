///|
pub const BATCH_SIZE : Int = 64

///|
pub const BLOCK_SIZE : Int = 256

///|
pub const TRAINING_STEPS : Int = 5000

///|
pub const LEARNING_RATE : Double = 1.0e-3

///|
pub const MIN_LR : Double = 1.0e-4

///|
pub const WARMUP_ITERS : Int = 100

///|
pub const EVAL_INTERVAL : Int = 250

///|
pub const EVAL_ITERS : Int = 200

///|
pub const LOG_INTERVAL : Int = 10

///|
pub const WEIGHT_DECAY : Double = 1.0e-1

///|
pub const BETA1 : Double = 0.9

///|
pub const BETA2 : Double = 0.99

///|
pub const GRAD_CLIP : Double = 1.0

///|
pub struct TrainingConfig {
  batch_size : Int
  block_size : Int
  steps : Int
  learning_rate : Double
  min_lr : Double
  warmup_iters : Int
  eval_interval : Int
  eval_iters : Int
  log_interval : Int
  weight_decay : Double
  beta1 : Double
  beta2 : Double
  grad_clip : Double
  always_save_checkpoint : Bool
}

///|
pub fn TrainingConfig::new(
  batch_size : Int,
  block_size : Int,
  steps : Int,
  learning_rate : Double,
  min_lr? : Double = MIN_LR,
  warmup_iters? : Int = WARMUP_ITERS,
  eval_interval? : Int = EVAL_INTERVAL,
  eval_iters? : Int = EVAL_ITERS,
  log_interval? : Int = LOG_INTERVAL,
  weight_decay? : Double = WEIGHT_DECAY,
  beta1? : Double = BETA1,
  beta2? : Double = BETA2,
  grad_clip? : Double = GRAD_CLIP,
  always_save_checkpoint? : Bool = false,
) -> TrainingConfig {
  if batch_size <= 0 {
    abort("batch_size must be positive")
  }
  if block_size <= 0 {
    abort("block_size must be positive")
  }
  if steps < 0 {
    abort("steps must not be negative")
  }
  if learning_rate <= 0.0 {
    abort("learning_rate must be positive")
  }
  if min_lr <= 0.0 {
    abort("min_lr must be positive")
  }
  if warmup_iters < 0 {
    abort("warmup_iters must not be negative")
  }
  if eval_interval <= 0 {
    abort("eval_interval must be positive")
  }
  if eval_iters <= 0 {
    abort("eval_iters must be positive")
  }
  if log_interval <= 0 {
    abort("log_interval must be positive")
  }
  if weight_decay < 0.0 {
    abort("weight_decay must not be negative")
  }
  if beta1 < 0.0 || beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0 {
    abort("AdamW beta values must be in [0, 1)")
  }
  if grad_clip < 0.0 {
    abort("grad_clip must not be negative")
  }
  {
    batch_size,
    block_size,
    steps,
    learning_rate,
    min_lr,
    warmup_iters,
    eval_interval,
    eval_iters,
    log_interval,
    weight_decay,
    beta1,
    beta2,
    grad_clip,
    always_save_checkpoint,
  }
}

///|
pub fn TrainingConfig::recommended() -> TrainingConfig {
  TrainingConfig::new(BATCH_SIZE, BLOCK_SIZE, TRAINING_STEPS, LEARNING_RATE)
}

///|
pub fn TrainingConfig::learning_rate_at(
  self : TrainingConfig,
  iter : Int,
) -> Double {
  if iter < 0 {
    abort("iter must not be negative")
  }
  if iter < self.warmup_iters {
    return self.learning_rate *
      (iter + 1).to_double() /
      (self.warmup_iters + 1).to_double()
  }
  if iter > self.steps {
    return self.min_lr
  }
  if self.steps <= self.warmup_iters {
    return self.min_lr
  }
  let decay_ratio = (iter - self.warmup_iters).to_double() /
    (self.steps - self.warmup_iters).to_double()
  let coeff = 0.5 * (1.0 + @math.cos(@math.PI * decay_ratio))
  self.min_lr + coeff * (self.learning_rate - self.min_lr)
}

///|
pub struct TrainingStats {
  losses : Array[Double]
  initial_eval_loss : Double
  final_eval_loss : Double
  best_val_loss : Double
  saved_checkpoints : Int
}

///|
pub fn sample_batch(
  token_ids : Array[Int],
  batch_size : Int,
  block_size : Int,
  rng : @random.Rand,
) -> (@tensor.TokenIds, @tensor.TokenIds) {
  if batch_size <= 0 {
    abort("batch_size must be positive")
  }
  if block_size <= 0 {
    abort("block_size must be positive")
  }
  if token_ids.length() <= block_size {
    abort("token_ids must contain more items than block_size")
  }
  let inputs : Array[Int] = []
  let targets : Array[Int] = []
  let limit = token_ids.length() - block_size
  for _ in 0..<batch_size {
    let start = rng.int(limit~)
    for offset in 0..<block_size {
      inputs.push(token_ids[start + offset])
      targets.push(token_ids[start + offset + 1])
    }
  }
  (
    @tensor.TokenIds::new(inputs, [batch_size, block_size]),
    @tensor.TokenIds::new(targets, [batch_size, block_size]),
  )
}

///|
fn eval_token_ids(
  model : MiniGPT,
  token_ids : Array[Int],
  config : TrainingConfig,
  rng : @random.Rand,
) -> Double {
  let mut sum = 0.0
  for _ in 0..<config.eval_iters {
    let (inputs, targets) = sample_batch(
      token_ids,
      config.batch_size,
      config.block_size,
      rng,
    )
    sum += model.loss(inputs, targets).data()[0]
    model.clear_graph()
  }
  sum / config.eval_iters.to_double()
}

///|
pub fn train_token_ids(
  model : MiniGPT,
  train_ids : Array[Int],
  val_ids : Array[Int],
  config : TrainingConfig,
  rng : @random.Rand,
) -> TrainingStats {
  train_token_ids_with_checkpoint_callback(
    model,
    train_ids,
    val_ids,
    config,
    rng,
    fn(_model, _optimizer, _iter_num, _best_val_loss, _config) { () },
  )
}

///|
pub fn train_token_ids_with_checkpoint_callback(
  model : MiniGPT,
  train_ids : Array[Int],
  val_ids : Array[Int],
  config : TrainingConfig,
  rng : @random.Rand,
  on_checkpoint : (MiniGPT, @optim.AdamW, Int, Double, TrainingConfig) -> Unit,
) -> TrainingStats {
  let losses : Array[Double] = []
  let params = model.parameters()
  let optimizer = @optim.AdamW::new_with_parameter_weight_decays(
    params,
    config.learning_rate,
    config.beta1,
    config.beta2,
    1.0e-8,
    model.parameter_weight_decays(config.weight_decay),
  )
  let mut iter_num = 0
  let mut best_val_loss = 1.0e9
  let mut saved_checkpoints = 0
  let mut initial_eval_loss = @double.not_a_number
  let mut final_eval_loss = @double.not_a_number
  while true {
    let lr = config.learning_rate_at(iter_num)
    optimizer.set_learning_rate(lr)
    if iter_num % config.eval_interval == 0 {
      let train_loss = eval_token_ids(model, train_ids, config, rng)
      let val_loss = eval_token_ids(model, val_ids, config, rng)
      println(
        "step \{iter_num}: train loss \{train_loss}, val loss \{val_loss}",
      )
      if initial_eval_loss != initial_eval_loss {
        initial_eval_loss = val_loss
      }
      final_eval_loss = val_loss
      let improved = val_loss < best_val_loss
      if improved {
        best_val_loss = val_loss
      }
      if iter_num > 0 && (improved || config.always_save_checkpoint) {
        on_checkpoint(model, optimizer, iter_num, best_val_loss, config)
        saved_checkpoints += 1
      }
    }
    let (inputs, targets) = sample_batch(
      train_ids,
      config.batch_size,
      config.block_size,
      rng,
    )
    let loss = model.loss_train(inputs, targets, rng)
    let loss_value = loss.data()[0]
    losses.push(loss_value)
    loss.backward()
    optimizer.step_with_grad_clip(config.grad_clip)
    model.clear_graph()
    if iter_num % config.log_interval == 0 {
      println("iter \{iter_num}: loss \{loss_value}, lr \{lr}")
    }
    iter_num += 1
    if iter_num > config.steps {
      break
    }
  }
  {
    losses,
    initial_eval_loss,
    final_eval_loss,
    best_val_loss,
    saved_checkpoints,
  }
}
