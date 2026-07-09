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
  priv batch_size : Int
  priv block_size : Int
  priv steps : Int
  priv learning_rate : Double
  priv min_lr : Double
  priv warmup_iters : Int
  priv eval_interval : Int
  priv eval_iters : Int
  priv log_interval : Int
  priv weight_decay : Double
  priv beta1 : Double
  priv beta2 : Double
  priv grad_clip : Double
  priv always_save_checkpoint : Bool
}

///|
pub fn TrainingConfig::TrainingConfig(
  batch_size~ : Int,
  block_size~ : Int,
  steps~ : Int,
  learning_rate~ : Double,
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
  TrainingConfig(
    batch_size=BATCH_SIZE,
    block_size=BLOCK_SIZE,
    steps=TRAINING_STEPS,
    learning_rate=LEARNING_RATE,
  )
}

///|
pub fn TrainingConfig::batch_size(self : TrainingConfig) -> Int {
  self.batch_size
}

///|
pub fn TrainingConfig::block_size(self : TrainingConfig) -> Int {
  self.block_size
}

///|
pub fn TrainingConfig::steps(self : TrainingConfig) -> Int {
  self.steps
}

///|
pub fn TrainingConfig::learning_rate(self : TrainingConfig) -> Double {
  self.learning_rate
}

///|
pub fn TrainingConfig::min_lr(self : TrainingConfig) -> Double {
  self.min_lr
}

///|
pub fn TrainingConfig::warmup_iters(self : TrainingConfig) -> Int {
  self.warmup_iters
}

///|
pub fn TrainingConfig::eval_interval(self : TrainingConfig) -> Int {
  self.eval_interval
}

///|
pub fn TrainingConfig::eval_iters(self : TrainingConfig) -> Int {
  self.eval_iters
}

///|
pub fn TrainingConfig::log_interval(self : TrainingConfig) -> Int {
  self.log_interval
}

///|
pub fn TrainingConfig::weight_decay(self : TrainingConfig) -> Double {
  self.weight_decay
}

///|
pub fn TrainingConfig::beta1(self : TrainingConfig) -> Double {
  self.beta1
}

///|
pub fn TrainingConfig::beta2(self : TrainingConfig) -> Double {
  self.beta2
}

///|
pub fn TrainingConfig::grad_clip(self : TrainingConfig) -> Double {
  self.grad_clip
}

///|
pub fn TrainingConfig::always_save_checkpoint(self : TrainingConfig) -> Bool {
  self.always_save_checkpoint
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
pub struct TrainingResult {
  model : MiniGPT
  tokenizer : @tokenizer.Tokenizer
  stats : TrainingStats
}

///|
pub struct TrainingState {
  priv model : MiniGPT
  priv optimizer : @optim.AdamW
  priv iter_num : Int
  priv best_val_loss : Double
  priv config : TrainingConfig
}

///|
fn TrainingState::TrainingState(
  model : MiniGPT,
  optimizer : @optim.AdamW,
  iter_num : Int,
  best_val_loss : Double,
  config : TrainingConfig,
) -> TrainingState {
  { model, optimizer, iter_num, best_val_loss, config }
}

///|
pub fn TrainingState::model(self : TrainingState) -> MiniGPT {
  self.model
}

///|
pub fn TrainingState::optimizer(self : TrainingState) -> @optim.AdamW {
  self.optimizer
}

///|
pub fn TrainingState::iter_num(self : TrainingState) -> Int {
  self.iter_num
}

///|
pub fn TrainingState::best_val_loss(self : TrainingState) -> Double {
  self.best_val_loss
}

///|
pub fn TrainingState::config(self : TrainingState) -> TrainingConfig {
  self.config
}

///|
pub struct EvalEvent {
  iter_num : Int
  train_loss : Double
  val_loss : Double
}

///|
pub struct LogEvent {
  iter_num : Int
  loss : Double
  learning_rate : Double
}

///|
pub struct TrainingCallbacks {
  priv on_eval : (EvalEvent) -> Unit
  priv on_log : (LogEvent) -> Unit
  priv on_checkpoint : (TrainingState) -> Unit
}

///|
pub fn TrainingCallbacks::TrainingCallbacks(
  on_eval? : (EvalEvent) -> Unit = fn(_event) { () },
  on_log? : (LogEvent) -> Unit = fn(_event) { () },
  on_checkpoint? : (TrainingState) -> Unit = fn(_state) { () },
) -> TrainingCallbacks {
  { on_eval, on_log, on_checkpoint }
}

///|
fn sample_batch(
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
    TokenIds(inputs, [batch_size, block_size]),
    TokenIds(targets, [batch_size, block_size]),
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
  train_token_ids_internal(
    model,
    train_ids,
    val_ids,
    config,
    rng,
    TrainingCallbacks(),
  )
}

///|
pub fn train_text(
  text : String,
  tokenizer_config : @tokenizer.TokenizerConfig,
  config : TrainingConfig,
  rng : @random.Rand,
) -> TrainingResult {
  train_text_with_architecture(
    text,
    tokenizer_config,
    config,
    ArchitectureConfig(),
    rng,
  )
}

///|
pub fn train_text_with_architecture(
  text : String,
  tokenizer_config : @tokenizer.TokenizerConfig,
  config : TrainingConfig,
  architecture : ArchitectureConfig,
  rng : @random.Rand,
) -> TrainingResult {
  let dataset = @tokenizer.prepare_token_dataset(text, tokenizer_config)
  let tokenizer = dataset.tokenizer()
  let train_ids = dataset.train_ids()
  let val_ids = dataset.val_ids()
  let model = MiniGPT(
    ModelConfig::from_architecture(
      tokenizer.vocab_size(),
      architecture,
      block_size=config.block_size,
    ),
    rng,
  )
  let stats = train_token_ids(model, train_ids, val_ids, config, rng)
  { model, tokenizer, stats }
}

///|
pub fn train_token_ids_with_callbacks(
  model : MiniGPT,
  train_ids : Array[Int],
  val_ids : Array[Int],
  config : TrainingConfig,
  rng : @random.Rand,
  callbacks : TrainingCallbacks,
) -> TrainingStats {
  train_token_ids_internal(model, train_ids, val_ids, config, rng, callbacks)
}

///|
fn train_token_ids_internal(
  model : MiniGPT,
  train_ids : Array[Int],
  val_ids : Array[Int],
  config : TrainingConfig,
  rng : @random.Rand,
  callbacks : TrainingCallbacks,
) -> TrainingStats {
  let losses : Array[Double] = []
  let params = model.parameters()
  let optimizer_config = @optim.AdamWConfig(
    config.learning_rate,
    beta1=config.beta1,
    beta2=config.beta2,
    eps=1.0e-8,
  )
  let optimizer = @optim.AdamW::with_parameter_weight_decays(
    params,
    optimizer_config,
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
      (callbacks.on_eval)({ iter_num, train_loss, val_loss })
      if initial_eval_loss != initial_eval_loss {
        initial_eval_loss = val_loss
      }
      final_eval_loss = val_loss
      let improved = val_loss < best_val_loss
      if improved {
        best_val_loss = val_loss
      }
      if iter_num > 0 && (improved || config.always_save_checkpoint) {
        (callbacks.on_checkpoint)(
          TrainingState(model, optimizer, iter_num, best_val_loss, config),
        )
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
    model.zero_grad()
    loss.backward()
    model.clear_graph()
    optimizer.step_with_grad_clip(config.grad_clip)
    if iter_num % config.log_interval == 0 {
      (callbacks.on_log)({ iter_num, loss: loss_value, learning_rate: lr })
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
