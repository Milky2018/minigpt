///|
pub const BATCH_SIZE : Int = 4

///|
pub const BLOCK_SIZE : Int = 8

///|
pub const TRAINING_STEPS : Int = 10

///|
pub const LEARNING_RATE : Double = 1.0e-3

///|
pub struct TrainingConfig {
  batch_size : Int
  block_size : Int
  steps : Int
  learning_rate : Double
}

///|
pub fn TrainingConfig::new(
  batch_size : Int,
  block_size : Int,
  steps : Int,
  learning_rate : Double,
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
  { batch_size, block_size, steps, learning_rate }
}

///|
pub fn TrainingConfig::recommended() -> TrainingConfig {
  TrainingConfig::new(BATCH_SIZE, BLOCK_SIZE, TRAINING_STEPS, LEARNING_RATE)
}

///|
pub struct TrainingStats {
  losses : Array[Double]
  initial_eval_loss : Double
  final_eval_loss : Double
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
fn eval_batch(
  token_ids : Array[Int],
  batch_size : Int,
  block_size : Int,
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
  for batch in 0..<batch_size {
    let start = if batch_size == 1 {
      0
    } else {
      batch * (limit - 1) / (batch_size - 1)
    }
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
) -> Double {
  let (inputs, targets) = eval_batch(
    token_ids,
    config.batch_size,
    config.block_size,
  )
  let loss = model.loss(inputs, targets).data()[0]
  model.clear_graph()
  loss
}

///|
pub fn train_token_ids(
  model : MiniGPT,
  token_ids : Array[Int],
  config : TrainingConfig,
  rng : @random.Rand,
) -> TrainingStats {
  let losses : Array[Double] = []
  let initial_eval_loss = eval_token_ids(model, token_ids, config)
  let optimizer = @optim.AdamW::new(model.parameters(), config.learning_rate)
  for _ in 0..<config.steps {
    let (inputs, targets) = sample_batch(
      token_ids,
      config.batch_size,
      config.block_size,
      rng,
    )
    let loss = model.loss(inputs, targets)
    losses.push(loss.data()[0])
    loss.backward()
    optimizer.step()
    model.clear_graph()
  }
  let final_eval_loss = eval_token_ids(model, token_ids, config)
  { losses, initial_eval_loss, final_eval_loss }
}
