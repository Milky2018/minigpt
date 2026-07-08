///|
test "MiniGPT forward and loss use next-token shapes" {
  let rng = @random.Rand::new()
  let model = MiniGPT::new_gpt_with_options(4, 12, 3, 1, 4, rng)
  let inputs = @tensor.TokenIds::new([0, 1, 2, 3], [2, 2])
  let targets = @tensor.TokenIds::new([1, 2, 3, 0], [2, 2])
  @test.assert_eq(model.kind_name(), "gpt-transformer")
  @test.assert_eq(model.n_head(), 3)
  @test.assert_eq(model.n_layer(), 1)
  let logits = model.forward(inputs)
  @test.assert_eq(logits.shape(), [2, 2, 4])
  let loss = model.loss(inputs, targets)
  @test.assert_eq(loss.shape(), [])
  assert_true(loss.data()[0] == loss.data()[0])
}

///|
test "sample_batch returns shifted input and target windows" {
  let rng = @random.Rand::new()
  let (inputs, targets) = sample_batch([0, 1, 2, 3], 1, 3, rng)
  @test.assert_eq(inputs.shape(), [1, 3])
  @test.assert_eq(targets.shape(), [1, 3])
  @test.assert_eq(inputs.data(), [0, 1, 2])
  @test.assert_eq(targets.data(), [1, 2, 3])
}

///|
test "train_token_ids records one loss per step" {
  let rng = @random.Rand::new()
  let model = MiniGPT::new_gpt_with_options(2, 6, 3, 1, 2, rng)
  let stats = train_token_ids(
    model,
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1],
    TrainingConfig::new(
      2,
      2,
      3,
      0.1,
      eval_interval=100,
      eval_iters=1,
      log_interval=100,
    ),
    rng,
  )
  @test.assert_eq(stats.losses.length(), 3)
  assert_true(stats.initial_eval_loss == stats.initial_eval_loss)
  assert_true(stats.final_eval_loss == stats.final_eval_loss)
  for loss in stats.losses {
    assert_true(loss == loss)
  }
}

///|
test "recommended training config uses public constants" {
  let config = TrainingConfig::recommended()
  @test.assert_eq(config.batch_size, BATCH_SIZE)
  @test.assert_eq(config.block_size, BLOCK_SIZE)
  @test.assert_eq(config.steps, TRAINING_STEPS)
  @test.assert_eq(config.learning_rate, LEARNING_RATE)
  @test.assert_eq(config.min_lr, MIN_LR)
  @test.assert_eq(config.warmup_iters, WARMUP_ITERS)
  @test.assert_eq(config.eval_interval, EVAL_INTERVAL)
  @test.assert_eq(config.eval_iters, EVAL_ITERS)
  @test.assert_eq(config.log_interval, LOG_INTERVAL)
  @test.assert_eq(config.weight_decay, WEIGHT_DECAY)
  @test.assert_eq(config.beta1, BETA1)
  @test.assert_eq(config.beta2, BETA2)
  @test.assert_eq(config.grad_clip, GRAD_CLIP)
}

///|
test "generate_token_ids extends prompt" {
  let rng = @random.Rand::new()
  let model = MiniGPT::new_gpt_with_options(3, 6, 3, 1, 4, rng)
  let output = generate_token_ids(model, [0, 1], 4, rng)
  @test.assert_eq(output.length(), 6)
  @test.assert_eq(output[0:2].to_owned(), [0, 1])
  for id in output {
    assert_true(id >= 0 && id < 3)
  }
}
