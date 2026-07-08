///|
fn assert_int_array_eq(actual : Array[Int], expected : Array[Int]) -> Unit {
  if actual.length() != expected.length() {
    abort("array lengths differ")
  }
  for i in 0..<actual.length() {
    if actual[i] != expected[i] {
      abort("arrays differ")
    }
  }
}

///|
test "MiniGPT forward and loss use next-token shapes" {
  let rng = @random.Rand::new()
  let model = MiniGPT::new(4, rng)
  let inputs = @tensor.TokenIds::new([0, 1, 2, 3], [2, 2])
  let targets = @tensor.TokenIds::new([1, 2, 3, 0], [2, 2])
  let logits = model.forward(inputs)
  assert_int_array_eq(logits.shape(), [2, 2, 4])
  let loss = model.loss(inputs, targets)
  assert_int_array_eq(loss.shape(), [])
  assert_true(loss.data()[0] == loss.data()[0])
}

///|
test "sample_batch returns shifted input and target windows" {
  let rng = @random.Rand::new()
  let (inputs, targets) = sample_batch([0, 1, 2, 3], 1, 3, rng)
  assert_int_array_eq(inputs.shape(), [1, 3])
  assert_int_array_eq(targets.shape(), [1, 3])
  assert_int_array_eq(inputs.data(), [0, 1, 2])
  assert_int_array_eq(targets.data(), [1, 2, 3])
}

///|
test "train_token_ids records one loss per step" {
  let rng = @random.Rand::new()
  let model = MiniGPT::new(2, rng)
  let stats = train_token_ids(
    model,
    [0, 1, 0, 1, 0, 1, 0, 1],
    TrainingConfig::new(2, 2, 3, 0.1),
    rng,
  )
  assert_eq(stats.losses.length(), 3)
  for loss in stats.losses {
    assert_true(loss == loss)
  }
}

///|
test "recommended training config uses public constants" {
  let config = TrainingConfig::recommended()
  assert_eq(config.batch_size, RECOMMENDED_BATCH_SIZE)
  assert_eq(config.block_size, RECOMMENDED_BLOCK_SIZE)
  assert_eq(config.steps, RECOMMENDED_TRAINING_STEPS)
  assert_true(config.learning_rate == RECOMMENDED_LEARNING_RATE)
}

///|
test "generate_token_ids extends prompt" {
  let rng = @random.Rand::new()
  let model = MiniGPT::new(3, rng)
  let output = generate_token_ids(model, [0, 1], 4, rng)
  assert_eq(output.length(), 6)
  assert_int_array_eq(output[0:2].to_owned(), [0, 1])
  for id in output {
    assert_true(id >= 0 && id < 3)
  }
}
