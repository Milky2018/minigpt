///|
test "train_text exposes a small library entrypoint" {
  let rng = @random.Rand::new()
  let config = @minigpt.TrainingConfig::new(
    1,
    2,
    0,
    0.001,
    eval_iters=1,
    log_interval=100,
  )
  let result = @minigpt.train_text_with_options(
    "abababababababababababababababababababab",
    CharacterLevel,
    config,
    6,
    3,
    1,
    rng,
  )
  @test.assert_eq(result.tokenizer.vocab_size(), 2)
  @test.assert_eq(result.model.vocab_size(), 2)
  @test.assert_eq(result.model.block_size(), 2)
  @test.assert_eq(result.stats.losses.length(), 1)
}
