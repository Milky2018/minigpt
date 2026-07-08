///|
test "checkpoint roundtrips tokenizer and model weights" {
  let rng = @random.Rand::new()
  let tokenizer = CharTokenizer::from_text("春江花月夜")
  let model = MiniGPT::new_with_options(tokenizer.vocab_size(), 3, rng)
  let text = encode_checkpoint(model, tokenizer)
  let (loaded_model, loaded_tokenizer) = decode_checkpoint(text)
  @test.assert_eq(loaded_tokenizer.vocabulary(), tokenizer.vocabulary())
  @test.assert_eq(loaded_model.vocab_size(), model.vocab_size())
  @test.assert_eq(loaded_model.n_embd(), model.n_embd())
  @test.assert_eq(
    loaded_model.token_embedding_data(),
    model.token_embedding_data(),
  )
  @test.assert_eq(loaded_model.lm_head_data(), model.lm_head_data())
}
