///|
test "checkpoint roundtrips tokenizer and model weights" {
  let rng = @random.Rand::new()
  let tokenizer = CharTokenizer::from_text("春江花月夜")
  let model = MiniGPT::new_with_options(tokenizer.vocab_size(), 3, rng)
  let bytes = encode_checkpoint(model, tokenizer)
  let (loaded_model, loaded_tokenizer) = decode_checkpoint(bytes)
  @test.assert_eq(loaded_tokenizer.vocabulary(), tokenizer.vocabulary())
  @test.assert_eq(loaded_model.vocab_size(), model.vocab_size())
  @test.assert_eq(loaded_model.n_embd(), model.n_embd())
  @test.assert_eq(loaded_model.block_size(), model.block_size())
  @test.assert_eq(loaded_model.kind_name(), "tiny-transformer")
  @test.assert_eq(
    loaded_model.token_embedding_data(),
    model.token_embedding_data(),
  )
  @test.assert_eq(
    loaded_model.position_embedding_data(),
    model.position_embedding_data(),
  )
  @test.assert_eq(loaded_model.wq_data(), model.wq_data())
  @test.assert_eq(loaded_model.wk_data(), model.wk_data())
  @test.assert_eq(loaded_model.wv_data(), model.wv_data())
  @test.assert_eq(loaded_model.wo_data(), model.wo_data())
  @test.assert_eq(loaded_model.lm_head_data(), model.lm_head_data())
}
