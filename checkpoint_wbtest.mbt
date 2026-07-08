///|
test "checkpoint roundtrips tokenizer and model weights" {
  let rng = @random.Rand::new()
  let (tokenizer, _) = BpeTokenizer::train("ROMEO ROMEO", 260)
  let model = MiniGPT::new_with_options(tokenizer.vocab_size(), 4, rng)
  let bytes = encode_checkpoint(model, tokenizer)
  let (loaded_model, loaded_tokenizer) = decode_checkpoint(bytes)
  @test.assert_eq(loaded_tokenizer.vocabulary(), tokenizer.vocabulary())
  @test.assert_eq(loaded_tokenizer.merges(), tokenizer.merges())
  @test.assert_eq(
    loaded_tokenizer.sampleable_token_ids(),
    tokenizer.sampleable_token_ids(),
  )
  @test.assert_eq(loaded_model.vocab_size(), model.vocab_size())
  @test.assert_eq(loaded_model.n_embd(), model.n_embd())
  @test.assert_eq(loaded_model.n_head(), model.n_head())
  @test.assert_eq(loaded_model.n_layer(), model.n_layer())
  @test.assert_eq(loaded_model.block_size(), model.block_size())
  @test.assert_eq(loaded_model.kind_name(), "gpt-transformer")
  let loaded = loaded_model.parameter_data()
  let original = model.parameter_data()
  @test.assert_eq(loaded.token_embedding_table, original.token_embedding_table)
  @test.assert_eq(
    loaded.position_embedding_table,
    original.position_embedding_table,
  )
  @test.assert_eq(loaded.ln1_weight, original.ln1_weight)
  @test.assert_eq(loaded.ln1_bias, original.ln1_bias)
  @test.assert_eq(loaded.wq, original.wq)
  @test.assert_eq(loaded.bq, original.bq)
  @test.assert_eq(loaded.wk, original.wk)
  @test.assert_eq(loaded.bk, original.bk)
  @test.assert_eq(loaded.wv, original.wv)
  @test.assert_eq(loaded.bv, original.bv)
  @test.assert_eq(loaded.wo, original.wo)
  @test.assert_eq(loaded.bo, original.bo)
  @test.assert_eq(loaded.ln2_weight, original.ln2_weight)
  @test.assert_eq(loaded.ln2_bias, original.ln2_bias)
  @test.assert_eq(loaded.w_fc, original.w_fc)
  @test.assert_eq(loaded.b_fc, original.b_fc)
  @test.assert_eq(loaded.w_proj, original.w_proj)
  @test.assert_eq(loaded.b_proj, original.b_proj)
  @test.assert_eq(loaded.ln_f_weight, original.ln_f_weight)
  @test.assert_eq(loaded.ln_f_bias, original.ln_f_bias)
}
