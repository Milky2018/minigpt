///|
test "BPE tokenizer roundtrips Shakespeare-style text" {
  let text = "To be\nor not to be\n"
  let (tokenizer, ids) = @minigpt.BpeTokenizer::train(text, 300)
  @test.assert_eq(tokenizer.decode(ids), text)
  @test.assert_eq(tokenizer.decode(tokenizer.encode(text)), text)
}

///|
test "BPE tokenizer starts from byte vocabulary" {
  let tokenizer = @minigpt.BpeTokenizer::byte_level()
  @test.assert_eq(tokenizer.vocab_size(), 256)
  @test.assert_eq(tokenizer.encode("A\n"), [65, 10])
  @test.assert_eq(tokenizer.decode([65, 10]), "A\n")
}

///|
test "BPE tokenizer learns deterministic merges" {
  let (tokenizer, ids) = @minigpt.BpeTokenizer::train("abababab", 257)
  @test.assert_eq(tokenizer.vocab_size(), 257)
  @test.assert_eq(ids, [256, 256, 256, 256])
  @test.assert_eq(tokenizer.encode("abab"), [256, 256])
  @test.assert_eq(tokenizer.decode([256, 256]), "abab")
}

///|
test "BPE tokenizer handles UTF-8 text" {
  let text = "中文 text"
  let (tokenizer, ids) = @minigpt.BpeTokenizer::train(text, 280)
  @test.assert_eq(tokenizer.decode(ids), text)
  @test.assert_eq(tokenizer.decode(tokenizer.encode(text)), text)
}

///|
test "BPE tokenizer rejects invalid token ids" {
  let tokenizer = @minigpt.BpeTokenizer::byte_level()
  try {
    let _ = tokenizer.decode([0, 256])
  } catch {
    InvalidTokenId(id) => @test.assert_eq(id, 256)
  } noraise {
    _ => fail("expected decode to raise")
  }
}
