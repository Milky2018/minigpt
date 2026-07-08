///|
test "character tokenizer roundtrips Shakespeare-style text" {
  let text = "To be\nor not to be\n"
  let (tokenizer, ids) = @minigpt.CharTokenizer::train(text)
  @test.assert_eq(tokenizer.decode(ids), text)
  @test.assert_eq(tokenizer.decode(tokenizer.encode(text)), text)
}

///|
test "character tokenizer uses sorted vocabulary" {
  let tokenizer = @minigpt.CharTokenizer::from_chars(['b', 'a', '\n'])
  @test.assert_eq(tokenizer.vocab_size(), 3)
  @test.assert_eq(tokenizer.vocabulary(), ['\n', 'a', 'b'])
  @test.assert_eq(tokenizer.encode("ba\n"), [2, 1, 0])
  @test.assert_eq(tokenizer.decode([2, 1, 0]), "ba\n")
}

///|
test "character tokenizer rejects unknown characters" {
  let (tokenizer, _) = @minigpt.CharTokenizer::train("ab")
  try {
    let _ = tokenizer.encode("abc")
  } catch {
    UnknownCharacter(ch) => @test.assert_eq(ch, 'c')
    _ => fail("expected unknown character")
  } noraise {
    _ => fail("expected encode to raise")
  }
}

///|
test "character tokenizer rejects invalid token ids" {
  let (tokenizer, _) = @minigpt.CharTokenizer::train("ab")
  try {
    let _ = tokenizer.decode([0, 2])
  } catch {
    InvalidTokenId(id) => @test.assert_eq(id, 2)
    _ => fail("expected invalid token id")
  } noraise {
    _ => fail("expected decode to raise")
  }
}

///|
test "prepare_shakespeare_char_dataset uses nanoGPT split" {
  let dataset = @minigpt.prepare_shakespeare_char_dataset("abcdefghij")
  @test.assert_eq(dataset.train_ids.length(), 9)
  @test.assert_eq(dataset.val_ids.length(), 1)
  @test.assert_eq(dataset.tokenizer.vocab_size(), 10)
}
