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

///|
test "word tokenizer roundtrips whitespace separated tokens" {
  let text = "heat olive oil\nadd salt\n"
  let (tokenizer, ids) = @minigpt.WordTokenizer::train(text)
  @test.assert_eq(tokenizer.kind_name(), "word")
  @test.assert_eq(tokenizer.vocabulary(), [
    "\n", "add", "oil", "heat", "salt", "olive",
  ])
  @test.assert_eq(tokenizer.decode(ids), "heat olive oil\nadd salt\n")
  @test.assert_eq(tokenizer.encode("heat oil"), [3, 2])
}

///|
test "word tokenizer rejects unknown tokens" {
  let (tokenizer, _) = @minigpt.WordTokenizer::train("heat oil")
  try {
    let _ = tokenizer.encode("heat butter")
  } catch {
    UnknownToken(token) => @test.assert_eq(token, "butter")
    _ => fail("expected unknown token")
  } noraise {
    _ => fail("expected encode to raise")
  }
}

///|
test "prepare_token_dataset can use word tokenizer" {
  let dataset = @minigpt.prepare_token_dataset("heat oil\nadd salt\n", "word")
  @test.assert_eq(dataset.tokenizer.kind_name(), "word")
  @test.assert_eq(dataset.tokenizer.vocab_size(), 5)
  @test.assert_eq(dataset.train_ids.length(), 5)
  @test.assert_eq(dataset.val_ids.length(), 1)
}
