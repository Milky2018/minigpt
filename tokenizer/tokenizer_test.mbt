///|
test "character tokenizer roundtrips Shakespeare-style text" {
  let text = "To be\nor not to be\n"
  let (tokenizer, ids) = @tokenizer.CharTokenizer::train(text)
  @test.assert_eq(tokenizer.decode(ids), text)
  @test.assert_eq(tokenizer.decode(tokenizer.encode(text)), text)
}

///|
test "character tokenizer uses sorted vocabulary" {
  let tokenizer = @tokenizer.CharTokenizer::from_chars(['b', 'a', '\n'])
  @test.assert_eq(tokenizer.vocab_size(), 3)
  @test.assert_eq(tokenizer.vocabulary(), ['\n', 'a', 'b'])
  @test.assert_eq(tokenizer.encode("ba\n"), [2, 1, 0])
  @test.assert_eq(tokenizer.decode([2, 1, 0]), "ba\n")
}

///|
test "character tokenizer rejects unknown characters" {
  let (tokenizer, _) = @tokenizer.CharTokenizer::train("ab")
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
  let (tokenizer, _) = @tokenizer.CharTokenizer::train("ab")
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
test "prepare_token_dataset uses nanoGPT split" {
  let dataset = @tokenizer.prepare_token_dataset("abcdefghij", CharacterLevel)
  @test.assert_eq(dataset.train_ids.length(), 9)
  @test.assert_eq(dataset.val_ids.length(), 1)
  @test.assert_eq(dataset.tokenizer.vocab_size(), 10)
}

///|
test "Tokenizer train helpers return unified tokenizer" {
  let (char_tokenizer, char_ids) = @tokenizer.Tokenizer::train_char("abca")
  @test.assert_eq(char_tokenizer.kind_name(), "char")
  @test.assert_eq(char_tokenizer.decode(char_ids), "abca")
  let (word_tokenizer, word_ids) = @tokenizer.Tokenizer::train_word("heat oil")
  @test.assert_eq(word_tokenizer.kind_name(), "word")
  @test.assert_eq(word_tokenizer.decode(word_ids), "heat oil")
  let (bpe_tokenizer, bpe_ids) = @tokenizer.Tokenizer::train_bpe("abababab", 4)
  @test.assert_eq(bpe_tokenizer.kind_name(), "bpe")
  @test.assert_eq(bpe_tokenizer.decode(bpe_ids), "abababab")
  let (generic_tokenizer, generic_ids) = @tokenizer.Tokenizer::train(
    "abcd",
    CharacterLevel,
  )
  @test.assert_eq(generic_tokenizer.kind_name(), "char")
  @test.assert_eq(generic_tokenizer.decode(generic_ids), "abcd")
  let (bpe_config_tokenizer, bpe_config_ids) = @tokenizer.Tokenizer::train(
    "abababab",
    BpeLevel(4),
  )
  @test.assert_eq(bpe_config_tokenizer.kind_name(), "bpe")
  @test.assert_eq(bpe_config_tokenizer.decode(bpe_config_ids), "abababab")
}

///|
test "word tokenizer roundtrips whitespace separated tokens" {
  let text = "heat olive oil\nadd salt\n"
  let (tokenizer, ids) = @tokenizer.WordTokenizer::train(text)
  @test.assert_eq(tokenizer.kind_name(), "word")
  @test.assert_eq(tokenizer.vocabulary(), [
    "\n", "add", "oil", "heat", "salt", "olive",
  ])
  @test.assert_eq(tokenizer.decode(ids), "heat olive oil\nadd salt\n")
  @test.assert_eq(tokenizer.encode("heat oil"), [3, 2])
}

///|
test "word tokenizer rejects unknown tokens" {
  let (tokenizer, _) = @tokenizer.WordTokenizer::train("heat oil")
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
  let dataset = @tokenizer.prepare_token_dataset(
    "heat oil\nadd salt\n",
    WordLevel,
  )
  @test.assert_eq(dataset.tokenizer.kind_name(), "word")
  @test.assert_eq(dataset.tokenizer.vocab_size(), 5)
  @test.assert_eq(dataset.train_ids.length(), 5)
  @test.assert_eq(dataset.val_ids.length(), 1)
}

///|
test "BPE tokenizer learns frequent adjacent pairs" {
  let (tokenizer, ids) = @tokenizer.BpeTokenizer::train("abababab", 4)
  @test.assert_eq(tokenizer.kind_name(), "bpe")
  @test.assert_eq(tokenizer.vocabulary(), ["a", "b", "ab", "abab"])
  @test.assert_eq(tokenizer.merges().length(), 2)
  @test.assert_eq(ids, [3, 3])
  @test.assert_eq(tokenizer.decode(ids), "abababab")
  @test.assert_eq(tokenizer.encode("abab"), [3])
}

///|
test "BPE tokenizer rejects unseen characters" {
  let (tokenizer, _) = @tokenizer.BpeTokenizer::train("abababab", 4)
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
test "prepare_token_dataset can use BPE tokenizer" {
  let dataset = @tokenizer.prepare_token_dataset(
    "abababababababab",
    BpeLevel(4),
  )
  @test.assert_eq(dataset.tokenizer.kind_name(), "bpe")
  @test.assert_eq(dataset.tokenizer.vocab_size(), 4)
  @test.assert_eq(dataset.train_ids.length(), 3)
  @test.assert_eq(dataset.val_ids.length(), 1)
}
