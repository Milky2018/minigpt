///|
test "char tokenizer roundtrips Shakespeare-style text" {
  let text = "To be\nor not to be\n"
  let tokenizer = @minigpt.CharTokenizer::from_text(text)
  let ids = tokenizer.encode(text)
  assert_eq(tokenizer.decode(ids), text)
}

///|
test "char tokenizer keeps newline as a token" {
  let tokenizer = @minigpt.CharTokenizer::from_text("A\nB")
  assert_eq(tokenizer.vocab_size(), 3)
  let ids = tokenizer.encode("\n")
  assert_eq(ids.length(), 1)
  assert_eq(tokenizer.decode(ids), "\n")
}

///|
test "char tokenizer assigns stable sorted ids" {
  let tokenizer = @minigpt.CharTokenizer::from_text("baba")
  let ids = tokenizer.encode("ab")
  assert_eq(ids.length(), 2)
  assert_eq(ids[0], 0)
  assert_eq(ids[1], 1)
}

///|
test "char tokenizer rejects unknown characters" {
  let tokenizer = @minigpt.CharTokenizer::from_text("ab")
  try {
    let _ = tokenizer.encode("ac")
  } catch {
    UnknownChar(ch) => assert_eq(ch, 'c')
    _ => fail("expected UnknownChar")
  } noraise {
    _ => fail("expected encode to raise")
  }
}

///|
test "char tokenizer rejects invalid token ids" {
  let tokenizer = @minigpt.CharTokenizer::from_text("ab")
  try {
    let _ = tokenizer.decode([0, 2])
  } catch {
    InvalidTokenId(id) => assert_eq(id, 2)
    _ => fail("expected InvalidTokenId")
  } noraise {
    _ => fail("expected decode to raise")
  }
}
