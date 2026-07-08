///|
pub(all) suberror TokenizerError {
  UnknownChar(Char)
  InvalidTokenId(Int)
} derive(Eq, Debug)

///|
pub struct CharTokenizer {
  char_to_id : Map[Char, Int]
  id_to_char : Array[Char]
}

///|
pub fn CharTokenizer::from_text(text : String) -> CharTokenizer {
  let seen : Map[Char, Bool] = Map::default()
  let chars : Array[Char] = Array::new()
  text
  .iter()
  .each(ch => {
    if !seen.contains(ch) {
      seen.set(ch, true)
      chars.push(ch)
    }
  })
  chars.sort()
  let char_to_id : Map[Char, Int] = Map::default()
  chars.eachi(fn(index, ch) { char_to_id.set(ch, index) })
  { char_to_id, id_to_char: chars }
}

///|
pub fn CharTokenizer::from_vocabulary(
  vocabulary : Array[String],
) -> CharTokenizer {
  if vocabulary.length() == 0 {
    abort("vocabulary must not be empty")
  }
  let char_to_id : Map[Char, Int] = Map::default()
  let id_to_char : Array[Char] = []
  for i in 0..<vocabulary.length() {
    if vocabulary[i].char_length() != 1 {
      abort("each vocabulary item must contain exactly one character")
    }
    let ch = vocabulary[i].iter().next().unwrap()
    if char_to_id.contains(ch) {
      abort("vocabulary contains duplicate characters")
    }
    char_to_id.set(ch, i)
    id_to_char.push(ch)
  }
  { char_to_id, id_to_char }
}

///|
pub fn CharTokenizer::vocabulary(self : CharTokenizer) -> Array[String] {
  self.id_to_char.map(ch => String::from_array([ch]))
}

///|
pub fn CharTokenizer::vocab_size(self : CharTokenizer) -> Int {
  self.id_to_char.length()
}

///|
pub fn CharTokenizer::encode(
  self : CharTokenizer,
  text : String,
) -> Array[Int] raise TokenizerError {
  let ids : Array[Int] = Array::new(capacity=text.char_length())
  text
  .iter()
  .each(ch => {
    match self.char_to_id.get(ch) {
      Some(id) => ids.push(id)
      None => raise UnknownChar(ch)
    }
  })
  ids
}

///|
pub fn CharTokenizer::decode(
  self : CharTokenizer,
  ids : Array[Int],
) -> String raise TokenizerError {
  let chars : Array[Char] = Array::new(capacity=ids.length())
  ids.each(id => {
    match self.id_to_char.get(id) {
      Some(ch) => chars.push(ch)
      None => raise InvalidTokenId(id)
    }
  })
  String::from_array(chars)
}
