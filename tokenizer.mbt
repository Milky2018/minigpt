///|
const TOKENIZER_KIND_CHAR : String = "char"

///|
pub(all) suberror TokenizerError {
  UnknownCharacter(Char)
  InvalidTokenId(Int)
} derive(Eq, Debug)

///|
pub struct CharTokenizer {
  priv chars : Array[Char]
  priv char_to_id : Map[Char, Int]
}

///|
pub struct ShakespeareCharDataset {
  tokenizer : CharTokenizer
  train_ids : Array[Int]
  val_ids : Array[Int]
}

///|
fn collect_sorted_chars(text : String) -> Array[Char] {
  let seen : Map[Char, Unit] = {}
  for ch in text {
    seen[ch] = ()
  }
  let chars : Array[Char] = []
  for ch, _ in seen {
    chars.push(ch)
  }
  chars.sort()
  chars
}

///|
fn char_map(chars : Array[Char]) -> Map[Char, Int] {
  let char_to_id : Map[Char, Int] = {}
  for i in 0..<chars.length() {
    char_to_id[chars[i]] = i
  }
  char_to_id
}

///|
pub fn CharTokenizer::from_chars(chars : Array[Char]) -> CharTokenizer {
  if chars.length() == 0 {
    abort("character tokenizer vocabulary must not be empty")
  }
  let sorted = chars.copy()
  sorted.sort()
  for i in 1..<sorted.length() {
    if sorted[i - 1] == sorted[i] {
      abort("character tokenizer vocabulary must not contain duplicates")
    }
  }
  { chars: sorted, char_to_id: char_map(sorted) }
}

///|
pub fn CharTokenizer::train(text : String) -> (CharTokenizer, Array[Int]) {
  if text.char_length() == 0 {
    abort("character tokenizer training text must not be empty")
  }
  let tokenizer = CharTokenizer::from_chars(collect_sorted_chars(text))
  let ids = tokenizer.encode(text) catch {
    UnknownCharacter(_) => abort("trained tokenizer rejected training text")
    InvalidTokenId(_) => abort("unexpected token id error while encoding text")
  }
  (tokenizer, ids)
}

///|
pub fn CharTokenizer::kind_name(_self : CharTokenizer) -> String {
  TOKENIZER_KIND_CHAR
}

///|
pub fn CharTokenizer::vocab_size(self : CharTokenizer) -> Int {
  self.chars.length()
}

///|
pub fn CharTokenizer::vocabulary(self : CharTokenizer) -> Array[Char] {
  self.chars.copy()
}

///|
pub fn CharTokenizer::encode(
  self : CharTokenizer,
  text : String,
) -> Array[Int] raise TokenizerError {
  let ids : Array[Int] = []
  for ch in text {
    match self.char_to_id.get(ch) {
      Some(id) => ids.push(id)
      None => raise UnknownCharacter(ch)
    }
  }
  ids
}

///|
pub fn CharTokenizer::decode(
  self : CharTokenizer,
  ids : Array[Int],
) -> String raise TokenizerError {
  let chars : Array[Char] = []
  for id in ids {
    match self.chars.get(id) {
      Some(ch) => chars.push(ch)
      None => raise InvalidTokenId(id)
    }
  }
  String::from_array(chars)
}

///|
pub fn prepare_shakespeare_char_dataset(
  text : String,
) -> ShakespeareCharDataset {
  let (tokenizer, ids) = CharTokenizer::train(text)
  let split = ids.length() * 9 / 10
  {
    tokenizer,
    train_ids: ids[0:split].to_owned(),
    val_ids: ids[split:ids.length()].to_owned(),
  }
}
