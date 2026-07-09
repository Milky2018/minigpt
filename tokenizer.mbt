///|
pub const TOKENIZER_KIND_CHAR : String = "char"

///|
pub const TOKENIZER_KIND_WORD : String = "word"

///|
pub(all) suberror TokenizerError {
  UnknownCharacter(Char)
  UnknownToken(String)
  InvalidTokenId(Int)
} derive(Eq, Debug)

///|
pub struct CharTokenizer {
  priv chars : Array[Char]
  priv char_to_id : Map[Char, Int]
}

///|
pub struct WordTokenizer {
  priv tokens : Array[String]
  priv token_to_id : Map[String, Int]
}

///|
pub(all) enum Tokenizer {
  Character(CharTokenizer)
  Word(WordTokenizer)
}

///|
pub struct TokenDataset {
  tokenizer : Tokenizer
  train_ids : Array[Int]
  val_ids : Array[Int]
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
fn split_word_tokens(text : String) -> Array[String] {
  let tokens : Array[String] = []
  let buf = StringBuilder::new()
  fn flush() -> Unit {
    if !buf.is_empty() {
      tokens.push(buf.to_string())
      buf.reset()
    }
  }
  for ch in text {
    match ch {
      '\n' => {
        flush()
        tokens.push("\n")
      }
      ' ' | '\t' | '\r' => flush()
      _ => buf.write_char(ch)
    }
  }
  flush()
  tokens
}

///|
fn collect_sorted_tokens(tokens : Array[String]) -> Array[String] {
  let seen : Map[String, Unit] = {}
  for token in tokens {
    seen[token] = ()
  }
  let vocabulary : Array[String] = []
  for token, _ in seen {
    vocabulary.push(token)
  }
  vocabulary.sort()
  vocabulary
}

///|
fn token_map(tokens : Array[String]) -> Map[String, Int] {
  let token_to_id : Map[String, Int] = {}
  for i in 0..<tokens.length() {
    token_to_id[tokens[i]] = i
  }
  token_to_id
}

///|
fn join_word_tokens(tokens : Array[String]) -> String {
  let buf = StringBuilder::new()
  let mut needs_space = false
  for token in tokens {
    if token == "\n" {
      buf.write_char('\n')
      needs_space = false
    } else {
      if needs_space {
        buf.write_char(' ')
      }
      buf.write_string(token)
      needs_space = true
    }
  }
  buf.to_string()
}

///|
fn single_char_token(text : String) -> Char {
  let chars : Array[Char] = []
  for ch in text {
    chars.push(ch)
  }
  if chars.length() != 1 {
    abort("character tokenizer vocabulary item must contain one character")
  }
  chars[0]
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
    UnknownToken(_) => abort("unexpected word tokenizer error")
    InvalidTokenId(_) => abort("unexpected token id error while encoding text")
  }
  (tokenizer, ids)
}

///|
pub fn WordTokenizer::from_tokens(tokens : Array[String]) -> WordTokenizer {
  if tokens.length() == 0 {
    abort("word tokenizer vocabulary must not be empty")
  }
  let sorted = tokens.copy()
  sorted.sort()
  for i in 0..<sorted.length() {
    if sorted[i] == "" {
      abort("word tokenizer vocabulary must not contain empty tokens")
    }
    if i > 0 && sorted[i - 1] == sorted[i] {
      abort("word tokenizer vocabulary must not contain duplicates")
    }
  }
  { tokens: sorted, token_to_id: token_map(sorted) }
}

///|
pub fn WordTokenizer::train(text : String) -> (WordTokenizer, Array[Int]) {
  let tokens = split_word_tokens(text)
  if tokens.length() == 0 {
    abort("word tokenizer training text must not be empty")
  }
  let tokenizer = WordTokenizer::from_tokens(collect_sorted_tokens(tokens))
  let ids = tokenizer.encode(text) catch {
    UnknownToken(_) => abort("trained tokenizer rejected training text")
    UnknownCharacter(_) => abort("unexpected character tokenizer error")
    InvalidTokenId(_) => abort("unexpected token id error while encoding text")
  }
  (tokenizer, ids)
}

///|
pub fn WordTokenizer::kind_name(_self : WordTokenizer) -> String {
  TOKENIZER_KIND_WORD
}

///|
pub fn WordTokenizer::vocab_size(self : WordTokenizer) -> Int {
  self.tokens.length()
}

///|
pub fn WordTokenizer::vocabulary(self : WordTokenizer) -> Array[String] {
  self.tokens.copy()
}

///|
pub fn WordTokenizer::encode(
  self : WordTokenizer,
  text : String,
) -> Array[Int] raise TokenizerError {
  let ids : Array[Int] = []
  for token in split_word_tokens(text) {
    match self.token_to_id.get(token) {
      Some(id) => ids.push(id)
      None => raise UnknownToken(token)
    }
  }
  ids
}

///|
pub fn WordTokenizer::decode(
  self : WordTokenizer,
  ids : Array[Int],
) -> String raise TokenizerError {
  let tokens : Array[String] = []
  for id in ids {
    match self.tokens.get(id) {
      Some(token) => tokens.push(token)
      None => raise InvalidTokenId(id)
    }
  }
  join_word_tokens(tokens)
}

///|
pub fn Tokenizer::from_char(tokenizer : CharTokenizer) -> Tokenizer {
  Character(tokenizer)
}

///|
pub fn Tokenizer::from_word(tokenizer : WordTokenizer) -> Tokenizer {
  Word(tokenizer)
}

///|
pub fn Tokenizer::from_kind_and_vocabulary(
  kind : String,
  vocabulary : Array[String],
) -> Tokenizer {
  match kind {
    "char" => {
      let chars : Array[Char] = []
      for token in vocabulary {
        chars.push(single_char_token(token))
      }
      Character(CharTokenizer::from_chars(chars))
    }
    "word" => Word(WordTokenizer::from_tokens(vocabulary))
    _ => abort("unsupported tokenizer kind: \{kind}")
  }
}

///|
pub fn Tokenizer::kind_name(self : Tokenizer) -> String {
  match self {
    Character(tokenizer) => tokenizer.kind_name()
    Word(tokenizer) => tokenizer.kind_name()
  }
}

///|
pub fn Tokenizer::vocab_size(self : Tokenizer) -> Int {
  match self {
    Character(tokenizer) => tokenizer.vocab_size()
    Word(tokenizer) => tokenizer.vocab_size()
  }
}

///|
pub fn Tokenizer::vocabulary(self : Tokenizer) -> Array[String] {
  match self {
    Character(tokenizer) => {
      let vocabulary : Array[String] = []
      for ch in tokenizer.vocabulary() {
        vocabulary.push(ch.to_string())
      }
      vocabulary
    }
    Word(tokenizer) => tokenizer.vocabulary()
  }
}

///|
pub fn Tokenizer::encode(
  self : Tokenizer,
  text : String,
) -> Array[Int] raise TokenizerError {
  match self {
    Character(tokenizer) => tokenizer.encode(text)
    Word(tokenizer) => tokenizer.encode(text)
  }
}

///|
pub fn Tokenizer::decode(
  self : Tokenizer,
  ids : Array[Int],
) -> String raise TokenizerError {
  match self {
    Character(tokenizer) => tokenizer.decode(ids)
    Word(tokenizer) => tokenizer.decode(ids)
  }
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

///|
pub fn prepare_token_dataset(
  text : String,
  tokenizer_kind : String,
) -> TokenDataset {
  let (tokenizer, ids) = match tokenizer_kind {
    "char" => {
      let (tokenizer, ids) = CharTokenizer::train(text)
      (Tokenizer::from_char(tokenizer), ids)
    }
    "word" => {
      let (tokenizer, ids) = WordTokenizer::train(text)
      (Tokenizer::from_word(tokenizer), ids)
    }
    _ => abort("unsupported tokenizer kind: \{tokenizer_kind}")
  }
  let split = ids.length() * 9 / 10
  {
    tokenizer,
    train_ids: ids[0:split].to_owned(),
    val_ids: ids[split:ids.length()].to_owned(),
  }
}
