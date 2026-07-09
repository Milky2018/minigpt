///|
const TOKENIZER_KIND_CHAR : String = "char"

///|
const TOKENIZER_KIND_WORD : String = "word"

///|
const TOKENIZER_KIND_BPE : String = "bpe"

///|
pub const DEFAULT_BPE_VOCAB_SIZE : Int = 512

///|
pub(all) enum TokenizerConfig {
  CharacterLevel
  WordLevel
  BpeLevel(Int)
} derive(Eq, Debug)

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
pub struct BpeMerge {
  priv left : String
  priv right : String
  priv merged : String
}

///|
pub fn BpeMerge::BpeMerge(
  left : String,
  right : String,
  merged : String,
) -> BpeMerge {
  { left, right, merged }
}

///|
pub fn BpeMerge::left(self : BpeMerge) -> String {
  self.left
}

///|
pub fn BpeMerge::right(self : BpeMerge) -> String {
  self.right
}

///|
pub fn BpeMerge::merged(self : BpeMerge) -> String {
  self.merged
}

///|
pub struct BpeTokenizer {
  priv tokens : Array[String]
  priv token_to_id : Map[String, Int]
  priv merges : Array[BpeMerge]
}

///|
pub(all) enum Tokenizer {
  Character(CharTokenizer)
  Word(WordTokenizer)
  Bpe(BpeTokenizer)
}

///|
pub struct TokenDataset {
  priv tokenizer : Tokenizer
  priv train_ids : Array[Int]
  priv val_ids : Array[Int]
}

///|
pub fn TokenDataset::tokenizer(self : TokenDataset) -> Tokenizer {
  self.tokenizer
}

///|
pub fn TokenDataset::train_ids(self : TokenDataset) -> Array[Int] {
  self.train_ids.copy()
}

///|
pub fn TokenDataset::val_ids(self : TokenDataset) -> Array[Int] {
  self.val_ids.copy()
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
  let buf = StringBuilder()
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
fn validate_ordered_tokens(tokens : Array[String]) -> Unit {
  if tokens.length() == 0 {
    abort("tokenizer vocabulary must not be empty")
  }
  let seen : Map[String, Unit] = {}
  for token in tokens {
    if token == "" {
      abort("tokenizer vocabulary must not contain empty tokens")
    }
    if seen.contains(token) {
      abort("tokenizer vocabulary must not contain duplicates")
    }
    seen[token] = ()
  }
}

///|
fn split_char_tokens(text : String) -> Array[String] {
  let tokens : Array[String] = []
  for ch in text {
    tokens.push(ch.to_string())
  }
  tokens
}

///|
fn join_word_tokens(tokens : Array[String]) -> String {
  let buf = StringBuilder()
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
fn join_tokens(tokens : Array[String]) -> String {
  let buf = StringBuilder()
  for token in tokens {
    buf.write_string(token)
  }
  buf.to_string()
}

///|
fn pair_key(left : String, right : String) -> String {
  left + "\u{1f}" + right
}

///|
priv struct BpePair {
  left : String
  right : String
  merged : String
}

///|
fn best_bpe_pair(
  tokens : Array[String],
  token_to_id : Map[String, Int],
) -> BpePair? {
  if tokens.length() < 2 {
    return None
  }
  let counts : Map[String, Int] = {}
  let lefts : Map[String, String] = {}
  let rights : Map[String, String] = {}
  for i in 0..<(tokens.length() - 1) {
    let left = tokens[i]
    let right = tokens[i + 1]
    let merged = left + right
    if !token_to_id.contains(merged) {
      let key = pair_key(left, right)
      counts[key] = counts.get_or_default(key, 0) + 1
      lefts[key] = left
      rights[key] = right
    }
  }
  let mut best_key = ""
  let mut best_count = 0
  for key, count in counts {
    if count > best_count ||
      (count == best_count && best_count > 0 && key.compare(best_key) < 0) {
      best_key = key
      best_count = count
    }
  }
  if best_count < 2 {
    return None
  }
  let left = lefts[best_key]
  let right = rights[best_key]
  Some({ left, right, merged: left + right })
}

///|
fn apply_bpe_merge(
  tokens : Array[String],
  left : String,
  right : String,
  merged : String,
) -> Array[String] {
  let out : Array[String] = []
  let mut i = 0
  while i < tokens.length() {
    if i + 1 < tokens.length() && tokens[i] == left && tokens[i + 1] == right {
      out.push(merged)
      i += 2
    } else {
      out.push(tokens[i])
      i += 1
    }
  }
  out
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
pub fn BpeTokenizer::from_vocabulary_and_merges(
  vocabulary : Array[String],
  merges : Array[BpeMerge],
) -> BpeTokenizer {
  validate_ordered_tokens(vocabulary)
  let token_to_id = token_map(vocabulary)
  for merge in merges {
    if !token_to_id.contains(merge.left) ||
      !token_to_id.contains(merge.right) ||
      !token_to_id.contains(merge.merged) {
      abort("BPE merge rule must reference vocabulary tokens")
    }
  }
  { tokens: vocabulary.copy(), token_to_id, merges: merges.copy() }
}

///|
pub fn BpeTokenizer::train(
  text : String,
  vocab_size : Int,
) -> (BpeTokenizer, Array[Int]) {
  if vocab_size <= 0 {
    abort("BPE vocabulary size must be positive")
  }
  let mut current = split_char_tokens(text)
  if current.length() == 0 {
    abort("BPE tokenizer training text must not be empty")
  }
  let vocabulary : Array[String] = []
  for ch in collect_sorted_chars(text) {
    vocabulary.push(ch.to_string())
  }
  let token_to_id = token_map(vocabulary)
  let merges : Array[BpeMerge] = []
  while vocabulary.length() < vocab_size {
    match best_bpe_pair(current, token_to_id) {
      Some(pair) => {
        current = apply_bpe_merge(current, pair.left, pair.right, pair.merged)
        token_to_id[pair.merged] = vocabulary.length()
        vocabulary.push(pair.merged)
        merges.push(BpeMerge(pair.left, pair.right, pair.merged))
      }
      None => break
    }
  }
  let tokenizer = BpeTokenizer::from_vocabulary_and_merges(vocabulary, merges)
  let ids = tokenizer.encode(text) catch {
    UnknownToken(_) => abort("trained tokenizer rejected training text")
    UnknownCharacter(_) => abort("trained tokenizer rejected training text")
    InvalidTokenId(_) => abort("unexpected token id error while encoding text")
  }
  (tokenizer, ids)
}

///|
pub fn BpeTokenizer::kind_name(_self : BpeTokenizer) -> String {
  TOKENIZER_KIND_BPE
}

///|
pub fn BpeTokenizer::vocab_size(self : BpeTokenizer) -> Int {
  self.tokens.length()
}

///|
pub fn BpeTokenizer::vocabulary(self : BpeTokenizer) -> Array[String] {
  self.tokens.copy()
}

///|
pub fn BpeTokenizer::merges(self : BpeTokenizer) -> Array[BpeMerge] {
  self.merges.copy()
}

///|
pub fn BpeTokenizer::encode(
  self : BpeTokenizer,
  text : String,
) -> Array[Int] raise TokenizerError {
  let mut tokens : Array[String] = []
  for ch in text {
    let token = ch.to_string()
    if !self.token_to_id.contains(token) {
      raise UnknownCharacter(ch)
    }
    tokens.push(token)
  }
  for merge in self.merges {
    tokens = apply_bpe_merge(tokens, merge.left, merge.right, merge.merged)
  }
  let ids : Array[Int] = []
  for token in tokens {
    match self.token_to_id.get(token) {
      Some(id) => ids.push(id)
      None => raise UnknownToken(token)
    }
  }
  ids
}

///|
pub fn BpeTokenizer::decode(
  self : BpeTokenizer,
  ids : Array[Int],
) -> String raise TokenizerError {
  let tokens : Array[String] = []
  for id in ids {
    match self.tokens.get(id) {
      Some(token) => tokens.push(token)
      None => raise InvalidTokenId(id)
    }
  }
  join_tokens(tokens)
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
pub fn Tokenizer::from_bpe(tokenizer : BpeTokenizer) -> Tokenizer {
  Bpe(tokenizer)
}

///|
pub fn Tokenizer::train_char(text : String) -> (Tokenizer, Array[Int]) {
  let (tokenizer, ids) = CharTokenizer::train(text)
  (Tokenizer::from_char(tokenizer), ids)
}

///|
pub fn Tokenizer::train_word(text : String) -> (Tokenizer, Array[Int]) {
  let (tokenizer, ids) = WordTokenizer::train(text)
  (Tokenizer::from_word(tokenizer), ids)
}

///|
pub fn Tokenizer::train_bpe(
  text : String,
  vocab_size : Int,
) -> (Tokenizer, Array[Int]) {
  let (tokenizer, ids) = BpeTokenizer::train(text, vocab_size)
  (Tokenizer::from_bpe(tokenizer), ids)
}

///|
pub fn Tokenizer::train(
  text : String,
  config : TokenizerConfig,
) -> (Tokenizer, Array[Int]) {
  match config {
    CharacterLevel => Tokenizer::train_char(text)
    WordLevel => Tokenizer::train_word(text)
    BpeLevel(vocab_size) => Tokenizer::train_bpe(text, vocab_size)
  }
}

///|
pub fn Tokenizer::kind_name(self : Tokenizer) -> String {
  match self {
    Character(tokenizer) => tokenizer.kind_name()
    Word(tokenizer) => tokenizer.kind_name()
    Bpe(tokenizer) => tokenizer.kind_name()
  }
}

///|
pub fn Tokenizer::vocab_size(self : Tokenizer) -> Int {
  match self {
    Character(tokenizer) => tokenizer.vocab_size()
    Word(tokenizer) => tokenizer.vocab_size()
    Bpe(tokenizer) => tokenizer.vocab_size()
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
    Bpe(tokenizer) => tokenizer.vocabulary()
  }
}

///|
pub fn Tokenizer::bpe_merges(self : Tokenizer) -> Array[BpeMerge] {
  match self {
    Bpe(tokenizer) => tokenizer.merges()
    _ => []
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
    Bpe(tokenizer) => tokenizer.encode(text)
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
    Bpe(tokenizer) => tokenizer.decode(ids)
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
pub fn prepare_token_dataset(
  text : String,
  config : TokenizerConfig,
) -> TokenDataset {
  let (tokenizer, ids) = Tokenizer::train(text, config)
  let split = ids.length() * 9 / 10
  {
    tokenizer,
    train_ids: ids[0:split].to_owned(),
    val_ids: ids[split:ids.length()].to_owned(),
  }
}
