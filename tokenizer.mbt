///|
pub const BPE_VOCAB_SIZE : Int = 512

///|
const BYTE_VOCAB_SIZE : Int = 256

///|
const TOKENIZER_KIND_BPE : String = "byte-bpe"

///|
pub(all) suberror TokenizerError {
  InvalidTokenId(Int)
} derive(Eq, Debug)

///|
pub(all) struct BpeMerge {
  left : Int
  right : Int
} derive(Eq, Debug)

///|
priv struct PairKey {
  left : Int
  right : Int
} derive(Eq, Hash)

///|
priv struct BestPair {
  left : Int
  right : Int
}

///|
pub struct BpeTokenizer {
  priv vocabulary : Array[Bytes]
  priv merges : Array[BpeMerge]
  priv sampleable_token_ids : Array[Int]
}

///|
fn byte_vocabulary() -> Array[Bytes] {
  let vocabulary : Array[Bytes] = []
  for i in 0..<BYTE_VOCAB_SIZE {
    vocabulary.push(Bytes::from_array([i.to_byte()]))
  }
  vocabulary
}

///|
fn encode_utf8_bytes(text : String) -> Array[Int] {
  let bytes = @utf8.encode(text[:])
  let ids : Array[Int] = Array::new(capacity=bytes.length())
  for byte in bytes {
    ids.push(byte.to_int())
  }
  ids
}

///|
fn full_byte_token_ids() -> Array[Int] {
  let ids : Array[Int] = []
  for i in 0..<BYTE_VOCAB_SIZE {
    ids.push(i)
  }
  ids
}

///|
fn collect_sampleable_token_ids(
  tokens : Array[Int],
  vocab_size : Int,
) -> Array[Int] {
  let seen = Array::make(vocab_size, false)
  for token in tokens {
    if token < 0 || token >= vocab_size {
      abort("BPE token id out of vocabulary range")
    }
    seen[token] = true
  }
  let ids : Array[Int] = []
  for id in 0..<seen.length() {
    if seen[id] {
      ids.push(id)
    }
  }
  ids
}

///|
fn pair_is_better(
  left : Int,
  right : Int,
  count : Int,
  best_left : Int,
  best_right : Int,
  best_count : Int,
) -> Bool {
  count > best_count ||
  (
    count == best_count &&
    (left < best_left || (left == best_left && right < best_right))
  )
}

///|
fn most_frequent_pair(tokens : Array[Int]) -> BestPair? {
  if tokens.length() < 2 {
    return None
  }
  let counts : Map[PairKey, Int] = {}
  for i in 0..<(tokens.length() - 1) {
    let pair : PairKey = { left: tokens[i], right: tokens[i + 1] }
    counts[pair] = counts.get_or_default(pair, 0) + 1
  }
  let mut best_left = @int.MAX_VALUE
  let mut best_right = @int.MAX_VALUE
  let mut best_count = 0
  for pair, count in counts {
    if pair_is_better(
        pair.left,
        pair.right,
        count,
        best_left,
        best_right,
        best_count,
      ) {
      best_left = pair.left
      best_right = pair.right
      best_count = count
    }
  }
  if best_count < 2 {
    None
  } else {
    Some({ left: best_left, right: best_right })
  }
}

///|
fn replace_pair(
  tokens : Array[Int],
  left : Int,
  right : Int,
  new_id : Int,
) -> Array[Int] {
  let output : Array[Int] = Array::new(capacity=tokens.length())
  let mut i = 0
  while i < tokens.length() {
    if i + 1 < tokens.length() && tokens[i] == left && tokens[i + 1] == right {
      output.push(new_id)
      i += 2
    } else {
      output.push(tokens[i])
      i += 1
    }
  }
  output
}

///|
pub fn BpeTokenizer::kind_name(_self : BpeTokenizer) -> String {
  TOKENIZER_KIND_BPE
}

///|
pub fn BpeTokenizer::train(
  text : String,
  vocab_size : Int,
) -> (BpeTokenizer, Array[Int]) {
  if vocab_size < BYTE_VOCAB_SIZE {
    abort("BPE vocab_size must be at least 256")
  }
  let vocabulary = byte_vocabulary()
  let merges : Array[BpeMerge] = []
  let mut tokens = encode_utf8_bytes(text)
  if tokens.length() == 0 {
    abort("BPE training text must not be empty")
  }
  while vocabulary.length() < vocab_size {
    match most_frequent_pair(tokens) {
      Some(best) => {
        let piece = vocabulary[best.left] + vocabulary[best.right]
        let new_id = vocabulary.length()
        vocabulary.push(piece)
        merges.push({ left: best.left, right: best.right })
        tokens = replace_pair(tokens, best.left, best.right, new_id)
      }
      None => break
    }
  }
  let sampleable_token_ids = collect_sampleable_token_ids(
    tokens,
    vocabulary.length(),
  )
  ({ vocabulary, merges, sampleable_token_ids }, tokens)
}

///|
pub fn BpeTokenizer::from_parts(
  vocabulary : Array[Bytes],
  merges : Array[BpeMerge],
  sampleable_token_ids : Array[Int],
) -> BpeTokenizer {
  if vocabulary.length() != BYTE_VOCAB_SIZE + merges.length() {
    abort("BPE vocabulary length must equal 256 plus merge count")
  }
  for i in 0..<BYTE_VOCAB_SIZE {
    if vocabulary[i] != Bytes::from_array([i.to_byte()]) {
      abort("BPE vocabulary must start with byte tokens")
    }
  }
  for i in 0..<merges.length() {
    let merge = merges[i]
    if merge.left < 0 ||
      merge.left >= BYTE_VOCAB_SIZE + i ||
      merge.right < 0 ||
      merge.right >= BYTE_VOCAB_SIZE + i {
      abort("BPE merge references an unknown token")
    }
    let expected = vocabulary[merge.left] + vocabulary[merge.right]
    if vocabulary[BYTE_VOCAB_SIZE + i] != expected {
      abort("BPE vocabulary does not match merge list")
    }
  }
  if sampleable_token_ids.length() == 0 {
    abort("BPE sampleable token ids must not be empty")
  }
  for id in sampleable_token_ids {
    if id < 0 || id >= vocabulary.length() {
      abort("BPE sampleable token id out of vocabulary range")
    }
  }
  {
    vocabulary: vocabulary.copy(),
    merges: merges.copy(),
    sampleable_token_ids: sampleable_token_ids.copy(),
  }
}

///|
pub fn BpeTokenizer::byte_level() -> BpeTokenizer {
  {
    vocabulary: byte_vocabulary(),
    merges: [],
    sampleable_token_ids: full_byte_token_ids(),
  }
}

///|
pub fn BpeTokenizer::vocab_size(self : BpeTokenizer) -> Int {
  self.vocabulary.length()
}

///|
pub fn BpeTokenizer::vocabulary(self : BpeTokenizer) -> Array[Bytes] {
  self.vocabulary.copy()
}

///|
pub fn BpeTokenizer::merges(self : BpeTokenizer) -> Array[BpeMerge] {
  self.merges.copy()
}

///|
pub fn BpeTokenizer::sampleable_token_ids(self : BpeTokenizer) -> Array[Int] {
  self.sampleable_token_ids.copy()
}

///|
pub fn BpeTokenizer::encode(self : BpeTokenizer, text : String) -> Array[Int] {
  let mut tokens = encode_utf8_bytes(text)
  for i in 0..<self.merges.length() {
    let merge = self.merges[i]
    tokens = replace_pair(tokens, merge.left, merge.right, BYTE_VOCAB_SIZE + i)
  }
  tokens
}

///|
pub fn BpeTokenizer::decode(
  self : BpeTokenizer,
  ids : Array[Int],
) -> String raise TokenizerError {
  let buffer = Buffer::Buffer()
  for id in ids {
    match self.vocabulary.get(id) {
      Some(piece) => buffer.write_bytes(piece)
      None => raise InvalidTokenId(id)
    }
  }
  @utf8.decode_lossy(buffer.to_bytes()[:])
}
