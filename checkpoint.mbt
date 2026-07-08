///|
const CHECKPOINT_VERSION : Int = 1

///|
const CHECKPOINT_MAGIC : Int = 1296649799

///|
priv struct MiniGPTCheckpoint {
  version : Int
  model_kind : String?
  vocabulary : Array[String]
  n_embd : Int
  token_embedding_table : Array[Double]
  lm_head : Array[Double]
}

///|
fn MiniGPTCheckpoint::from_model(
  model : MiniGPT,
  tokenizer : CharTokenizer,
) -> MiniGPTCheckpoint {
  {
    version: CHECKPOINT_VERSION,
    model_kind: Some(model.kind_name()),
    vocabulary: tokenizer.vocabulary(),
    n_embd: model.n_embd(),
    token_embedding_table: model.token_embedding_data(),
    lm_head: model.lm_head_data(),
  }
}

///|
fn MiniGPTCheckpoint::to_model_and_tokenizer(
  self : MiniGPTCheckpoint,
) -> (MiniGPT, CharTokenizer) {
  if self.version != CHECKPOINT_VERSION {
    abort("unsupported MiniGPT checkpoint version")
  }
  let tokenizer = CharTokenizer::from_vocabulary(self.vocabulary)
  let model = match self.model_kind {
    Some("full-rank-bigram") =>
      MiniGPT::from_full_rank_parameters(
        tokenizer.vocab_size(),
        self.token_embedding_table,
      )
    _ =>
      MiniGPT::from_parameters(
        tokenizer.vocab_size(),
        self.n_embd,
        self.token_embedding_table,
        self.lm_head,
      )
  }
  (model, tokenizer)
}

///|
pub fn encode_checkpoint(model : MiniGPT, tokenizer : CharTokenizer) -> Bytes {
  let checkpoint = MiniGPTCheckpoint::from_model(model, tokenizer)
  let buffer = Buffer::Buffer()
  buffer.write_int_le(CHECKPOINT_MAGIC)
  buffer.write_int_le(CHECKPOINT_VERSION)
  write_string(buffer, model.kind_name())
  buffer.write_int_le(checkpoint.vocabulary.length())
  for item in checkpoint.vocabulary {
    write_string(buffer, item)
  }
  buffer.write_int_le(checkpoint.n_embd)
  write_doubles(buffer, checkpoint.token_embedding_table)
  write_doubles(buffer, checkpoint.lm_head)
  buffer.to_bytes()
}

///|
pub fn decode_checkpoint(bytes : Bytes) -> (MiniGPT, CharTokenizer) {
  let reader = BytesReader::new(bytes)
  let magic = reader.read_int_le()
  if magic != CHECKPOINT_MAGIC {
    abort("invalid MiniGPT checkpoint magic")
  }
  let version = reader.read_int_le()
  if version != CHECKPOINT_VERSION {
    abort("unsupported MiniGPT checkpoint version")
  }
  let model_kind = reader.read_string()
  let vocab_size = reader.read_int_le()
  let vocabulary : Array[String] = []
  for _ in 0..<vocab_size {
    vocabulary.push(reader.read_string())
  }
  let n_embd = reader.read_int_le()
  let token_embedding_table = reader.read_doubles()
  let lm_head = reader.read_doubles()
  if !reader.is_done() {
    abort("checkpoint contains trailing bytes")
  }
  let checkpoint : MiniGPTCheckpoint = {
    version,
    model_kind: Some(model_kind),
    vocabulary,
    n_embd,
    token_embedding_table,
    lm_head,
  }
  checkpoint.to_model_and_tokenizer()
}

///|
fn write_string(buffer : Buffer, text : String) -> Unit {
  let bytes = @utf8.encode(text[:])
  buffer.write_int_le(bytes.length())
  buffer.write_bytes(bytes[:])
}

///|
fn write_doubles(buffer : Buffer, values : Array[Double]) -> Unit {
  buffer.write_int_le(values.length())
  for value in values {
    buffer.write_double_le(value)
  }
}

///|
priv struct BytesReader {
  bytes : Bytes
  mut offset : Int
}

///|
fn BytesReader::new(bytes : Bytes) -> BytesReader {
  { bytes, offset: 0 }
}

///|
fn BytesReader::is_done(self : BytesReader) -> Bool {
  self.offset == self.bytes.length()
}

///|
fn BytesReader::read_byte(self : BytesReader) -> Int {
  if self.offset >= self.bytes.length() {
    abort("checkpoint ended unexpectedly")
  }
  let value = self.bytes[self.offset].to_int()
  self.offset += 1
  value
}

///|
fn BytesReader::read_int_le(self : BytesReader) -> Int {
  let b0 = self.read_byte()
  let b1 = self.read_byte()
  let b2 = self.read_byte()
  let b3 = self.read_byte()
  b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

///|
fn BytesReader::read_string(self : BytesReader) -> String {
  let length = self.read_int_le()
  if length < 0 || self.offset + length > self.bytes.length() {
    abort("invalid checkpoint string length")
  }
  let view = self.bytes.view(start=self.offset, end=self.offset + length)
  self.offset += length
  @utf8.decode(view) catch {
    _ => abort("invalid checkpoint UTF-8 string")
  }
}

///|
fn BytesReader::read_double_le(self : BytesReader) -> Double {
  let b0 = self.read_byte().to_int64()
  let b1 = self.read_byte().to_int64()
  let b2 = self.read_byte().to_int64()
  let b3 = self.read_byte().to_int64()
  let b4 = self.read_byte().to_int64()
  let b5 = self.read_byte().to_int64()
  let b6 = self.read_byte().to_int64()
  let b7 = self.read_byte().to_int64()
  let bits = b0 |
    (b1 << 8) |
    (b2 << 16) |
    (b3 << 24) |
    (b4 << 32) |
    (b5 << 40) |
    (b6 << 48) |
    (b7 << 56)
  bits.reinterpret_as_double()
}

///|
fn BytesReader::read_doubles(self : BytesReader) -> Array[Double] {
  let length = self.read_int_le()
  if length < 0 {
    abort("invalid checkpoint array length")
  }
  let values : Array[Double] = []
  for _ in 0..<length {
    values.push(self.read_double_le())
  }
  values
}
