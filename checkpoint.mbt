///|
const CHECKPOINT_VERSION : Int = 9

///|
const LEGACY_CHAR_CHECKPOINT_VERSION : Int = 8

///|
const CHECKPOINT_MAGIC : Int = 1296649799

///|
priv struct MiniGPTCheckpoint {
  version : Int
  model_kind : String?
  tokenizer_kind : String?
  vocabulary : Array[String]
  n_embd : Int
  n_head : Int
  n_layer : Int
  block_size : Int
  parameters : MiniGPTParameterData
  training_config : TrainingConfig?
  optimizer : @optim.AdamWCheckpoint?
  iter_num : Int
  best_val_loss : Double
}

///|
fn MiniGPTCheckpoint::from_model(
  model : MiniGPT,
  tokenizer : Tokenizer,
) -> MiniGPTCheckpoint {
  {
    version: CHECKPOINT_VERSION,
    model_kind: Some(model.kind_name()),
    tokenizer_kind: Some(tokenizer.kind_name()),
    vocabulary: tokenizer.vocabulary(),
    n_embd: model.n_embd(),
    n_head: model.n_head(),
    n_layer: model.n_layer(),
    block_size: model.block_size(),
    parameters: model.parameter_data(),
    training_config: None,
    optimizer: None,
    iter_num: 0,
    best_val_loss: 1.0e9,
  }
}

///|
fn MiniGPTCheckpoint::to_model_and_tokenizer(
  self : MiniGPTCheckpoint,
) -> (MiniGPT, Tokenizer) {
  if self.version != CHECKPOINT_VERSION &&
    self.version != LEGACY_CHAR_CHECKPOINT_VERSION {
    abort("unsupported MiniGPT checkpoint version")
  }
  if self.model_kind != Some("gpt-transformer") {
    abort("unsupported MiniGPT checkpoint model kind")
  }
  let tokenizer_kind = match self.tokenizer_kind {
    Some(kind) => kind
    None => abort("missing MiniGPT checkpoint tokenizer kind")
  }
  let tokenizer = Tokenizer::from_kind_and_vocabulary(
    tokenizer_kind,
    self.vocabulary,
  )
  let model = MiniGPT::from_parameter_data(
    tokenizer.vocab_size(),
    self.n_embd,
    self.n_head,
    self.n_layer,
    self.block_size,
    self.parameters,
  )
  (model, tokenizer)
}

///|
pub fn encode_checkpoint(model : MiniGPT, tokenizer : Tokenizer) -> Bytes {
  let checkpoint = MiniGPTCheckpoint::from_model(model, tokenizer)
  encode_checkpoint_data(checkpoint)
}

///|
pub fn encode_training_checkpoint(
  model : MiniGPT,
  tokenizer : Tokenizer,
  optimizer : @optim.AdamW,
  iter_num : Int,
  best_val_loss : Double,
  config : TrainingConfig,
) -> Bytes {
  let checkpoint = MiniGPTCheckpoint::from_model(model, tokenizer)
  encode_checkpoint_data({
    ..checkpoint,
    training_config: Some(config),
    optimizer: Some(optimizer.checkpoint()),
    iter_num,
    best_val_loss,
  })
}

///|
fn encode_checkpoint_data(checkpoint : MiniGPTCheckpoint) -> Bytes {
  let params = checkpoint.parameters
  let buffer = Buffer::Buffer()
  buffer.write_int_le(CHECKPOINT_MAGIC)
  buffer.write_int_le(CHECKPOINT_VERSION)
  write_string(buffer, checkpoint.model_kind.unwrap())
  write_string(buffer, checkpoint.tokenizer_kind.unwrap())
  buffer.write_int_le(checkpoint.vocabulary.length())
  for token in checkpoint.vocabulary {
    write_string(buffer, token)
  }
  buffer.write_int_le(checkpoint.n_embd)
  buffer.write_int_le(checkpoint.n_head)
  buffer.write_int_le(checkpoint.n_layer)
  buffer.write_int_le(checkpoint.block_size)
  write_doubles(buffer, params.token_embedding_table)
  write_doubles(buffer, params.position_embedding_table)
  write_double_arrays(buffer, params.ln1_weight)
  write_double_arrays(buffer, params.ln1_bias)
  write_double_arrays(buffer, params.wq)
  write_double_arrays(buffer, params.bq)
  write_double_arrays(buffer, params.wk)
  write_double_arrays(buffer, params.bk)
  write_double_arrays(buffer, params.wv)
  write_double_arrays(buffer, params.bv)
  write_double_arrays(buffer, params.wo)
  write_double_arrays(buffer, params.bo)
  write_double_arrays(buffer, params.ln2_weight)
  write_double_arrays(buffer, params.ln2_bias)
  write_double_arrays(buffer, params.w_fc)
  write_double_arrays(buffer, params.b_fc)
  write_double_arrays(buffer, params.w_proj)
  write_double_arrays(buffer, params.b_proj)
  write_doubles(buffer, params.ln_f_weight)
  write_doubles(buffer, params.ln_f_bias)
  buffer.write_int_le(checkpoint.iter_num)
  buffer.write_double_le(checkpoint.best_val_loss)
  match (checkpoint.training_config, checkpoint.optimizer) {
    (Some(config), Some(optimizer)) => {
      buffer.write_int_le(1)
      write_training_config(buffer, config)
      write_adamw_checkpoint(buffer, optimizer)
    }
    (None, None) => buffer.write_int_le(0)
    _ => abort("training checkpoint requires both config and optimizer")
  }
  buffer.to_bytes()
}

///|
pub fn decode_checkpoint(bytes : Bytes) -> (MiniGPT, Tokenizer) {
  let reader = BytesReader::new(bytes)
  let magic = reader.read_int_le()
  if magic != CHECKPOINT_MAGIC {
    abort("invalid MiniGPT checkpoint magic")
  }
  let version = reader.read_int_le()
  if version != CHECKPOINT_VERSION && version != LEGACY_CHAR_CHECKPOINT_VERSION {
    abort("unsupported MiniGPT checkpoint version")
  }
  let model_kind = reader.read_string()
  let tokenizer_kind = reader.read_string()
  let vocab_size = reader.read_int_le()
  let vocabulary : Array[String] = []
  for _ in 0..<vocab_size {
    vocabulary.push(reader.read_string())
  }
  let n_embd = reader.read_int_le()
  let n_head = reader.read_int_le()
  let n_layer = reader.read_int_le()
  let block_size = reader.read_int_le()
  let parameters : MiniGPTParameterData = {
    token_embedding_table: reader.read_doubles(),
    position_embedding_table: reader.read_doubles(),
    ln1_weight: reader.read_double_arrays(),
    ln1_bias: reader.read_double_arrays(),
    wq: reader.read_double_arrays(),
    bq: reader.read_double_arrays(),
    wk: reader.read_double_arrays(),
    bk: reader.read_double_arrays(),
    wv: reader.read_double_arrays(),
    bv: reader.read_double_arrays(),
    wo: reader.read_double_arrays(),
    bo: reader.read_double_arrays(),
    ln2_weight: reader.read_double_arrays(),
    ln2_bias: reader.read_double_arrays(),
    w_fc: reader.read_double_arrays(),
    b_fc: reader.read_double_arrays(),
    w_proj: reader.read_double_arrays(),
    b_proj: reader.read_double_arrays(),
    ln_f_weight: reader.read_doubles(),
    ln_f_bias: reader.read_doubles(),
  }
  let iter_num = reader.read_int_le()
  let best_val_loss = reader.read_double_le()
  let has_training = reader.read_int_le()
  let training_config : TrainingConfig? = if has_training == 0 {
    None
  } else if has_training == 1 {
    let config = reader.read_training_config()
    reader.skip_adamw_checkpoint()
    Some(config)
  } else {
    abort("invalid checkpoint training metadata flag")
  }
  if !reader.is_done() {
    abort("checkpoint contains trailing bytes")
  }
  let checkpoint : MiniGPTCheckpoint = {
    version,
    model_kind: Some(model_kind),
    tokenizer_kind: Some(tokenizer_kind),
    vocabulary,
    n_embd,
    n_head,
    n_layer,
    block_size,
    parameters,
    training_config,
    optimizer: None,
    iter_num,
    best_val_loss,
  }
  checkpoint.to_model_and_tokenizer()
}

///|
fn write_training_config(buffer : Buffer, config : TrainingConfig) -> Unit {
  buffer.write_int_le(config.batch_size)
  buffer.write_int_le(config.block_size)
  buffer.write_int_le(config.steps)
  buffer.write_double_le(config.learning_rate)
  buffer.write_double_le(config.min_lr)
  buffer.write_int_le(config.warmup_iters)
  buffer.write_int_le(config.eval_interval)
  buffer.write_int_le(config.eval_iters)
  buffer.write_int_le(config.log_interval)
  buffer.write_double_le(config.weight_decay)
  buffer.write_double_le(config.beta1)
  buffer.write_double_le(config.beta2)
  buffer.write_double_le(config.grad_clip)
  buffer.write_int_le(if config.always_save_checkpoint { 1 } else { 0 })
}

///|
fn write_adamw_checkpoint(
  buffer : Buffer,
  optimizer : @optim.AdamWCheckpoint,
) -> Unit {
  write_double_arrays(buffer, optimizer.m)
  write_double_arrays(buffer, optimizer.v)
  buffer.write_double_le(optimizer.lr)
  buffer.write_double_le(optimizer.beta1)
  buffer.write_double_le(optimizer.beta2)
  buffer.write_double_le(optimizer.eps)
  write_doubles(buffer, optimizer.weight_decays)
  buffer.write_int_le(optimizer.step)
  buffer.write_double_le(optimizer.beta1_power)
  buffer.write_double_le(optimizer.beta2_power)
}

///|
fn write_string(buffer : Buffer, text : String) -> Unit {
  let bytes = @utf8.encode(text[:])
  write_bytes(buffer, bytes)
}

///|
fn write_bytes(buffer : Buffer, bytes : Bytes) -> Unit {
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
fn write_double_arrays(buffer : Buffer, values : Array[Array[Double]]) -> Unit {
  buffer.write_int_le(values.length())
  for value in values {
    write_doubles(buffer, value)
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
  let bytes = self.read_bytes()
  @utf8.decode(bytes[:]) catch {
    _ => abort("invalid checkpoint UTF-8 string")
  }
}

///|
fn BytesReader::read_bytes(self : BytesReader) -> Bytes {
  let length = self.read_int_le()
  if length < 0 || self.offset + length > self.bytes.length() {
    abort("invalid checkpoint bytes length")
  }
  let view = self.bytes.view(start=self.offset, end=self.offset + length)
  self.offset += length
  view.to_owned()
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

///|
fn BytesReader::read_double_arrays(self : BytesReader) -> Array[Array[Double]] {
  let length = self.read_int_le()
  if length < 0 {
    abort("invalid checkpoint nested array length")
  }
  let values : Array[Array[Double]] = []
  for _ in 0..<length {
    values.push(self.read_doubles())
  }
  values
}

///|
fn BytesReader::read_training_config(self : BytesReader) -> TrainingConfig {
  TrainingConfig::new(
    self.read_int_le(),
    self.read_int_le(),
    self.read_int_le(),
    self.read_double_le(),
    min_lr=self.read_double_le(),
    warmup_iters=self.read_int_le(),
    eval_interval=self.read_int_le(),
    eval_iters=self.read_int_le(),
    log_interval=self.read_int_le(),
    weight_decay=self.read_double_le(),
    beta1=self.read_double_le(),
    beta2=self.read_double_le(),
    grad_clip=self.read_double_le(),
    always_save_checkpoint=self.read_bool(),
  )
}

///|
fn BytesReader::read_bool(self : BytesReader) -> Bool {
  match self.read_int_le() {
    0 => false
    1 => true
    _ => abort("invalid checkpoint boolean")
  }
}

///|
fn BytesReader::skip_adamw_checkpoint(self : BytesReader) -> Unit {
  ignore(self.read_double_arrays())
  ignore(self.read_double_arrays())
  ignore(self.read_double_le())
  ignore(self.read_double_le())
  ignore(self.read_double_le())
  ignore(self.read_double_le())
  ignore(self.read_doubles())
  ignore(self.read_int_le())
  ignore(self.read_double_le())
  ignore(self.read_double_le())
}
