///|
pub struct MiniGPT {
  priv ctx : @tensor.AutogradContext
  priv token_embedding_table : @tensor.Tensor
  priv vocab_size : Int
}

///|
pub fn MiniGPT::new(vocab_size : Int, rng : @random.Rand) -> MiniGPT {
  if vocab_size <= 0 {
    abort("vocab_size must be positive")
  }
  let ctx = @tensor.AutogradContext::new()
  let table = @tensor.Tensor::randn(ctx, [vocab_size, vocab_size], rng)
  let data = table.data()
  for i in 0..<data.length() {
    data[i] *= 0.02
  }
  table.set_data_in_place(data)
  { ctx, token_embedding_table: table, vocab_size }
}

///|
pub fn MiniGPT::vocab_size(self : MiniGPT) -> Int {
  self.vocab_size
}

///|
pub fn MiniGPT::forward(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  @nn.embedding(self.token_embedding_table, input_ids)
}

///|
pub fn MiniGPT::loss(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
  target_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  @nn.cross_entropy(self.forward(input_ids), target_ids)
}

///|
pub fn MiniGPT::parameters(self : MiniGPT) -> Array[@tensor.Tensor] {
  [self.token_embedding_table]
}

///|
pub fn MiniGPT::clear_graph(self : MiniGPT) -> Unit {
  self.ctx.clear_graph()
}
