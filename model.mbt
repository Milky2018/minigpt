///|
pub const N_EMBD : Int = 32

///|
pub struct MiniGPT {
  ctx : @tensor.AutogradContext
  token_embedding_table : @tensor.Tensor
  lm_head : @tensor.Tensor
  vocab_size : Int
  n_embd : Int
}

///|
pub fn MiniGPT::new(vocab_size : Int, rng : @random.Rand) -> MiniGPT {
  MiniGPT::new_with_options(vocab_size, N_EMBD, rng)
}

///|
pub fn MiniGPT::new_with_options(
  vocab_size : Int,
  n_embd : Int,
  rng : @random.Rand,
) -> MiniGPT {
  if vocab_size <= 0 {
    abort("vocab_size must be positive")
  }
  if n_embd <= 0 {
    abort("n_embd must be positive")
  }
  let ctx = @tensor.AutogradContext::new()
  let token_embedding_table = scaled_randn(ctx, [vocab_size, n_embd], rng)
  let lm_head = scaled_randn(ctx, [n_embd, vocab_size], rng)
  { ctx, token_embedding_table, lm_head, vocab_size, n_embd }
}

///|
pub fn MiniGPT::from_parameters(
  vocab_size : Int,
  n_embd : Int,
  token_embedding_table : Array[Double],
  lm_head : Array[Double],
) -> MiniGPT {
  if vocab_size <= 0 {
    abort("vocab_size must be positive")
  }
  if n_embd <= 0 {
    abort("n_embd must be positive")
  }
  let ctx = @tensor.AutogradContext::new()
  let token_embedding_table = @tensor.Tensor::parameter(
    ctx,
    token_embedding_table,
    [vocab_size, n_embd],
  )
  let lm_head = @tensor.Tensor::parameter(ctx, lm_head, [n_embd, vocab_size])
  { ctx, token_embedding_table, lm_head, vocab_size, n_embd }
}

///|
fn scaled_randn(
  ctx : @tensor.AutogradContext,
  shape : Array[Int],
  rng : @random.Rand,
) -> @tensor.Tensor {
  let tensor = @tensor.Tensor::randn(ctx, shape, rng)
  let data = tensor.data()
  for i in 0..<data.length() {
    data[i] *= 0.02
  }
  tensor.set_data_in_place(data)
  tensor
}

///|
pub fn MiniGPT::vocab_size(self : MiniGPT) -> Int {
  self.vocab_size
}

///|
pub fn MiniGPT::n_embd(self : MiniGPT) -> Int {
  self.n_embd
}

///|
pub fn MiniGPT::token_embedding_data(self : MiniGPT) -> Array[Double] {
  self.token_embedding_table.data()
}

///|
pub fn MiniGPT::lm_head_data(self : MiniGPT) -> Array[Double] {
  self.lm_head.data()
}

///|
pub fn MiniGPT::forward(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  @nn.embedding(self.token_embedding_table, input_ids).matmul(self.lm_head)
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
  [self.token_embedding_table, self.lm_head]
}

///|
pub fn MiniGPT::clear_graph(self : MiniGPT) -> Unit {
  self.ctx.clear_graph()
}
