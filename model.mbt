///|
pub const N_EMBD : Int = 32

///|
const KIND_TRANSFORMER : String = "tiny-transformer"

///|
pub struct MiniGPT {
  ctx : @tensor.AutogradContext
  token_embedding_table : @tensor.Tensor
  position_embedding_table : @tensor.Tensor
  wq : @tensor.Tensor
  wk : @tensor.Tensor
  wv : @tensor.Tensor
  wo : @tensor.Tensor
  lm_head : @tensor.Tensor
  vocab_size : Int
  n_embd : Int
  block_size : Int
}

///|
pub fn MiniGPT::new(vocab_size : Int, rng : @random.Rand) -> MiniGPT {
  MiniGPT::new_transformer_with_options(vocab_size, N_EMBD, BLOCK_SIZE, rng)
}

///|
pub fn MiniGPT::new_with_options(
  vocab_size : Int,
  n_embd : Int,
  rng : @random.Rand,
) -> MiniGPT {
  MiniGPT::new_transformer_with_options(vocab_size, n_embd, BLOCK_SIZE, rng)
}

///|
pub fn MiniGPT::new_transformer_with_options(
  vocab_size : Int,
  n_embd : Int,
  block_size : Int,
  rng : @random.Rand,
) -> MiniGPT {
  if vocab_size <= 0 {
    abort("vocab_size must be positive")
  }
  if n_embd <= 0 {
    abort("n_embd must be positive")
  }
  if block_size <= 0 {
    abort("block_size must be positive")
  }
  let ctx = @tensor.AutogradContext::new()
  let token_embedding_table = scaled_randn(ctx, [vocab_size, n_embd], rng)
  let position_embedding_table = scaled_randn(ctx, [block_size, n_embd], rng)
  let wq = scaled_randn(ctx, [n_embd, n_embd], rng)
  let wk = scaled_randn(ctx, [n_embd, n_embd], rng)
  let wv = scaled_randn(ctx, [n_embd, n_embd], rng)
  let wo = scaled_randn(ctx, [n_embd, n_embd], rng)
  let lm_head = scaled_randn(ctx, [n_embd, vocab_size], rng)
  {
    ctx,
    token_embedding_table,
    position_embedding_table,
    wq,
    wk,
    wv,
    wo,
    lm_head,
    vocab_size,
    n_embd,
    block_size,
  }
}

///|
pub fn MiniGPT::from_transformer_parameters(
  vocab_size : Int,
  n_embd : Int,
  block_size : Int,
  token_embedding_table : Array[Double],
  position_embedding_table : Array[Double],
  wq : Array[Double],
  wk : Array[Double],
  wv : Array[Double],
  wo : Array[Double],
  lm_head : Array[Double],
) -> MiniGPT {
  if vocab_size <= 0 {
    abort("vocab_size must be positive")
  }
  if n_embd <= 0 {
    abort("n_embd must be positive")
  }
  if block_size <= 0 {
    abort("block_size must be positive")
  }
  let ctx = @tensor.AutogradContext::new()
  let token_embedding_table = @tensor.Tensor::parameter(
    ctx,
    token_embedding_table,
    [vocab_size, n_embd],
  )
  let position_embedding_table = @tensor.Tensor::parameter(
    ctx,
    position_embedding_table,
    [block_size, n_embd],
  )
  let wq = @tensor.Tensor::parameter(ctx, wq, [n_embd, n_embd])
  let wk = @tensor.Tensor::parameter(ctx, wk, [n_embd, n_embd])
  let wv = @tensor.Tensor::parameter(ctx, wv, [n_embd, n_embd])
  let wo = @tensor.Tensor::parameter(ctx, wo, [n_embd, n_embd])
  let lm_head = @tensor.Tensor::parameter(ctx, lm_head, [n_embd, vocab_size])
  {
    ctx,
    token_embedding_table,
    position_embedding_table,
    wq,
    wk,
    wv,
    wo,
    lm_head,
    vocab_size,
    n_embd,
    block_size,
  }
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
pub fn MiniGPT::block_size(self : MiniGPT) -> Int {
  self.block_size
}

///|
pub fn MiniGPT::kind_name(_self : MiniGPT) -> String {
  KIND_TRANSFORMER
}

///|
pub fn MiniGPT::token_embedding_data(self : MiniGPT) -> Array[Double] {
  self.token_embedding_table.data()
}

///|
pub fn MiniGPT::position_embedding_data(self : MiniGPT) -> Array[Double] {
  self.position_embedding_table.data()
}

///|
pub fn MiniGPT::wq_data(self : MiniGPT) -> Array[Double] {
  self.wq.data()
}

///|
pub fn MiniGPT::wk_data(self : MiniGPT) -> Array[Double] {
  self.wk.data()
}

///|
pub fn MiniGPT::wv_data(self : MiniGPT) -> Array[Double] {
  self.wv.data()
}

///|
pub fn MiniGPT::wo_data(self : MiniGPT) -> Array[Double] {
  self.wo.data()
}

///|
pub fn MiniGPT::lm_head_data(self : MiniGPT) -> Array[Double] {
  self.lm_head.data()
}

///|
fn position_ids(
  input_ids : @tensor.TokenIds,
  block_size : Int,
) -> @tensor.TokenIds {
  let shape = input_ids.shape()
  if shape.length() == 0 {
    abort("input_ids must have at least one dimension")
  }
  let time = shape[shape.length() - 1]
  if time > block_size {
    abort("input sequence is longer than model block_size")
  }
  let count = input_ids.data().length()
  let ids : Array[Int] = []
  for i in 0..<count {
    ids.push(i % time)
  }
  @tensor.TokenIds::new(ids, shape)
}

///|
fn causal_mask(time : Int) -> @tensor.Tensor {
  if time <= 0 {
    abort("causal mask requires a positive time dimension")
  }
  let data = Array::make(time * time, 0.0)
  for row in 0..<time {
    for col in 0..<time {
      if col > row {
        data[row * time + col] = -1.0e9
      }
    }
  }
  @tensor.Tensor::from_array(data, [time, time])
}

///|
pub fn MiniGPT::forward(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  let embeddings = @nn.embedding(self.token_embedding_table, input_ids)
  let shape = input_ids.shape()
  let time = shape[shape.length() - 1]
  let pos = @nn.embedding(
    self.position_embedding_table,
    position_ids(input_ids, self.block_size),
  )
  let x = embeddings + pos
  let q = x.matmul(self.wq)
  let k = x.matmul(self.wk)
  let v = x.matmul(self.wv)
  let scale = @tensor.Tensor::scalar(1.0 / self.n_embd.to_double().sqrt())
  let scores = q.matmul(k.swap_axes(-2, -1)) * scale + causal_mask(time)
  let weights = scores.softmax(-1)
  let attended = weights.matmul(v)
  let hidden = attended.matmul(self.wo) + x
  hidden.matmul(self.lm_head)
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
  [
    self.token_embedding_table,
    self.position_embedding_table,
    self.wq,
    self.wk,
    self.wv,
    self.wo,
    self.lm_head,
  ]
}

///|
pub fn MiniGPT::clear_graph(self : MiniGPT) -> Unit {
  self.ctx.clear_graph()
}
