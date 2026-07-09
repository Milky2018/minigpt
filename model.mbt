///|
pub const N_EMBD : Int = 384

///|
pub const N_HEAD : Int = 6

///|
pub const N_LAYER : Int = 6

///|
pub const MLP_MULTIPLIER : Int = 4

///|
pub const DROPOUT : Double = 0.2

///|
const LAYER_NORM_EPS : Double = 1.0e-5

///|
const KIND_TRANSFORMER : String = "gpt-transformer"

///|
priv struct MiniGPTParameterData {
  token_embedding_table : Array[Double]
  position_embedding_table : Array[Double]
  ln1_weight : Array[Array[Double]]
  ln1_bias : Array[Array[Double]]
  wq : Array[Array[Double]]
  bq : Array[Array[Double]]
  wk : Array[Array[Double]]
  bk : Array[Array[Double]]
  wv : Array[Array[Double]]
  bv : Array[Array[Double]]
  wo : Array[Array[Double]]
  bo : Array[Array[Double]]
  ln2_weight : Array[Array[Double]]
  ln2_bias : Array[Array[Double]]
  w_fc : Array[Array[Double]]
  b_fc : Array[Array[Double]]
  w_proj : Array[Array[Double]]
  b_proj : Array[Array[Double]]
  ln_f_weight : Array[Double]
  ln_f_bias : Array[Double]
}

///|
pub struct MiniGPT {
  priv ctx : @tensor.AutogradContext
  priv token_embedding_table : @tensor.Tensor
  priv position_embedding_table : @tensor.Tensor
  priv ln1_weight : Array[@tensor.Tensor]
  priv ln1_bias : Array[@tensor.Tensor]
  priv wq : Array[@tensor.Tensor]
  priv bq : Array[@tensor.Tensor]
  priv wk : Array[@tensor.Tensor]
  priv bk : Array[@tensor.Tensor]
  priv wv : Array[@tensor.Tensor]
  priv bv : Array[@tensor.Tensor]
  priv wo : Array[@tensor.Tensor]
  priv bo : Array[@tensor.Tensor]
  priv ln2_weight : Array[@tensor.Tensor]
  priv ln2_bias : Array[@tensor.Tensor]
  priv w_fc : Array[@tensor.Tensor]
  priv b_fc : Array[@tensor.Tensor]
  priv w_proj : Array[@tensor.Tensor]
  priv b_proj : Array[@tensor.Tensor]
  priv ln_f_weight : @tensor.Tensor
  priv ln_f_bias : @tensor.Tensor
  priv vocab_size : Int
  priv n_embd : Int
  priv n_head : Int
  priv n_layer : Int
  priv block_size : Int
}

///|
pub struct ModelConfig {
  priv vocab_size : Int
  priv n_embd : Int
  priv n_head : Int
  priv n_layer : Int
  priv block_size : Int
}

///|
pub struct ArchitectureConfig {
  priv n_embd : Int
  priv n_head : Int
  priv n_layer : Int
}

///|
pub fn ArchitectureConfig::ArchitectureConfig(
  n_embd? : Int = N_EMBD,
  n_head? : Int = N_HEAD,
  n_layer? : Int = N_LAYER,
) -> ArchitectureConfig {
  validate_architecture_options(n_embd, n_head, n_layer)
  { n_embd, n_head, n_layer }
}

///|
pub fn ArchitectureConfig::n_embd(self : ArchitectureConfig) -> Int {
  self.n_embd
}

///|
pub fn ArchitectureConfig::n_head(self : ArchitectureConfig) -> Int {
  self.n_head
}

///|
pub fn ArchitectureConfig::n_layer(self : ArchitectureConfig) -> Int {
  self.n_layer
}

///|
pub fn ModelConfig::ModelConfig(
  vocab_size : Int,
  n_embd? : Int = N_EMBD,
  n_head? : Int = N_HEAD,
  n_layer? : Int = N_LAYER,
  block_size? : Int = BLOCK_SIZE,
) -> ModelConfig {
  validate_transformer_options(vocab_size, n_embd, n_head, n_layer, block_size)
  { vocab_size, n_embd, n_head, n_layer, block_size }
}

///|
pub fn ModelConfig::from_architecture(
  vocab_size : Int,
  architecture : ArchitectureConfig,
  block_size? : Int = BLOCK_SIZE,
) -> ModelConfig {
  ModelConfig(
    vocab_size,
    n_embd=architecture.n_embd,
    n_head=architecture.n_head,
    n_layer=architecture.n_layer,
    block_size~,
  )
}

///|
pub fn ModelConfig::vocab_size(self : ModelConfig) -> Int {
  self.vocab_size
}

///|
pub fn ModelConfig::n_embd(self : ModelConfig) -> Int {
  self.n_embd
}

///|
pub fn ModelConfig::n_head(self : ModelConfig) -> Int {
  self.n_head
}

///|
pub fn ModelConfig::n_layer(self : ModelConfig) -> Int {
  self.n_layer
}

///|
pub fn ModelConfig::block_size(self : ModelConfig) -> Int {
  self.block_size
}

///|
pub fn MiniGPT::MiniGPT(config : ModelConfig, rng : @random.Rand) -> MiniGPT {
  MiniGPT::init_gpt(
    config.vocab_size,
    config.n_embd,
    config.n_head,
    config.n_layer,
    config.block_size,
    rng,
  )
}

///|
fn MiniGPT::init_gpt(
  vocab_size : Int,
  n_embd : Int,
  n_head : Int,
  n_layer : Int,
  block_size : Int,
  rng : @random.Rand,
) -> MiniGPT {
  validate_transformer_options(vocab_size, n_embd, n_head, n_layer, block_size)
  let ctx = @tensor.AutogradContext()
  let mlp_hidden = n_embd * MLP_MULTIPLIER
  let residual_scale = 0.02 / (2.0 * n_layer.to_double()).sqrt()
  {
    ctx,
    token_embedding_table: scaled_randn(ctx, [vocab_size, n_embd], rng),
    position_embedding_table: scaled_randn(ctx, [block_size, n_embd], rng),
    ln1_weight: repeated_parameter(ctx, n_layer, [n_embd], 1.0),
    ln1_bias: repeated_tensor(n_layer, [n_embd], 0.0),
    wq: repeated_scaled_randn(ctx, n_layer, [n_embd, n_embd], rng),
    bq: repeated_tensor(n_layer, [n_embd], 0.0),
    wk: repeated_scaled_randn(ctx, n_layer, [n_embd, n_embd], rng),
    bk: repeated_tensor(n_layer, [n_embd], 0.0),
    wv: repeated_scaled_randn(ctx, n_layer, [n_embd, n_embd], rng),
    bv: repeated_tensor(n_layer, [n_embd], 0.0),
    wo: repeated_scaled_randn_with_scale(
      ctx,
      n_layer,
      [n_embd, n_embd],
      rng,
      residual_scale,
    ),
    bo: repeated_tensor(n_layer, [n_embd], 0.0),
    ln2_weight: repeated_parameter(ctx, n_layer, [n_embd], 1.0),
    ln2_bias: repeated_tensor(n_layer, [n_embd], 0.0),
    w_fc: repeated_scaled_randn(ctx, n_layer, [n_embd, mlp_hidden], rng),
    b_fc: repeated_tensor(n_layer, [mlp_hidden], 0.0),
    w_proj: repeated_scaled_randn_with_scale(
      ctx,
      n_layer,
      [mlp_hidden, n_embd],
      rng,
      residual_scale,
    ),
    b_proj: repeated_tensor(n_layer, [n_embd], 0.0),
    ln_f_weight: parameter(ctx, [n_embd], 1.0),
    ln_f_bias: @tensor.Tensor::zeros([n_embd]),
    vocab_size,
    n_embd,
    n_head,
    n_layer,
    block_size,
  }
}

///|
fn MiniGPT::from_parameter_data(
  vocab_size : Int,
  n_embd : Int,
  n_head : Int,
  n_layer : Int,
  block_size : Int,
  data : MiniGPTParameterData,
) -> MiniGPT {
  validate_transformer_options(vocab_size, n_embd, n_head, n_layer, block_size)
  let ctx = @tensor.AutogradContext()
  let mlp_hidden = n_embd * MLP_MULTIPLIER
  {
    ctx,
    token_embedding_table: @tensor.Tensor::parameter(
      ctx,
      data.token_embedding_table,
      [vocab_size, n_embd],
    ),
    position_embedding_table: @tensor.Tensor::parameter(
      ctx,
      data.position_embedding_table,
      [block_size, n_embd],
    ),
    ln1_weight: parameter_array(ctx, data.ln1_weight, n_layer, [n_embd]),
    ln1_bias: tensor_array(data.ln1_bias, n_layer, [n_embd]),
    wq: parameter_array(ctx, data.wq, n_layer, [n_embd, n_embd]),
    bq: tensor_array(data.bq, n_layer, [n_embd]),
    wk: parameter_array(ctx, data.wk, n_layer, [n_embd, n_embd]),
    bk: tensor_array(data.bk, n_layer, [n_embd]),
    wv: parameter_array(ctx, data.wv, n_layer, [n_embd, n_embd]),
    bv: tensor_array(data.bv, n_layer, [n_embd]),
    wo: parameter_array(ctx, data.wo, n_layer, [n_embd, n_embd]),
    bo: tensor_array(data.bo, n_layer, [n_embd]),
    ln2_weight: parameter_array(ctx, data.ln2_weight, n_layer, [n_embd]),
    ln2_bias: tensor_array(data.ln2_bias, n_layer, [n_embd]),
    w_fc: parameter_array(ctx, data.w_fc, n_layer, [n_embd, mlp_hidden]),
    b_fc: tensor_array(data.b_fc, n_layer, [mlp_hidden]),
    w_proj: parameter_array(ctx, data.w_proj, n_layer, [mlp_hidden, n_embd]),
    b_proj: tensor_array(data.b_proj, n_layer, [n_embd]),
    ln_f_weight: @tensor.Tensor::parameter(ctx, data.ln_f_weight, [n_embd]),
    ln_f_bias: @tensor.Tensor::from_array(data.ln_f_bias, [n_embd]),
    vocab_size,
    n_embd,
    n_head,
    n_layer,
    block_size,
  }
}

///|
fn validate_transformer_options(
  vocab_size : Int,
  n_embd : Int,
  n_head : Int,
  n_layer : Int,
  block_size : Int,
) -> Unit {
  if vocab_size <= 0 {
    abort("vocab_size must be positive")
  }
  validate_architecture_options(n_embd, n_head, n_layer)
  if block_size <= 0 {
    abort("block_size must be positive")
  }
}

///|
fn validate_architecture_options(
  n_embd : Int,
  n_head : Int,
  n_layer : Int,
) -> Unit {
  if n_embd <= 0 {
    abort("n_embd must be positive")
  }
  if n_head <= 0 {
    abort("n_head must be positive")
  }
  if n_embd % n_head != 0 {
    abort("n_embd must be divisible by n_head")
  }
  if n_layer <= 0 {
    abort("n_layer must be positive")
  }
}

///|
fn scaled_randn(
  ctx : @tensor.AutogradContext,
  shape : Array[Int],
  rng : @random.Rand,
) -> @tensor.Tensor {
  scaled_randn_with_scale(ctx, shape, rng, 0.02)
}

///|
fn scaled_randn_with_scale(
  ctx : @tensor.AutogradContext,
  shape : Array[Int],
  rng : @random.Rand,
  scale : Double,
) -> @tensor.Tensor {
  let tensor = @tensor.Tensor::randn(ctx, shape, rng)
  let data = tensor.data()
  for i in 0..<data.length() {
    data[i] *= scale
  }
  tensor.assign_parameter_data(data)
  tensor
}

///|
fn parameter(
  ctx : @tensor.AutogradContext,
  shape : Array[Int],
  value : Double,
) -> @tensor.Tensor {
  @tensor.Tensor::parameter(
    ctx,
    Array::make(shape_size_local(shape), value),
    shape,
  )
}

///|
fn repeated_parameter(
  ctx : @tensor.AutogradContext,
  count : Int,
  shape : Array[Int],
  value : Double,
) -> Array[@tensor.Tensor] {
  let tensors : Array[@tensor.Tensor] = []
  for _ in 0..<count {
    tensors.push(parameter(ctx, shape, value))
  }
  tensors
}

///|
fn repeated_tensor(
  count : Int,
  shape : Array[Int],
  value : Double,
) -> Array[@tensor.Tensor] {
  let tensors : Array[@tensor.Tensor] = []
  for _ in 0..<count {
    tensors.push(
      @tensor.Tensor::from_array(
        Array::make(shape_size_local(shape), value),
        shape,
      ),
    )
  }
  tensors
}

///|
fn repeated_scaled_randn(
  ctx : @tensor.AutogradContext,
  count : Int,
  shape : Array[Int],
  rng : @random.Rand,
) -> Array[@tensor.Tensor] {
  repeated_scaled_randn_with_scale(ctx, count, shape, rng, 0.02)
}

///|
fn repeated_scaled_randn_with_scale(
  ctx : @tensor.AutogradContext,
  count : Int,
  shape : Array[Int],
  rng : @random.Rand,
  scale : Double,
) -> Array[@tensor.Tensor] {
  let tensors : Array[@tensor.Tensor] = []
  for _ in 0..<count {
    tensors.push(scaled_randn_with_scale(ctx, shape, rng, scale))
  }
  tensors
}

///|
fn parameter_array(
  ctx : @tensor.AutogradContext,
  values : Array[Array[Double]],
  count : Int,
  shape : Array[Int],
) -> Array[@tensor.Tensor] {
  if values.length() != count {
    abort("checkpoint parameter array count does not match n_layer")
  }
  let tensors : Array[@tensor.Tensor] = []
  for value in values {
    tensors.push(@tensor.Tensor::parameter(ctx, value, shape))
  }
  tensors
}

///|
fn tensor_array(
  values : Array[Array[Double]],
  count : Int,
  shape : Array[Int],
) -> Array[@tensor.Tensor] {
  if values.length() != count {
    abort("checkpoint tensor array count does not match n_layer")
  }
  let tensors : Array[@tensor.Tensor] = []
  for value in values {
    tensors.push(@tensor.Tensor::from_array(value, shape))
  }
  tensors
}

///|
fn shape_size_local(shape : Array[Int]) -> Int {
  let mut size = 1
  for dim in shape {
    if dim < 0 {
      abort("shape dimensions must not be negative")
    }
    size *= dim
  }
  size
}

///|
fn tensors_data(tensors : Array[@tensor.Tensor]) -> Array[Array[Double]] {
  let data : Array[Array[Double]] = []
  for tensor in tensors {
    data.push(tensor.data())
  }
  data
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
pub fn MiniGPT::n_head(self : MiniGPT) -> Int {
  self.n_head
}

///|
pub fn MiniGPT::n_layer(self : MiniGPT) -> Int {
  self.n_layer
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
fn MiniGPT::parameter_data(self : MiniGPT) -> MiniGPTParameterData {
  {
    token_embedding_table: self.token_embedding_table.data(),
    position_embedding_table: self.position_embedding_table.data(),
    ln1_weight: tensors_data(self.ln1_weight),
    ln1_bias: tensors_data(self.ln1_bias),
    wq: tensors_data(self.wq),
    bq: tensors_data(self.bq),
    wk: tensors_data(self.wk),
    bk: tensors_data(self.bk),
    wv: tensors_data(self.wv),
    bv: tensors_data(self.bv),
    wo: tensors_data(self.wo),
    bo: tensors_data(self.bo),
    ln2_weight: tensors_data(self.ln2_weight),
    ln2_bias: tensors_data(self.ln2_bias),
    w_fc: tensors_data(self.w_fc),
    b_fc: tensors_data(self.b_fc),
    w_proj: tensors_data(self.w_proj),
    b_proj: tensors_data(self.b_proj),
    ln_f_weight: self.ln_f_weight.data(),
    ln_f_bias: self.ln_f_bias.data(),
  }
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
  TokenIds(ids, shape)
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
        data[row * time + col] = @double.neg_infinity
      }
    }
  }
  @tensor.Tensor::from_array(data, [time, time])
}

///|
fn linear(
  input : @tensor.Tensor,
  weight : @tensor.Tensor,
  bias : @tensor.Tensor,
) -> @tensor.Tensor {
  input.matmul(weight) + bias
}

///|
fn split_heads(
  input : @tensor.Tensor,
  n_head : Int,
  head_dim : Int,
) -> @tensor.Tensor {
  let shape = input.shape()
  let time = shape[shape.length() - 2]
  let prefix = shape[0:shape.length() - 2].to_owned()
  prefix.push(time)
  prefix.push(n_head)
  prefix.push(head_dim)
  input.reshape(prefix).swap_axes(-3, -2)
}

///|
fn merge_heads(
  input : @tensor.Tensor,
  output_shape : Array[Int],
) -> @tensor.Tensor {
  input.swap_axes(-3, -2).reshape(output_shape)
}

///|
fn MiniGPT::self_attention(
  self : MiniGPT,
  input : @tensor.Tensor,
  layer : Int,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  let head_dim = self.n_embd / self.n_head
  let shape = input.shape()
  let time = shape[shape.length() - 2]
  let q = split_heads(
    linear(input, self.wq[layer], self.bq[layer]),
    self.n_head,
    head_dim,
  )
  let k = split_heads(
    linear(input, self.wk[layer], self.bk[layer]),
    self.n_head,
    head_dim,
  )
  let v = split_heads(
    linear(input, self.wv[layer], self.bv[layer]),
    self.n_head,
    head_dim,
  )
  let scale = @tensor.Tensor::scalar(1.0 / head_dim.to_double().sqrt())
  let scores = q.matmul(k.swap_axes(-2, -1)) * scale + causal_mask(time)
  let weights = dropout(scores.softmax(-1), DROPOUT, rng)
  let attended = merge_heads(weights.matmul(v), shape)
  dropout(linear(attended, self.wo[layer], self.bo[layer]), DROPOUT, rng)
}

///|
fn MiniGPT::mlp(
  self : MiniGPT,
  input : @tensor.Tensor,
  layer : Int,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  let hidden = linear(input, self.w_fc[layer], self.b_fc[layer]).gelu()
  dropout(linear(hidden, self.w_proj[layer], self.b_proj[layer]), DROPOUT, rng)
}

///|
fn MiniGPT::block(
  self : MiniGPT,
  input : @tensor.Tensor,
  layer : Int,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  let attn_input = input.layer_norm(
    self.ln1_weight[layer],
    self.ln1_bias[layer],
    LAYER_NORM_EPS,
  )
  let x = input + self.self_attention(attn_input, layer, rng)
  let mlp_input = x.layer_norm(
    self.ln2_weight[layer],
    self.ln2_bias[layer],
    LAYER_NORM_EPS,
  )
  x + self.mlp(mlp_input, layer, rng)
}

///|
fn dropout(
  input : @tensor.Tensor,
  probability : Double,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  if probability < 0.0 || probability >= 1.0 {
    abort("dropout probability must be in [0, 1)")
  }
  match rng {
    Some(random) if probability > 0.0 => {
      let shape = input.shape()
      let mask = Array::make(input.numel(), 0.0)
      let scale = 1.0 / (1.0 - probability)
      for i in 0..<mask.length() {
        if random.double() >= probability {
          mask[i] = scale
        }
      }
      input * @tensor.Tensor::from_array(mask, shape)
    }
    _ => input
  }
}

///|
fn MiniGPT::hidden_states(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  let token_embeddings = @nn.embedding(self.token_embedding_table, input_ids)
  let position_embeddings = @nn.embedding(
    self.position_embedding_table,
    position_ids(input_ids, self.block_size),
  )
  let mut hidden = dropout(token_embeddings + position_embeddings, DROPOUT, rng)
  for layer in 0..<self.n_layer {
    hidden = self.block(hidden, layer, rng)
  }
  hidden.layer_norm(self.ln_f_weight, self.ln_f_bias, LAYER_NORM_EPS)
}

///|
fn MiniGPT::forward_internal(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  self
  .hidden_states(input_ids, rng)
  .matmul(self.token_embedding_table.transpose2d())
}

///|
pub fn MiniGPT::forward(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  self.forward_internal(input_ids, None)
}

///|
fn MiniGPT::forward_train(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
  rng : @random.Rand,
) -> @tensor.Tensor {
  self.forward_internal(input_ids, Some(rng))
}

///|
pub fn MiniGPT::last_logits(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  let hidden = self.hidden_states(input_ids, None)
  let hidden_data = hidden.data()
  let last_offset = hidden_data.length() - self.n_embd
  let embedding = self.token_embedding_table.data()
  let logits = Array::make(self.vocab_size, 0.0)
  for token in 0..<self.vocab_size {
    let mut value = 0.0
    for dim in 0..<self.n_embd {
      value += hidden_data[last_offset + dim] *
        embedding[token * self.n_embd + dim]
    }
    logits[token] = value
  }
  self.clear_graph()
  @tensor.Tensor::from_array(logits, [self.vocab_size])
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
fn MiniGPT::loss_train(
  self : MiniGPT,
  input_ids : @tensor.TokenIds,
  target_ids : @tensor.TokenIds,
  rng : @random.Rand,
) -> @tensor.Tensor {
  @nn.cross_entropy(self.forward_train(input_ids, rng), target_ids)
}

///|
fn MiniGPT::parameters(self : MiniGPT) -> Array[@tensor.Tensor] {
  let params : Array[@tensor.Tensor] = [
    self.token_embedding_table,
    self.position_embedding_table,
  ]
  for layer in 0..<self.n_layer {
    params.push(self.ln1_weight[layer])
    params.push(self.wq[layer])
    params.push(self.wk[layer])
    params.push(self.wv[layer])
    params.push(self.wo[layer])
    params.push(self.ln2_weight[layer])
    params.push(self.w_fc[layer])
    params.push(self.w_proj[layer])
  }
  params.push(self.ln_f_weight)
  params
}

///|
fn MiniGPT::parameter_weight_decays(
  self : MiniGPT,
  weight_decay : Double,
) -> Array[Double] {
  let decays : Array[Double] = [weight_decay, weight_decay]
  for _ in 0..<self.n_layer {
    decays.push(0.0)
    decays.push(weight_decay)
    decays.push(weight_decay)
    decays.push(weight_decay)
    decays.push(weight_decay)
    decays.push(0.0)
    decays.push(weight_decay)
    decays.push(weight_decay)
  }
  decays.push(0.0)
  decays
}

///|
fn MiniGPT::clear_graph(self : MiniGPT) -> Unit {
  self.ctx.clear_graph()
}

///|
fn MiniGPT::zero_grad(self : MiniGPT) -> Unit {
  self.ctx.zero_grad()
}
