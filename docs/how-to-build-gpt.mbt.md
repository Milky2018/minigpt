# How to Build a nanoGPT-Style GPT

这个教程从零写一个很小的 GPT，而不是调用根包里已经实现好的 `MiniGPT`、`TrainingConfig`、`train_text` 或 `generate_text`。我们只使用本库公开的基础设施子包：`tokenizer/` 负责文本到 token id，`tensor/` 负责张量和自动微分，`nn/` 负责 embedding 和 cross entropy，`optim/` 负责 AdamW。

目标是做一个完整的端到端例子：文本语料 -> tokenizer -> batch -> decoder-only transformer -> loss -> backward -> AdamW step -> prompt completion。代码块使用 `mbt check`，所以它会随 MoonBit 类型检查一起保持可编译。

这个版本保留 nanoGPT 的关键结构：token embedding、position embedding、pre-norm block、causal self-attention、MLP/GELU、residual connection、final layer norm、tied output head、next-token cross entropy 和 AdamW。为了让教程短一些，它不包含 checkpoint、eval loop、dropout、top-k sampling 和学习率调度；这些可以在理解本文件后再加。

```mbt check
///|
const BLOCK_SIZE : Int = 8

///|
const N_EMBD : Int = 16

///|
const N_HEAD : Int = 4

///|
const N_LAYER : Int = 2

///|
const MLP_MULTIPLIER : Int = 4

///|
const LAYER_NORM_EPS : Double = 1.0e-5

///|
fn shape_size(shape : Array[Int]) -> Int {
  let mut size = 1
  for dim in shape {
    if dim <= 0 {
      abort("shape dimensions must be positive")
    }
    size *= dim
  }
  size
}

///|
fn parameter(
  ctx : @tensor.AutogradContext,
  shape : Array[Int],
  value : Double,
) -> @tensor.Tensor {
  @tensor.Tensor::parameter(ctx, Array::make(shape_size(shape), value), shape)
}

///|
fn scaled_randn(
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
  tensor.set_data_in_place(data)
  tensor
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
fn repeated_scaled_randn(
  ctx : @tensor.AutogradContext,
  count : Int,
  shape : Array[Int],
  rng : @random.Rand,
  scale : Double,
) -> Array[@tensor.Tensor] {
  let tensors : Array[@tensor.Tensor] = []
  for _ in 0..<count {
    tensors.push(scaled_randn(ctx, shape, rng, scale))
  }
  tensors
}

///|
fn zero_tensor(shape : Array[Int]) -> @tensor.Tensor {
  @tensor.Tensor::zeros(shape)
}

///|
fn repeated_zero_tensor(
  count : Int,
  shape : Array[Int],
) -> Array[@tensor.Tensor] {
  let tensors : Array[@tensor.Tensor] = []
  for _ in 0..<count {
    tensors.push(zero_tensor(shape))
  }
  tensors
}

///|
struct TutorialGPT {
  ctx : @tensor.AutogradContext
  token_embedding : @tensor.Tensor
  position_embedding : @tensor.Tensor
  ln1_weight : Array[@tensor.Tensor]
  ln1_bias : Array[@tensor.Tensor]
  wq : Array[@tensor.Tensor]
  wk : Array[@tensor.Tensor]
  wv : Array[@tensor.Tensor]
  wo : Array[@tensor.Tensor]
  ln2_weight : Array[@tensor.Tensor]
  ln2_bias : Array[@tensor.Tensor]
  w_fc : Array[@tensor.Tensor]
  w_proj : Array[@tensor.Tensor]
  ln_f_weight : @tensor.Tensor
  ln_f_bias : @tensor.Tensor
  vocab_size : Int
  n_embd : Int
  n_head : Int
  n_layer : Int
  block_size : Int
}

///|
fn TutorialGPT::new(vocab_size : Int, rng : @random.Rand) -> TutorialGPT {
  if N_EMBD % N_HEAD != 0 {
    abort("N_EMBD must be divisible by N_HEAD")
  }
  let ctx = @tensor.AutogradContext::new()
  let mlp_hidden = N_EMBD * MLP_MULTIPLIER
  let residual_scale = 0.02 / (2.0 * N_LAYER.to_double()).sqrt()
  {
    ctx,
    token_embedding: scaled_randn(ctx, [vocab_size, N_EMBD], rng, 0.02),
    position_embedding: scaled_randn(ctx, [BLOCK_SIZE, N_EMBD], rng, 0.02),
    ln1_weight: repeated_parameter(ctx, N_LAYER, [N_EMBD], 1.0),
    ln1_bias: repeated_zero_tensor(N_LAYER, [N_EMBD]),
    wq: repeated_scaled_randn(ctx, N_LAYER, [N_EMBD, N_EMBD], rng, 0.02),
    wk: repeated_scaled_randn(ctx, N_LAYER, [N_EMBD, N_EMBD], rng, 0.02),
    wv: repeated_scaled_randn(ctx, N_LAYER, [N_EMBD, N_EMBD], rng, 0.02),
    wo: repeated_scaled_randn(
      ctx,
      N_LAYER,
      [N_EMBD, N_EMBD],
      rng,
      residual_scale,
    ),
    ln2_weight: repeated_parameter(ctx, N_LAYER, [N_EMBD], 1.0),
    ln2_bias: repeated_zero_tensor(N_LAYER, [N_EMBD]),
    w_fc: repeated_scaled_randn(ctx, N_LAYER, [N_EMBD, mlp_hidden], rng, 0.02),
    w_proj: repeated_scaled_randn(
      ctx,
      N_LAYER,
      [mlp_hidden, N_EMBD],
      rng,
      residual_scale,
    ),
    ln_f_weight: parameter(ctx, [N_EMBD], 1.0),
    ln_f_bias: zero_tensor([N_EMBD]),
    vocab_size,
    n_embd: N_EMBD,
    n_head: N_HEAD,
    n_layer: N_LAYER,
    block_size: BLOCK_SIZE,
  }
}

///|
fn position_ids(
  input_ids : @tensor.TokenIds,
  block_size : Int,
) -> @tensor.TokenIds {
  let shape = input_ids.shape()
  let time = shape[shape.length() - 1]
  if time > block_size {
    abort("input sequence is longer than block_size")
  }
  let ids : Array[Int] = []
  for i in 0..<input_ids.data().length() {
    ids.push(i % time)
  }
  @tensor.TokenIds::new(ids, shape)
}

///|
fn causal_mask(time : Int) -> @tensor.Tensor {
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
fn TutorialGPT::self_attention(
  self : TutorialGPT,
  input : @tensor.Tensor,
  layer : Int,
) -> @tensor.Tensor {
  let head_dim = self.n_embd / self.n_head
  let shape = input.shape()
  let time = shape[shape.length() - 2]
  let q = split_heads(input.matmul(self.wq[layer]), self.n_head, head_dim)
  let k = split_heads(input.matmul(self.wk[layer]), self.n_head, head_dim)
  let v = split_heads(input.matmul(self.wv[layer]), self.n_head, head_dim)
  let scale = @tensor.Tensor::scalar(1.0 / head_dim.to_double().sqrt())
  let scores = q.matmul(k.swap_axes(-2, -1)) * scale + causal_mask(time)
  let weights = scores.softmax(-1)
  merge_heads(weights.matmul(v), shape).matmul(self.wo[layer])
}

///|
fn TutorialGPT::mlp(
  self : TutorialGPT,
  input : @tensor.Tensor,
  layer : Int,
) -> @tensor.Tensor {
  input.matmul(self.w_fc[layer]).gelu().matmul(self.w_proj[layer])
}

///|
fn TutorialGPT::block(
  self : TutorialGPT,
  input : @tensor.Tensor,
  layer : Int,
) -> @tensor.Tensor {
  let attn_input = input.layer_norm(
    self.ln1_weight[layer],
    self.ln1_bias[layer],
    LAYER_NORM_EPS,
  )
  let x = input + self.self_attention(attn_input, layer)
  let mlp_input = x.layer_norm(
    self.ln2_weight[layer],
    self.ln2_bias[layer],
    LAYER_NORM_EPS,
  )
  x + self.mlp(mlp_input, layer)
}

///|
fn TutorialGPT::hidden_states(
  self : TutorialGPT,
  input_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  let token_embeddings = @nn.embedding(self.token_embedding, input_ids)
  let position_embeddings = @nn.embedding(
    self.position_embedding,
    position_ids(input_ids, self.block_size),
  )
  let mut hidden = token_embeddings + position_embeddings
  for layer in 0..<self.n_layer {
    hidden = self.block(hidden, layer)
  }
  hidden.layer_norm(self.ln_f_weight, self.ln_f_bias, LAYER_NORM_EPS)
}

///|
fn TutorialGPT::forward(
  self : TutorialGPT,
  input_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  self.hidden_states(input_ids).matmul(self.token_embedding.transpose2d())
}

///|
fn TutorialGPT::loss(
  self : TutorialGPT,
  input_ids : @tensor.TokenIds,
  target_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  @nn.cross_entropy(self.forward(input_ids), target_ids)
}

///|
fn TutorialGPT::parameters(self : TutorialGPT) -> Array[@tensor.Tensor] {
  let params : Array[@tensor.Tensor] = [
    self.token_embedding,
    self.position_embedding,
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
fn TutorialGPT::clear_graph(self : TutorialGPT) -> Unit {
  self.ctx.clear_graph()
}

///|
fn sample_batch(
  token_ids : Array[Int],
  batch_size : Int,
  block_size : Int,
  rng : @random.Rand,
) -> (@tensor.TokenIds, @tensor.TokenIds) {
  let inputs : Array[Int] = []
  let targets : Array[Int] = []
  let limit = token_ids.length() - block_size
  for _ in 0..<batch_size {
    let start = rng.int(limit~)
    for offset in 0..<block_size {
      inputs.push(token_ids[start + offset])
      targets.push(token_ids[start + offset + 1])
    }
  }
  (
    @tensor.TokenIds::new(inputs, [batch_size, block_size]),
    @tensor.TokenIds::new(targets, [batch_size, block_size]),
  )
}

///|
fn train_one_step(
  model : TutorialGPT,
  optimizer : @optim.AdamW,
  train_ids : Array[Int],
  rng : @random.Rand,
) -> Double {
  let (inputs, targets) = sample_batch(train_ids, 2, model.block_size, rng)
  let loss = model.loss(inputs, targets)
  let loss_value = loss.data()[0]
  loss.backward()
  optimizer.step_with_grad_clip(1.0)
  model.clear_graph()
  loss_value
}

///|
fn argmax(values : Array[Double]) -> Int {
  let mut best = 0
  for i in 1..<values.length() {
    if values[i] > values[best] {
      best = i
    }
  }
  best
}

///|
fn TutorialGPT::next_token(self : TutorialGPT, context_ids : Array[Int]) -> Int {
  let input = @tensor.TokenIds::new(context_ids, [1, context_ids.length()])
  let logits = self.forward(input).data()
  self.clear_graph()
  let offset = (context_ids.length() - 1) * self.vocab_size
  argmax(logits[offset:offset + self.vocab_size].to_owned())
}

///|
fn generate(
  model : TutorialGPT,
  tokenizer : @tokenizer.Tokenizer,
  prompt : String,
  max_new_tokens : Int,
) -> String raise @tokenizer.TokenizerError {
  let ids = tokenizer.encode(prompt)
  for _ in 0..<max_new_tokens {
    let start = if ids.length() > model.block_size {
      ids.length() - model.block_size
    } else {
      0
    }
    let context_ids = ids[start:].to_owned()
    ids.push(model.next_token(context_ids))
  }
  tokenizer.decode(ids)
}

///|
pub fn tutorial_e2e() -> String raise @tokenizer.TokenizerError {
  let corpus = "To be, or not to be:\nTo eat, or not to eat:\n"
  let dataset = @tokenizer.prepare_token_dataset(
    corpus,
    @tokenizer.TOKENIZER_KIND_CHAR,
  )
  let rng = @random.Rand::new()
  let model = TutorialGPT::new(dataset.tokenizer.vocab_size(), rng)
  let optimizer = @optim.AdamW::new_with_options(
    model.parameters(),
    1.0e-3,
    0.9,
    0.99,
    1.0e-8,
    0.1,
  )
  for _ in 0..<3 {
    let _loss = train_one_step(model, optimizer, dataset.train_ids, rng)
  }
  generate(model, dataset.tokenizer, "To", 8)
}
```

`tutorial_e2e` 只是一个可检查的小模型流程。真实训练时，把 `corpus` 换成文件内容，把 `N_EMBD`、`N_HEAD`、`N_LAYER`、`BLOCK_SIZE` 调大，把训练步数从 `3` 提高到几千或更多，再加入 eval/checkpoint/sampling，就会逐步接近根包 CLI 已经封装好的训练程序。

如果你想继续贴近 nanoGPT，下一步通常按这个顺序做：先加 train/val eval loop 和 best checkpoint，再加 cosine learning-rate schedule，再加 dropout，最后把 greedy `argmax` 换成 temperature + top-k sampling。
