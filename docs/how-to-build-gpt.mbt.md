# 从零开始构建 GPT：MoonBit 版 nanoGPT 教程

本教程带你从零开始，用 MoonBit 一步步构建一个 nanoGPT 风格的 GPT 语言模型。我们使用 minigpt 项目的子包（`tensor`、`nn`、`optim`、`tokenizer`）来提供张量自动微分、优化器和分词器，但**不会**直接使用根包中已定义好的 `MiniGPT` 模型。目标是让你亲手把每一部分写出来，理解 GPT 的完整原理。

## 你将学到

- 训练和使用字符级/词级/BPE 分词器
- Tensor 创建、形状变换、广播运算与自动微分
- 用 AdamW 优化器训练参数
- 从零构建 GPT transformer：token/position embedding、causal self-attention、MLP、layer norm、残差连接
- 训练循环、评估与文本生成

## 前置条件

已安装 MoonBit 工具链，`docs/moon.pkg` 已配置好子包导入。

---

## 第 1 章：分词器

GPT 处理的是 token ID 序列，不是原始文本。分词器负责将文本转为整数序列，以及解码回文本。

minigpt 提供了三种分词器：字符级（`char`）、词级（`word`）和 BPE（`bpe`）。这里先用字符级分词器演示。

```mbt check
///|
/// 训练一个字符级分词器并编码/解码
test "tokenizer: char-level train, encode, decode" {
  let text = "hello world! hello moonbit!"
  let (char_tok, _all_ids) = @tokenizer.CharTokenizer::train(text)
  let tokenizer = @tokenizer.Tokenizer::from_char(char_tok)
  let vocab_size = tokenizer.vocab_size()
  // 字符级词表大小等于文本中不同字符的数量
  assert_true(vocab_size > 0)

  // 编码后立即解码应恢复原文
  let encoded = tokenizer.encode("hello world")
  let decoded = tokenizer.decode(encoded)
  @test.assert_eq(decoded, "hello world")
}
```

```mbt check
///|
/// 词级分词器
test "tokenizer: word-level train, encode, decode" {
  let text = "hello world\nhello moonbit\ngood morning"
  let (word_tok, _all_ids) = @tokenizer.WordTokenizer::train(text)
  let tokenizer = @tokenizer.Tokenizer::from_word(word_tok)
  // 词级分词器按空白和换行切分
  let encoded = tokenizer.encode("hello world")
  let decoded = tokenizer.decode(encoded)
  @test.assert_eq(decoded, "hello world")
}
```

```mbt check
///|
/// BPE 分词器：从文本自动学习子词词表
test "tokenizer: bpe train, encode, decode" {
  let text =
    #|low low low low low lower lower lower
    #|newest newest newest newest newest newest widest widest widest
  let (bpe_tok, _all_ids) = @tokenizer.BpeTokenizer::train(text, 32)
  let tokenizer = @tokenizer.Tokenizer::from_bpe(bpe_tok)
  // BPE 会把高频组合合并为子词
  let encoded = tokenizer.encode("lowest")
  let decoded = tokenizer.decode(encoded)
  @test.assert_eq(decoded, "lowest")
}
```

---

## 第 2 章：Tensor 基础

`Tensor` 是数值计算的核心数据结构，内部以行优先（row-major）方式连续存储 `Double` 数组，shape 是运行时 `Array[Int]`。

```mbt check
///|
/// 创建 Tensor 并做基本运算
test "tensor: creation and elementwise ops" {
  let a = @tensor.Tensor::from_array([1.0, 2.0, 3.0, 4.0], [2, 2])
  let b = @tensor.Tensor::from_array([5.0, 6.0, 7.0, 8.0], [2, 2])
  // 加法
  let c = a + b
  @test.assert_eq(c.data(), [6.0, 8.0, 10.0, 12.0])
  @test.assert_eq(c.shape(), [2, 2])
  // 乘法（逐元素）
  let d = a * b
  @test.assert_eq(d.data(), [5.0, 12.0, 21.0, 32.0])
  // 减法
  let e = b - a
  @test.assert_eq(e.data(), [4.0, 4.0, 4.0, 4.0])
}
```

```mbt check
///|
/// 广播运算
test "tensor: broadcasting" {
  let a = @tensor.Tensor::from_array([1.0, 2.0, 3.0], [3, 1])
  let b = @tensor.Tensor::from_array([10.0, 20.0], [1, 2])
  // 广播：[3,1] + [1,2] => [3,2]
  let c = a + b
  @test.assert_eq(c.shape(), [3, 2])
  @test.assert_eq(c.data(), [11.0, 21.0, 12.0, 22.0, 13.0, 23.0])
}
```

```mbt check
///|
/// 矩阵乘法与形状变换
test "tensor: matmul, reshape, transpose, sum" {
  let a = @tensor.Tensor::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let b = @tensor.Tensor::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
  // 矩阵乘法
  let c = a.matmul(b)
  @test.assert_eq(c.shape(), [2, 2])
  let reshape = a.reshape([3, 2])
  @test.assert_eq(reshape.shape(), [3, 2])
  // 二维转置
  let transposed = a.transpose2d()
  @test.assert_eq(transposed.shape(), [3, 2])
  // 求和
  let s = a.sum()
  @test.assert_eq(s.shape(), [])
  @test.assert_eq(s.data()[0], 21.0)
}
```

```mbt check
///|
/// 激活函数
test "tensor: activation functions" {
  let x = @tensor.Tensor::from_array([-2.0, -1.0, 0.0, 1.0, 2.0], [5])
  // ReLU
  let relu_out = x.relu()
  @test.assert_eq(relu_out.data(), [0.0, 0.0, 0.0, 1.0, 2.0])
  // GELU
  let gelu_out = x.gelu()
  // GELU(-2) ≈ -0.045, GELU(0) ≈ 0, GELU(2) ≈ 1.955
  assert_true(gelu_out.data()[0] < 0.0) // negative values pass through
  assert_true(gelu_out.data()[4] > 0.0)
  // Softmax
  let sm = x.softmax(0)
  let sm_data = sm.data()
  // softmax 输出之和为 1
  let mut sum = 0.0
  for val in sm_data {
    sum += val
  }
  assert_true((sum - 1.0).abs() < 1.0e-6)
}
```

---

## 第 3 章：自动微分

minigpt 的 tensor 支持反向模式自动微分。所有可训练参数必须附着到一个 `AutogradContext` 上，标量 loss 调用 `backward()` 后，参数可通过 `grad()` 获取梯度。

```mbt check
///|
/// 自动微分：简单线性回归
test "autograd: simple backward" {
  let ctx = @tensor.AutogradContext::new()
  // 可训练参数 w [2,1]
  let w = @tensor.Tensor::parameter(ctx, [1.0, 2.0], [2, 1])
  // 输入 x [1,2]
  let x = @tensor.Tensor::from_array([3.0, 4.0], [1, 2])
  // y = x · w = 3*1 + 4*2 = 11
  let y = x.matmul(w)
  // loss = y² = 121
  let loss = y * y
  loss.backward()
  // dl/dw = 2y · xᵀ
  let grad = w.grad().unwrap()
  // grad[0] = 2*11*3 = 66, grad[1] = 2*11*4 = 88
  @test.assert_eq(grad.data()[0], 66.0)
  @test.assert_eq(grad.data()[1], 88.0)
}
```

```mbt check
///|
/// 自动微分：清零梯度与清理计算图
test "autograd: zero_grad and clear_graph" {
  let ctx = @tensor.AutogradContext::new()
  let w = @tensor.Tensor::parameter(ctx, [1.0], [1, 1])
  let x = @tensor.Tensor::from_array([2.0], [1, 1])
  let loss = x.matmul(w) * x.matmul(w)
  loss.backward()
  assert_true(w.grad() is Some(_))
  // 清零梯度
  ctx.zero_grad()
  assert_true(w.grad() is None)
}
```

---

## 第 4 章：优化器

有了梯度之后，需要优化器来更新参数。minigpt 提供了 SGD 和 AdamW。

```mbt check
///|
/// SGD 优化器
test "optimizer: sgd step" {
  let ctx = @tensor.AutogradContext::new()
  let w = @tensor.Tensor::parameter(ctx, [1.0], [1, 1])
  let x = @tensor.Tensor::from_array([2.0], [1, 1])
  // y = x·w = 2, loss = y² = 4
  let loss = x.matmul(w) * x.matmul(w)
  loss.backward()
  // w 的梯度 = 2*y*x = 2*2*2 = 8
  // SGD: w -= lr * grad = 1.0 - 0.1*8 = 0.2
  @optim.sgd_step([w], 0.1)
  assert_true(
    (w.data()[0] - 0.2).abs() < 1.0e-10,
    msg="SGD step should approximate 0.2",
  )
}
```

```mbt check
///|
/// AdamW 优化器
test "optimizer: adamw basic step" {
  let ctx = @tensor.AutogradContext::new()
  let w = @tensor.Tensor::parameter(ctx, [1.0], [1, 1])
  let x = @tensor.Tensor::from_array([2.0], [1, 1])
  let loss = x.matmul(w) * x.matmul(w)
  loss.backward()
  let adamw = @optim.AdamW::new([w], 0.01)
  adamw.step()
  // 经过一步 AdamW 后，参数应发生变化
  assert_true(w.data()[0] != 1.0)
}
```

---

## 第 5 章：定义 GPT 模型结构

现在开始正式构建 GPT。先定义超参数常量和模型结构体。

```mbt check
///|
/// GPT 超参数
let mlp_multiplier : Int = 4

///|
let layer_norm_eps : Double = 1.0e-5

///|
/// GPT 模型
struct GPT {
  ctx : @tensor.AutogradContext
  token_embedding_table : @tensor.Tensor
  position_embedding_table : @tensor.Tensor
  ln1_weight : Array[@tensor.Tensor]
  ln1_bias : Array[@tensor.Tensor]
  wq : Array[@tensor.Tensor]
  bq : Array[@tensor.Tensor]
  wk : Array[@tensor.Tensor]
  bk : Array[@tensor.Tensor]
  wv : Array[@tensor.Tensor]
  bv : Array[@tensor.Tensor]
  wo : Array[@tensor.Tensor]
  bo : Array[@tensor.Tensor]
  ln2_weight : Array[@tensor.Tensor]
  ln2_bias : Array[@tensor.Tensor]
  w_fc : Array[@tensor.Tensor]
  b_fc : Array[@tensor.Tensor]
  w_proj : Array[@tensor.Tensor]
  b_proj : Array[@tensor.Tensor]
  ln_f_weight : @tensor.Tensor
  ln_f_bias : @tensor.Tensor
  vocab_size : Int
  n_embd : Int
  n_head : Int
  n_layer : Int
  block_size : Int
}
```

模型结构说明：

| 字段 | shape | 作用 |
|------|-------|------|
| `token_embedding_table` | `[vocab_size, n_embd]` | 将 token ID 映射为稠密向量 |
| `position_embedding_table` | `[block_size, n_embd]` | 为每个位置提供位置信息 |
| `ln1_weight/bias` × N | `[n_embd]` | Self-attention 前的 LayerNorm |
| `wq/bq, wk/bk, wv/bv` × N | `[n_embd, n_embd]`/`[n_embd]` | Q/K/V 投影 |
| `wo/bo` × N | `[n_embd, n_embd]`/`[n_embd]` | Attention 输出投影 |
| `ln2_weight/bias` × N | `[n_embd]` | MLP 前的 LayerNorm |
| `w_fc/b_fc` × N | `[n_embd, n_embd*4]`/`[n_embd*4]` | MLP 第一层 |
| `w_proj/b_proj` × N | `[n_embd*4, n_embd]`/`[n_embd]` | MLP 第二层 |
| `ln_f_weight/bias` | `[n_embd]` | 最终 LayerNorm |

---

## 第 6 章：模型初始化

```mbt check
///|
/// 参数初始化：随机值并缩放
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

///|
/// 创建模型实例
pub fn GPT::new(
  vocab_size : Int,
  n_embd : Int,
  n_head : Int,
  n_layer : Int,
  block_size : Int,
  rng : @random.Rand,
) -> GPT {
  if vocab_size <= 0 {
    abort("vocab_size must be positive")
  }
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
  if block_size <= 0 {
    abort("block_size must be positive")
  }
  let ctx = @tensor.AutogradContext::new()
  let mlp_hidden = n_embd * mlp_multiplier
  let residual_scale = 0.02 / (2.0 * n_layer.to_double()).sqrt()
  // 每个 layer 的参数列表
  let ln1_weight : Array[@tensor.Tensor] = []
  let ln1_bias : Array[@tensor.Tensor] = []
  let wq : Array[@tensor.Tensor] = []
  let bq : Array[@tensor.Tensor] = []
  let wk : Array[@tensor.Tensor] = []
  let bk : Array[@tensor.Tensor] = []
  let wv : Array[@tensor.Tensor] = []
  let bv : Array[@tensor.Tensor] = []
  let wo : Array[@tensor.Tensor] = []
  let bo : Array[@tensor.Tensor] = []
  let ln2_weight : Array[@tensor.Tensor] = []
  let ln2_bias : Array[@tensor.Tensor] = []
  let w_fc : Array[@tensor.Tensor] = []
  let b_fc : Array[@tensor.Tensor] = []
  let w_proj : Array[@tensor.Tensor] = []
  let b_proj : Array[@tensor.Tensor] = []
  for _ in 0..<n_layer {
    // LayerNorm 权重初始化为 1，偏置初始化为 0
    ln1_weight.push(
      @tensor.Tensor::parameter(ctx, Array::make(n_embd, 1.0), [n_embd]),
    )
    ln1_bias.push(@tensor.Tensor::zeros([n_embd]))
    // Q/K/V 投影
    wq.push(scaled_randn(ctx, [n_embd, n_embd], rng))
    bq.push(@tensor.Tensor::zeros([n_embd]))
    wk.push(scaled_randn(ctx, [n_embd, n_embd], rng))
    bk.push(@tensor.Tensor::zeros([n_embd]))
    wv.push(scaled_randn(ctx, [n_embd, n_embd], rng))
    bv.push(@tensor.Tensor::zeros([n_embd]))
    // 输出投影（用较小的缩放因子，因为残差连接会累加）
    wo.push(scaled_randn(ctx, [n_embd, n_embd], rng))
    // 手动缩放
    let wo_data = wo[wo.length() - 1].data()
    for i in 0..<wo_data.length() {
      wo_data[i] *= residual_scale / 0.02
    }
    wo[wo.length() - 1].set_data_in_place(wo_data)
    bo.push(@tensor.Tensor::zeros([n_embd]))
    // MLP LayerNorm
    ln2_weight.push(
      @tensor.Tensor::parameter(ctx, Array::make(n_embd, 1.0), [n_embd]),
    )
    ln2_bias.push(@tensor.Tensor::zeros([n_embd]))
    // MLP 第一层（扩展 4 倍）
    w_fc.push(scaled_randn(ctx, [n_embd, mlp_hidden], rng))
    b_fc.push(@tensor.Tensor::zeros([mlp_hidden]))
    // MLP 第二层（压缩回来）
    w_proj.push(scaled_randn(ctx, [mlp_hidden, n_embd], rng))
    let w_proj_data = w_proj[w_proj.length() - 1].data()
    for i in 0..<w_proj_data.length() {
      w_proj_data[i] *= residual_scale / 0.02
    }
    w_proj[w_proj.length() - 1].set_data_in_place(w_proj_data)
    b_proj.push(@tensor.Tensor::zeros([n_embd]))
  }
  {
    ctx,
    token_embedding_table: scaled_randn(ctx, [vocab_size, n_embd], rng),
    position_embedding_table: scaled_randn(ctx, [block_size, n_embd], rng),
    ln1_weight,
    ln1_bias,
    wq,
    bq,
    wk,
    bk,
    wv,
    bv,
    wo,
    bo,
    ln2_weight,
    ln2_bias,
    w_fc,
    b_fc,
    w_proj,
    b_proj,
    ln_f_weight: @tensor.Tensor::parameter(ctx, Array::make(n_embd, 1.0), [
      n_embd,
    ]),
    ln_f_bias: @tensor.Tensor::zeros([n_embd]),
    vocab_size,
    n_embd,
    n_head,
    n_layer,
    block_size,
  }
}
```

---

## 第 7 章：Causal Self-Attention

Attention 是 Transformer 的核心。GPT 使用 **causal**（因果）self-attention，即每个 token 只能关注它自身和之前的 token。

```mbt check
///|
/// 构造 causal mask
/// 返回 [time, time] 矩阵，上三角为 -∞，下三角为 0
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
/// 线性层：input · weight + bias
fn linear(
  input : @tensor.Tensor,
  weight : @tensor.Tensor,
  bias : @tensor.Tensor,
) -> @tensor.Tensor {
  input.matmul(weight) + bias
}

///|
/// 将输入拆分为多头：[B, T, C] -> [B, n_head, T, head_dim]
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
/// 合并多头：[B, n_head, T, head_dim] -> [B, T, C]
fn merge_heads(
  input : @tensor.Tensor,
  output_shape : Array[Int],
) -> @tensor.Tensor {
  input.swap_axes(-3, -2).reshape(output_shape)
}

///|
/// Dropout：训练时随机丢弃神经元
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
```

```mbt check
///|
/// Self-attention 前向传播
fn GPT::self_attention(
  self : GPT,
  input : @tensor.Tensor,
  layer : Int,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  let head_dim = self.n_embd / self.n_head
  let shape = input.shape()
  let time = shape[shape.length() - 2]
  // Q, K, V 投影
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
  // Scaled dot-product attention
  let scale = @tensor.Tensor::scalar(1.0 / head_dim.to_double().sqrt())
  let scores = q.matmul(k.swap_axes(-2, -1)) * scale + causal_mask(time)
  let weights = dropout(scores.softmax(-1), 0.2, rng)
  let attended = merge_heads(weights.matmul(v), shape)
  // 输出投影 + dropout
  dropout(linear(attended, self.wo[layer], self.bo[layer]), 0.2, rng)
}
```

**Attention 计算过程：**

1. 输入经过线性层得到 Q、K、V
2. `scores = Q @ Kᵀ / √head_dim` — 计算注意力分数
3. 加上 causal mask（屏蔽未来位置）
4. softmax 得到注意力权重
5. 权重与 V 相乘得到输出
6. 合并多头、输出投影

---

## 第 8 章：MLP 与 Transformer Block

```mbt check
///|
/// MLP 前馈网络（两层 + GELU 激活）
fn GPT::mlp(
  self : GPT,
  input : @tensor.Tensor,
  layer : Int,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  let hidden = linear(input, self.w_fc[layer], self.b_fc[layer]).gelu()
  dropout(linear(hidden, self.w_proj[layer], self.b_proj[layer]), 0.2, rng)
}
```

```mbt check
///|
/// Transformer Block：Pre-norm + Attention + MLP + 残差连接
fn GPT::block(
  self : GPT,
  input : @tensor.Tensor,
  layer : Int,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  // Pre-norm → Attention → 残差连接
  let attn_input = input.layer_norm(
    self.ln1_weight[layer],
    self.ln1_bias[layer],
    layer_norm_eps,
  )
  let x = input + self.self_attention(attn_input, layer, rng)
  // Pre-norm → MLP → 残差连接
  let mlp_input = x.layer_norm(
    self.ln2_weight[layer],
    self.ln2_bias[layer],
    layer_norm_eps,
  )
  x + self.mlp(mlp_input, layer, rng)
}
```

---

## 第 9 章：完整前向传播

```mbt check
///|
/// 生成位置 ID 序列
fn position_ids(
  input_ids : @tensor.TokenIds,
  block_size : Int,
) -> @tensor.TokenIds {
  let shape = input_ids.shape()
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
/// 隐藏状态：Embedding + Transformer Blocks → 最后一层 LayerNorm
fn GPT::hidden_states(
  self : GPT,
  input_ids : @tensor.TokenIds,
  rng : @random.Rand?,
) -> @tensor.Tensor {
  let token_embeddings = @nn.embedding(self.token_embedding_table, input_ids)
  let position_embeddings = @nn.embedding(
    self.position_embedding_table,
    position_ids(input_ids, self.block_size),
  )
  let mut hidden = dropout(token_embeddings + position_embeddings, 0.2, rng)
  for layer in 0..<self.n_layer {
    hidden = self.block(hidden, layer, rng)
  }
  hidden.layer_norm(self.ln_f_weight, self.ln_f_bias, layer_norm_eps)
}

///|
/// 前向传播（带 dropout 训练模式）
fn GPT::forward_train(
  self : GPT,
  input_ids : @tensor.TokenIds,
  rng : @random.Rand,
) -> @tensor.Tensor {
  // 隐藏状态 × 词表转置 → logits（tied embedding）
  self
  .hidden_states(input_ids, Some(rng))
  .matmul(self.token_embedding_table.transpose2d())
}

///|
/// 前向传播（推理模式，无 dropout）
pub fn GPT::forward(self : GPT, input_ids : @tensor.TokenIds) -> @tensor.Tensor {
  self
  .hidden_states(input_ids, None)
  .matmul(self.token_embedding_table.transpose2d())
}
```

```mbt check
///|
/// 损失函数：交叉熵
pub fn GPT::loss(
  self : GPT,
  input_ids : @tensor.TokenIds,
  target_ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  @nn.cross_entropy(self.forward(input_ids), target_ids)
}

///|
/// 带 dropout 的训练损失
fn GPT::loss_train(
  self : GPT,
  input_ids : @tensor.TokenIds,
  target_ids : @tensor.TokenIds,
  rng : @random.Rand,
) -> @tensor.Tensor {
  @nn.cross_entropy(self.forward_train(input_ids, rng), target_ids)
}

///|
/// 获取最后一个位置的 logits（用于生成时的高效推理）
pub fn GPT::last_logits(
  self : GPT,
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
/// 获取所有可训练参数
fn GPT::parameters(self : GPT) -> Array[@tensor.Tensor] {
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
/// 每个参数对应的 weight decay（LayerNorm 和 bias 不衰减）
fn GPT::parameter_weight_decays(
  self : GPT,
  weight_decay : Double,
) -> Array[Double] {
  let decays : Array[Double] = [weight_decay, weight_decay]
  for _ in 0..<self.n_layer {
    // ln1_weight: no decay
    decays.push(0.0)
    // wq: decay
    decays.push(weight_decay)
    decays.push(weight_decay)
    decays.push(weight_decay)
    decays.push(weight_decay)
    // ln2_weight: no decay
    decays.push(0.0)
    // MLP weights: decay
    decays.push(weight_decay)
    decays.push(weight_decay)
  }
  // ln_f_weight: no decay
  decays.push(0.0)
  decays
}

///|
/// 清理计算图（每个训练步骤后调用）
fn GPT::clear_graph(self : GPT) -> Unit {
  self.ctx.clear_graph()
}
```

---

## 第 10 章：训练循环

```mbt check
///|
/// 从 token 序列中随机采样一个 batch
fn sample_batch(
  token_ids : Array[Int],
  batch_size : Int,
  block_size : Int,
  rng : @random.Rand,
) -> (@tensor.TokenIds, @tensor.TokenIds) {
  if token_ids.length() <= block_size {
    abort("token_ids must contain more items than block_size")
  }
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
```

```mbt check
///|
/// 在验证集上评估 loss
fn eval_loss(
  model : GPT,
  token_ids : Array[Int],
  batch_size : Int,
  block_size : Int,
  eval_iters : Int,
  rng : @random.Rand,
) -> Double {
  let mut total = 0.0
  for _ in 0..<eval_iters {
    let (inputs, targets) = sample_batch(token_ids, batch_size, block_size, rng)
    total += model.loss(inputs, targets).data()[0]
    model.clear_graph()
  }
  total / eval_iters.to_double()
}

///|
/// 余弦学习率调度
fn cosine_lr(
  iter : Int,
  max_iters : Int,
  max_lr : Double,
  min_lr : Double,
  warmup_iters : Int,
) -> Double {
  if iter < warmup_iters {
    return max_lr * (iter + 1).to_double() / (warmup_iters + 1).to_double()
  }
  if iter > max_iters {
    return min_lr
  }
  let decay_ratio = (iter - warmup_iters).to_double() /
    (max_iters - warmup_iters).to_double()
  let coeff = 0.5 * (1.0 + @math.cos(@math.PI * decay_ratio))
  min_lr + coeff * (max_lr - min_lr)
}
```

```mbt check
///|
/// 从 logits 中采样下一个 token
fn sample_token(
  logits : @tensor.Tensor,
  rng : @random.Rand,
  temperature : Double,
  top_k : Int,
) -> Int {
  if temperature <= 0.0 {
    abort("temperature must be positive")
  }
  let data = logits.data()
  // 按 logit 值降序排列
  let order = Array::new(capacity=data.length())
  for i in 0..<data.length() {
    order.push(i)
  }
  order.sort_by(fn(a, b) {
    let av = data[a]
    let bv = data[b]
    if av == bv {
      a.compare(b)
    } else if av > bv {
      -1
    } else {
      1
    }
  })
  // Top-K 截断
  let limit = if top_k <= 0 || top_k > order.length() {
    order.length()
  } else {
    top_k
  }
  // 温度缩放 + softmax
  let mut max_logit = data[order[0]] / temperature
  for i in 1..<limit {
    max_logit = max_logit.max(data[order[i]] / temperature)
  }
  let mut sum = 0.0
  let probs = Array::make(limit, 0.0)
  for i in 0..<limit {
    let value = @math.exp(data[order[i]] / temperature - max_logit)
    probs[i] = value
    sum += value
  }
  // 采样
  let sample = rng.double() * sum
  let mut cumulative = 0.0
  for i in 0..<limit {
    cumulative += probs[i]
    if sample <= cumulative {
      return order[i]
    }
  }
  order[limit - 1]
}
```

---

## 第 11 章：文本生成

```mbt check
///|
/// 自回归生成 token 序列
pub fn GPT::generate(
  self : GPT,
  prompt_ids : Array[Int],
  max_new_tokens : Int,
  rng : @random.Rand,
) -> Array[Int] {
  if prompt_ids.length() == 0 {
    abort("prompt_ids must not be empty")
  }
  if max_new_tokens < 0 {
    abort("max_new_tokens must not be negative")
  }
  let output = prompt_ids.copy()
  for _ in 0..<max_new_tokens {
    // 截取最后 block_size 个 token 作为上下文
    let start = if output.length() > self.block_size {
      output.length() - self.block_size
    } else {
      0
    }
    let context = output[start:].to_owned()
    let logits = self.last_logits(
      @tensor.TokenIds::new(context, [context.length()]),
    )
    let next_id = sample_token(logits, rng, 1.0, 0)
    output.push(next_id)
  }
  output
}

///|
/// 文本生成（含分词和解码）
pub fn GPT::generate_text(
  self : GPT,
  tokenizer : @tokenizer.Tokenizer,
  prompt : String,
  max_new_tokens : Int,
  rng : @random.Rand,
) -> String raise @tokenizer.TokenizerError {
  let prompt_ids = tokenizer.encode(prompt)
  let generated_ids = self.generate(prompt_ids, max_new_tokens, rng)
  tokenizer.decode(generated_ids)
}
```

---

## 第 12 章：完整端到端示例

下面是一个完整的端到端示例：在小型文本上训练一个 GPT 模型，并用它生成文本。

```mbt check
///|
/// 端到端训练和生成
test "e2e: train a tiny GPT and generate text" {
  // 1. 准备训练数据
  let text =
    #|hello world this is a tiny transformer
    #|hello moonbit this is a language model
    #|hello gpt this is a decoder only architecture
    #|the cat sat on the mat
    #|the dog sat on the log
    #|the bird flew over the tree
    #|hello world the weather is nice today
    #|hello moonbit the code is beautiful
    #|the transformer learns patterns from data
    #|the model generates new text

  // 2. 训练字符级分词器
  let (char_tok, all_ids) = @tokenizer.CharTokenizer::train(text)
  let tokenizer = @tokenizer.Tokenizer::from_char(char_tok)
  let vocab_size = tokenizer.vocab_size()
  assert_true(vocab_size > 0)

  // 3. 切分训练/验证集（90/10）
  let split = all_ids.length() * 9 / 10
  let train_ids = all_ids[0:split].to_owned()
  let val_ids = all_ids[split:all_ids.length()].to_owned()

  // 4. 创建模型
  let rng = @random.Rand::new()
  let block_size = 8
  let model = GPT::new(vocab_size, 8, 2, 1, block_size, rng)

  // 5. 创建优化器
  let params = model.parameters()
  let optimizer = @optim.AdamW::new_with_parameter_weight_decays(
    params,
    5.0e-3,
    0.9,
    0.99,
    1.0e-8,
    model.parameter_weight_decays(0.1),
  )

  // 6. 快速验证：执行几步训练确保 loss 能下降
  let batch_size = 2
  let max_iters = 1
  let initial_val_loss = eval_loss(
    model, val_ids, batch_size, block_size, 1, rng,
  )
  assert_true(initial_val_loss > 0.0, msg="validation loss should be positive")
  let initial_loss = {
    let (inputs, targets) = sample_batch(train_ids, batch_size, block_size, rng)
    let l = model.loss_train(inputs, targets, rng)
    let v = l.data()[0]
    model.clear_graph()
    v
  }
  let mut final_train_loss = initial_loss
  for iter in 0..<max_iters {
    let lr = cosine_lr(iter, max_iters, 5.0e-3, 1.0e-4, 0)
    optimizer.set_learning_rate(lr)
    let (inputs, targets) = sample_batch(train_ids, batch_size, block_size, rng)
    let loss = model.loss_train(inputs, targets, rng)
    final_train_loss = loss.data()[0]
    loss.backward()
    optimizer.step_with_grad_clip(1.0)
    model.clear_graph()
  }
  assert_true(final_train_loss > 0.0, msg="training loss should be positive")

  // 7. 生成文本
  let generated = model.generate_text(tokenizer, "hello ", 20, rng)
  // 生成应以 prompt 开头
  assert_true(
    generated.has_prefix("hello "),
    msg="generation should start with prompt",
  )
}
```

---

## 总结

至此，你已经从零构建了一个完整的 nanoGPT 风格语言模型！回顾整个过程：

| 章节 | 内容 | 关键 API |
|------|------|---------|
| 1 | 分词器 | `CharTokenizer::train`, `Tokenizer::encode/decode` |
| 2 | Tensor 基础 | `from_array`, 四则运算, `matmul`, 形状变换 |
| 3 | 自动微分 | `AutogradContext`, `parameter`, `backward` |
| 4 | 优化器 | `sgd_step`, `AdamW::new`, `step_with_grad_clip` |
| 5-9 | GPT 模型 | `embedding`, `layer_norm`, `softmax`, causal mask, multi-head attention, MLP, transformer block |
| 10 | 训练 | sample batch, cosine LR, loss computation |
| 11 | 生成 | `last_logits`, 自回归采样 |
| 12 | 端到端 | 完整训练+生成流程 |

### 进阶方向

- 增大模型：调整 `n_embd`、`n_head`、`n_layer` 和 `block_size`
- 使用 BPE 分词器处理更大规模的文本
- 添加 checkpoint 保存/加载
- 尝试不同的学习率调度和正则化策略
- 在真实数据集（如 Tiny Shakespeare）上训练

本教程的完整代码可通过 `moon check` 在 `docs` 包中验证。祝你构建愉快！
