# MiniGPT in MoonBit

这是一个用 MoonBit 写的极简自动补全程序。当前目标是尽量对齐 Andrej Karpathy `nanoGPT` 的 `shakespeare_char` 配置：同一份 Tiny Shakespeare 语料、字符级 tokenizer、90/10 train/val 切分，以及同一组小型 GPT 超参数。命令行里训练时主要只改 `--steps`。

当前模型是 GPT decoder-only transformer：token embedding + position embedding + 6 层 decoder block。每个 block 包含 pre-norm multi-head causal self-attention、MLP/GELU、dropout 和残差连接，最后用 tied token embedding 作为输出头。它不是语料检索，不会复制 prompt 后面的原文片段。

## 数据

默认语料文件：

```bash
data/tiny_shakespeare.txt
```

这个文件来自 Karpathy `char-rnn` 仓库里的 Tiny Shakespeare 数据集，也是 nanoGPT `data/shakespeare_char/prepare.py` 下载的同一份文本。

字符级准备流程：

```text
characters = 1,115,394
vocab size = 65
train tokens = 1,003,854
val tokens = 111,540
split = first 90% train, last 10% val
```

## 训练

训练会从语料中构造字符级 tokenizer，并按 nanoGPT 的 `always_save_checkpoint = False` 行为在 eval 时保存验证集 loss 创新低的 checkpoint。默认输出文件是 `minigpt-model.bin`：

```bash
moon run --release cmd/main -- train
```

指定输出文件或训练迭代数：

```bash
moon run --release cmd/main -- train --out minigpt-model.bin --steps 5000
```

训练参数默认对齐 nanoGPT，也可以显式覆盖成小配置做 smoke/benchmark：

```text
--data                    UTF-8 语料路径，默认 data/tiny_shakespeare.txt
--out                     checkpoint 输出路径，默认 minigpt-model.bin
--steps                   训练迭代数，默认 5000
--batch-size              batch size，默认 64
--block-size              上下文长度，默认 256
--n-embd                  embedding width，默认 384
--n-head                  attention head count，默认 6
--n-layer                 transformer block count，默认 6
--learning-rate           学习率，默认 0.001
--eval-interval           每多少个 iter 做 eval，默认 250
--eval-iters              每次 eval 的 batch 数，默认 200
--log-interval            每多少个 iter 打印 train loss，默认 10
--always-save-checkpoint  每次 eval 后保存 checkpoint，默认 false
```

`--steps` 对应 nanoGPT 的 `max_iters`，循环结束条件是 `iter_num > max_iters`，因此 `--steps 1` 会执行 iter 0 和 iter 1 两次更新。iter 0 的 eval 不保存 checkpoint；只有 `iter_num > 0` 且 val loss 创新低时才会写入 `--out`。

快速走通完整训练和生成流程可以用小模型配置：

```bash
moon run --release cmd/main -- train \
  --out /tmp/minigpt-small.bin \
  --steps 3 \
  --batch-size 4 \
  --block-size 8 \
  --n-embd 24 \
  --n-head 4 \
  --n-layer 2 \
  --eval-interval 1 \
  --eval-iters 1 \
  --log-interval 1 \
  --always-save-checkpoint true

moon run --release cmd/main -- generate \
  --model /tmp/minigpt-small.bin \
  --prompt ROMEO: \
  --max-new-tokens 8
```

内置训练超参对齐 nanoGPT `config/train_shakespeare_char.py`：

```text
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
learning_rate = 0.001
min_lr = 0.0001
warmup_iters = 100
eval_interval = 250
eval_iters = 200
log_interval = 10
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
```

## 生成

生成只加载已经训练好的 checkpoint，不会重新训练：

```bash
moon run --release cmd/main -- generate --model minigpt-model.bin --prompt ROMEO:
```

如果 `minigpt-model.bin` 不存在，先运行上面的 `train` 命令；训练步数太少时可能不会触发 nanoGPT 的保存条件。

每生成一个字符，都会打印当前已经补全出的完整内容；生成长度由 `--max-new-tokens` 控制：

```text
completion:
ROMEO:
ROMEO:T
ROMEO:Th
ROMEO:The
```

生成参数：

```text
--model           checkpoint 路径，默认 minigpt-model.bin
--prompt          补全起始文本，默认 ROMEO:
--max-new-tokens  生成字符数，默认 80
--top-k           从模型 logits 最高的几个候选中采样，默认 20
--temperature     采样温度，默认 0.8
```

字符级 tokenizer 只能编码训练语料词表里的 65 个字符。默认 Shakespeare 语料不包含中文字符，所以中文 prompt 会被拒绝。

## Checkpoint 大小

当前 checkpoint 使用紧凑二进制格式。默认 GPT 配置大致为：

```text
tokenizer = char
vocab size = 65
model kind = gpt-transformer
n_embd = 384
n_head = 6
n_layer = 6
block size = 256
model parameters ~= 10.6M Doubles
training checkpoint ~= 250MB
```

## 项目结构

```text
tokenizer.mbt       character tokenizer and train/val split
tensor/             Tensor 和自动微分基础
nn/                 神经网络基础算子
optim/              AdamW 优化器
model.mbt           MiniGPT 模型
train.mbt           nanoGPT-style 训练循环
generate.mbt        采样生成
checkpoint.mbt      checkpoint 编解码
cmd/main/           CLI 入口
docs/               教学架构图
data/               Tiny Shakespeare 语料
```

## 验证

```bash
moon check --warn-list +73 --target native
moon test --target native
```
