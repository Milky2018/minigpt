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

训练会从语料中构造字符级 tokenizer，然后生成一个二进制 checkpoint。默认输出文件是 `minigpt-model.bin`：

```bash
moon run --release cmd/main -- train
```

指定输出文件或训练迭代数：

```bash
moon run --release cmd/main -- train --out minigpt-model.bin --steps 5000
```

训练参数：

```text
--data   UTF-8 语料路径，默认 data/tiny_shakespeare.txt
--out    checkpoint 输出路径，默认 minigpt-model.bin
--steps  训练迭代数，默认 5000
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

如果 `minigpt-model.bin` 不存在，先运行上面的 `train` 命令生成它。

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
parameters ~= 10.6M Doubles
binary checkpoint ~= 80-90MB
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
