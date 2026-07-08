# MiniGPT in MoonBit

这是一个用 MoonBit 写的极简自动补全程序。它默认使用 Andrej Karpathy 的 Tiny Shakespeare UTF-8 文本训练一个小型 byte-level BPE tokenizer 和语言模型，然后根据 prompt 逐步补全文本。

当前默认模型是一个小型 GPT transformer：token embedding + position embedding + 2 层 decoder block。每个 block 包含 pre-norm multi-head causal self-attention、MLP/GELU 和残差连接，最后用 tied token embedding 作为输出头。它不是语料检索，不会复制 prompt 后面的原文片段。

## 数据

默认语料文件：

```bash
data/tiny_shakespeare.txt
```

这个文件来自 Karpathy `char-rnn` 仓库里的 Tiny Shakespeare 数据集，也是 nanoGPT 准备脚本下载的同一份文本。CLI 默认使用完整语料；需要更快的 smoke test 或更小的 checkpoint 时，可以限制语料窗口：

```bash
--max-chars 5000
```

## 训练

训练会先从语料中训练一个小型 byte-level BPE tokenizer，然后生成一个二进制 checkpoint。默认输出文件是 `minigpt-model.bin`：

```bash
moon run --release cmd/main -- train
```

指定输出文件：

```bash
moon run --release cmd/main -- train --out minigpt-model.bin
```

训练参数：

```text
--data            UTF-8 语料路径，默认 data/tiny_shakespeare.txt
--max-chars       使用多少个语料字符，0 表示完整语料，默认 0
--out             checkpoint 输出路径，默认 minigpt-model.bin
--steps           梯度训练步数，默认 10
--batch-size      batch size，默认 4
--block-size      上下文长度，默认 8
--vocab-size      BPE tokenizer 词表大小，默认 512
--learning-rate   学习率，默认 0.001
```

## 生成

生成只加载已经训练好的 checkpoint，不会重新训练：

```bash
moon run --release cmd/main -- generate --model minigpt-model.bin --prompt ROMEO:
```

如果 `minigpt-model.bin` 不存在，先运行上面的 `train` 命令生成它。

每生成一个 token，都会打印当前已经补全出的完整内容；生成长度由 `--max-new-tokens` 控制：

```text
completion:
ROMEO:
ROMEO:T
ROMEO:Th
ROMEO:The
ROMEO:The
ROMEO:The l
```

生成参数：

```text
--model           checkpoint 路径，默认 minigpt-model.bin
--prompt          补全起始文本，默认 ROMEO:
--max-new-tokens  生成 token 数，默认 80
--top-k           从模型 logits 最高的几个候选中采样，默认 20
--temperature     采样温度，默认 0.8
```

## Checkpoint 大小

当前 checkpoint 使用紧凑二进制格式。默认 GPT transformer 会保存 embedding、每层 LayerNorm、attention projection、MLP 参数和最终 LayerNorm；输出头复用 token embedding，不再单独保存一份 lm head。

默认完整语料配置大致为：

```text
max chars = 0
tokenizer = byte-bpe
vocab size = 512
model kind = gpt-transformer
n_embd = 32
n_head = 4
n_layer = 2
block size = 8
parameters ~= 42K Doubles
binary checkpoint ~= 340KB
```

如果希望 checkpoint 小一点，可以限制 `--max-chars`：

```bash
moon run --release cmd/main -- train --max-chars 2000
```

BPE tokenizer 和 merge list 会保存在 checkpoint 里；`generate` 只加载 checkpoint，不会重新训练 tokenizer。byte-level BPE 可以编码任意 UTF-8 prompt，不会再因为 prompt 含有训练语料外的字符而失败。

## 项目结构

```text
tokenizer.mbt       byte-level BPE tokenizer
tensor/             Tensor 和自动微分基础
nn/                 神经网络基础算子
optim/              优化器
model.mbt           MiniGPT 模型
train.mbt           训练循环
generate.mbt        采样生成
checkpoint.mbt      checkpoint 编解码
cmd/main/           CLI 入口
docs/               教学架构图
data/               Tiny Shakespeare 语料
```

## 验证

```bash
moon check --warn-list +73
moon test
moon run --release cmd/main -- train --out /tmp/minigpt-smoke.bin
moon run --release cmd/main -- generate --model /tmp/minigpt-smoke.bin --prompt ROMEO: --max-new-tokens 5
```
