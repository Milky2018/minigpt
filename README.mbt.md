# MiniGPT in MoonBit

这是一个用 MoonBit 写的极简中文自动补全程序。它用 2MB 以内的古诗 UTF-8 文本训练字符级语言模型，然后根据 prompt 逐步补全文本。

当前默认模型是 full-rank character bigram：每个字符直接学习一个完整的后继字符 logits 分布。它不是语料检索，不会复制 prompt 后面的原文片段。

## 数据

默认语料文件：

```bash
data/poems_2m.txt
```

CLI 默认使用完整语料。需要更快的 smoke test 或更小的 checkpoint 时，可以限制语料窗口：

```bash
--max-chars 5000
```

## 训练

训练会生成一个二进制 checkpoint。默认输出文件是 `minigpt-model.bin`：

```bash
moon run --release cmd/main -- train
```

指定输出文件：

```bash
moon run --release cmd/main -- train --out minigpt-model.bin
```

训练参数：

```text
--data            UTF-8 语料路径，默认 data/poems_2m.txt
--max-chars       使用多少个语料字符，0 表示完整语料，默认 0
--out             checkpoint 输出路径，默认 minigpt-model.bin
--steps           额外梯度微调步数，默认 0
--batch-size      batch size，默认 4
--block-size      上下文长度，默认 4
--learning-rate   学习率，默认 0.001
```

## 生成

生成只加载已经训练好的 checkpoint，不会重新训练：

```bash
moon run --release cmd/main -- generate --model minigpt-model.bin --prompt 春
```

如果 `minigpt-model.bin` 不存在，先运行上面的 `train` 命令生成它。

每生成一个字，都会打印当前已经补全出的完整内容；生成长度由 `--max-new-tokens` 控制：

```text
completion:
花
花鳥
花鳥聲
花鳥聲。
花鳥聲。玉
花鳥聲。玉，
```

生成参数：

```text
--model           checkpoint 路径，默认 minigpt-model.bin
--prompt          补全起始文本，默认 春
--max-new-tokens  生成字符数，默认 80
--top-k           从模型 logits 最高的几个候选中采样，默认 5
```

## Checkpoint 大小

当前 checkpoint 使用紧凑二进制格式。默认 full-rank bigram 会保存一个 `[vocab_size, vocab_size]` logits 矩阵。

默认完整语料配置大致为：

```text
max chars = 0
vocab size ~= 5934
model kind = full-rank-bigram
parameters = vocab_size * vocab_size ~= 35,212,356 Doubles
binary checkpoint ~= 269MB
```

如果希望 checkpoint 小一点，可以限制 `--max-chars`：

```bash
moon run --release cmd/main -- train --max-chars 2000
```

注意：如果训练语料窗口太小，prompt 里的字可能不在 tokenizer 词表里，生成时需要换成词表中出现过的字符，或者增大 `--max-chars`。

## 项目结构

```text
tokenizer.mbt       字符级 tokenizer
tensor/             Tensor 和自动微分基础
nn/                 神经网络基础算子
optim/              优化器
model.mbt           MiniGPT 模型
train.mbt           训练循环
generate.mbt        采样生成
checkpoint.mbt      checkpoint 编解码
cmd/main/           CLI 入口
docs/               教学架构图
data/               古诗语料
```

## 验证

```bash
moon check --warn-list +73
moon test
moon run --release cmd/main -- train --out /tmp/minigpt-smoke.bin
moon run --release cmd/main -- generate --model /tmp/minigpt-smoke.bin --prompt 上 --max-new-tokens 5
```
