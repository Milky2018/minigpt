# MiniGPT in MoonBit

这是一个教学用的极简 MiniGPT 项目。它用 2MB 以内的古诗 UTF-8 文本作为语料，演示从字符 tokenizer、Tensor/自动微分、简单语言模型训练，到自动补全生成的完整流程。

当前目标不是训练一个高质量模型，而是让学生能看到一个小型 GPT 风格程序的主要部件如何连起来。

## 数据

默认语料文件：

```bash
data/poems_2m.txt
```

CLI 默认只使用语料前 5000 个字符，避免课堂演示时训练和 checkpoint 文件过大。需要使用完整语料时可以传：

```bash
--max-chars 0
```

## 训练

训练会生成一个 checkpoint。默认输出文件是 `minigpt-model.json`：

```bash
moon run --target native cmd/main -- train
```

常用小步演示命令：

```bash
moon run --target native cmd/main -- train --steps 10 --out minigpt-model.json
```

训练参数：

```text
--data            UTF-8 语料路径，默认 data/poems_2m.txt
--max-chars       使用多少个语料字符，0 表示完整语料，默认 5000
--out             checkpoint 输出路径，默认 minigpt-model.json
--steps           训练步数，默认 10
--batch-size      batch size，默认 4
--block-size      上下文长度，默认 4
--n-embd          embedding 维度，默认 32
--learning-rate   学习率，默认 0.001
```

## 生成

生成只加载已经训练好的 checkpoint，不会重新训练：

```bash
moon run --target native cmd/main -- generate --model minigpt-model.json --prompt 春
```

每生成一个字，都会打印当前已经补全出的完整内容：

```text
completion:
春
春神
春神五
春神五失
春神五失九
春神五失九所
```

生成参数：

```text
--model           checkpoint 路径，默认 minigpt-model.json
--prompt          补全起始文本，默认 春
--max-new-tokens  生成字符数，默认 80
```

## Checkpoint 大小

当前 checkpoint 使用 JSON，方便教学时打开查看，但比二进制格式大。

默认演示配置大致为：

```text
max chars = 5000
vocab size ~= 1309
n_embd = 32
parameters = 2 * vocab_size * n_embd ~= 83,776 Doubles
JSON checkpoint ~= 2.1MB
```

如果希望 checkpoint 小一点，可以降低 `--max-chars` 或 `--n-embd`：

```bash
moon run --target native cmd/main -- train --max-chars 2000 --n-embd 16
```

注意：如果训练语料窗口太小，默认 prompt `春` 可能不在 tokenizer 词表里，生成时需要换成词表中出现过的字符，或者增大 `--max-chars`。

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
moon run --target native cmd/main -- train --steps 1 --out /tmp/minigpt-smoke.json
moon run --target native cmd/main -- generate --model /tmp/minigpt-smoke.json --prompt 上 --max-new-tokens 5
```
