// Learn more about moon.mod configuration:
// https://docs.moonbitlang.com/en/latest/toolchain/moon/module.html
//
// To add a dependency, run this command in your terminal:
//   moon add moonbitlang/x
//
// Or manually declare it in `import`, for example:
// import {
//   "moonbitlang/x@0.4.6",
// }

name = "Milky2018/minigpt"

version = "0.1.2"

readme = "README.mbt.md"

repository = "https://github.com/Milky2018/minigpt"

license = "Apache-2.0"

keywords = [ "gpt", "transformer", "tensor", "autodiff", "training" ]

preferred_target = "native"

description = "A small MoonBit GPT training toolkit with tensor autodiff, tokenizer support, checkpoints, and a teaching CLI."

import {
  "moonbitlang/x@0.5.1",
}
