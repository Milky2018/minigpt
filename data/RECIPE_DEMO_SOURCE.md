# Recipe Demo Corpus Source

`recipe_demo.txt` started from the Cooklang-style recipe sentences in
`src/config.mbt` from `habakan/moonbit-gpt-edge-demo`, then was rewritten into a
small controlled-vocabulary teaching corpus.

Source repository:
<https://github.com/habakan/moonbit-gpt-edge-demo>

Source file:
<https://github.com/habakan/moonbit-gpt-edge-demo/blob/main/src/config.mbt>

Source commit:
`65b7e1a790d6aea702bb3169d1908ccc85e5129f`

License: Apache-2.0. The upstream repository contains no separate `NOTICE`
file at the time this corpus was added. This repository also carries the
Apache-2.0 license in the root `LICENSE` file.

Changes made here:

- Converted the MoonBit `Array[String]` seed corpus into a UTF-8 plain text file.
- Rewrote and expanded it into one recipe sentence per line.
- Kept the word-level tokenizer vocabulary at 50 tokens including the newline
  token.
