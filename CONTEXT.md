# MiniGPT

This context describes the teaching vocabulary for a small MoonBit GPT-style
project.

## Language

**Tensor Autodiff**:
Reverse-mode automatic differentiation over tensor values, where tensor
operations build a computation graph and gradients are propagated from a scalar
loss.
_Avoid_: Scalar tape autodiff, numerical differentiation

**Operator-backed Tensor Operation**:
A tensor operation exposed through MoonBit operator traits such as `Add` and
`Mul`, so ordinary expressions participate in automatic differentiation.
_Avoid_: Only named helper functions

**N-dimensional Tensor**:
A contiguous tensor whose rank is determined by a runtime shape array, rather
than being limited to vectors or matrices.
_Avoid_: Fixed-rank tensor

**Broadcasting**:
NumPy-style right-aligned expansion where dimensions are compatible when equal
or when one side is `1`; gradients for broadcast inputs are summed back to the
input's original shape.
_Avoid_: Ad hoc shape expansion

**Contiguous Tensor**:
An N-dimensional tensor stored in row-major contiguous memory; reshaping or
permuting does not create shared-memory views in the first design.
_Avoid_: Strided view

**Autograd Context**:
An explicit owner of computation graph nodes used by tensors during reverse-mode
automatic differentiation.
_Avoid_: Global tape

**Context-compatible Operation**:
A tensor operation whose differentiable inputs are attached to the same
autograd context; mixed contexts are invalid.
_Avoid_: Cross-context operation

**Panic-on-invalid Tensor Operation**:
A tensor operation that aborts immediately when shape or autograd-context
requirements are violated, instead of returning a recoverable error.
_Avoid_: Recoverable tensor operation error

**Elementwise Multiplication**:
The meaning of the `*` operator for tensors; matrix multiplication is represented
by an explicit `matmul` operation.
_Avoid_: Star-as-matmul

**MiniGPT Tensor Core**:
The first tensor/autodiff surface should include enough creation, shape,
elementwise, linear algebra, embedding, softmax, and cross-entropy operations to
write the GPT model without awkward workarounds.
_Avoid_: Minimal scalar-only operator set

**Scalar Loss**:
A single-value tensor used as the starting point for `backward`; non-scalar
outputs must be reduced before gradient propagation.
_Avoid_: Implicit non-scalar backward seed

**Node-owned Gradient**:
Gradient data stored on the autograd context's graph node and accessed through
tensors by node id.
_Avoid_: Tensor-owned gradient

**Optimizer**:
A separate training component that reads parameter gradients and mutates
parameter values; it is not part of tensor value semantics.
_Avoid_: Tensor-owned optimizer step

**SGD**:
A simple optimizer used for teaching and gradient-correctness tests.
_Avoid_: Only production optimizer

**AdamW**:
The adaptive optimizer used for practical MiniGPT training.
_Avoid_: SGD-only training

**Double Tensor**:
The only numeric tensor type in the first design, using `Double` values for
data, gradients, and optimizer state.
_Avoid_: Generic tensor dtype, Float32 tensor

**Tensor**:
The public name for the project's `Double`-backed N-dimensional tensor type.
_Avoid_: DoubleTensor, TensorD

**TokenIds**:
A non-differentiable integer tensor-like value for token indices and class
targets, stored separately from `Tensor`.
_Avoid_: Encoding token ids as Double tensors

**Cross Entropy**:
A scalar loss operation that consumes logits as `Tensor` and target classes as
`TokenIds`, without requiring one-hot target tensors.
_Avoid_: One-hot target loss

**Attention Mask**:
A non-gradient tensor added to attention scores, using `0.0` for allowed
positions and a large negative value for blocked positions.
_Avoid_: Boolean mask in tensor math

**Softmax**:
A public differentiable tensor operation for normalizing values along an axis.
_Avoid_: Loss-only softmax

**Fused Cross Entropy**:
A numerically stable cross-entropy implementation that computes its own
backward pass instead of expanding into a large softmax/log/gather graph.
_Avoid_: Naive softmax-log loss graph

**Explicit RNG**:
A random number generator passed explicitly into initialization, sampling, and
any stochastic training path so runs can be reproduced from a seed.
_Avoid_: Global random state

**Requires Grad**:
A tensor flag indicating that operations involving the tensor should create
autograd graph nodes and retain gradients.
_Avoid_: Always-on graph recording

**Parameter Node**:
A long-lived autograd node that owns trainable tensor state across optimization
steps.
_Avoid_: Treating parameters as disposable operation nodes

**Operation Node**:
A short-lived autograd node created during a forward pass and cleared after the
training step.
_Avoid_: Keeping every forward graph forever

**Parameter**:
A trainable tensor value whose data may be mutated by an optimizer while normal
tensor expressions remain immutable.
_Avoid_: Mutable temporary tensor

**Low-rank Bigram Model**:
The first runnable MiniGPT model factorizes next-token logits as token
embedding plus output projection, using `[vocab_size, n_embd]` and
`[n_embd, vocab_size]` parameters instead of a dense `[vocab_size, vocab_size]`
bigram table.
_Avoid_: Full-vocabulary bigram matrix

**Full-rank Bigram Model**:
A MiniGPT backend whose trainable logits are a full `[vocab_size, vocab_size]`
next-token matrix initialized from adjacent-token maximum-likelihood counts and
then saved as normal model parameters.
_Avoid_: Corpus substring lookup or side-channel sampler

**Gradient Checker**:
A testing utility that compares autodiff gradients against finite-difference
numerical gradients for small tensor programs.
_Avoid_: Trusting unverified gradients

**CLI Entry Command**:
A runnable `cmd/main` subcommand that wires corpus loading, tokenization,
training, and generation together for demos without owning model logic.
_Avoid_: Putting core model or tensor implementation in `cmd/main`

**Train Command**:
The CLI entry command that reads a text corpus, builds the character tokenizer,
creates a fresh MiniGPT model, trains it for the requested number of steps, and
persists a checkpoint for later generation.
_Avoid_: Training without saving weights

**Generate Command**:
The CLI entry command that loads a trained checkpoint, restores its tokenizer
and model weights, and samples continuation text from a prompt.
_Avoid_: Re-training inside generation

**Model Checkpoint**:
A compact binary file containing checkpoint version, model kind, character
vocabulary, embedding dimension, token embedding weights, and output projection
weights.
_Avoid_: Saving weights without tokenizer vocabulary

**Completion Trace**:
The CLI output that prints the full accumulated completion after each generated
token until the `--max-new-tokens` limit.
_Avoid_: Final-only generation output

**CLI Corpus Window**:
The optional character prefix limit used by CLI demos to shrink the bigram
checkpoint for smoke tests; `0` means use the full corpus and is the default.
_Avoid_: Assuming the default model only saw a tiny corpus prefix
