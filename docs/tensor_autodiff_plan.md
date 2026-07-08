# Tensor Autodiff Implementation Plan

This plan captures the agreed design for the MiniGPT tensor/autodiff foundation.
It is a design and sequencing document, not the implementation.

## Confirmed Decisions

- Public numeric tensor type is `Tensor`.
- Tensor data, gradients, and optimizer state use `Double`.
- Tensor shape is runtime-ranked: `Array[Int]`.
- Tensor storage is contiguous row-major.
- There are no strided views in the first design.
- NumPy-style right-aligned broadcasting is supported.
- Gradients for broadcast inputs are summed back to the original input shape.
- Reverse-mode autodiff is owned by an explicit `AutogradContext`.
- No global tape is used.
- Tensor operations panic on invalid shape or mixed-context inputs.
- `+`, `-`, `*`, `/`, and unary negation are elementwise operations.
- `*` is not matrix multiplication; matrix multiplication is explicit `matmul`.
- `backward()` starts from a scalar loss only.
- Gradients are owned by autograd nodes and queried through tensors.
- Parameters are long-lived trainable values.
- Operation nodes are short-lived forward-pass nodes.
- Normal tensor expressions are immutable.
- Optimizers mutate parameters, not arbitrary temporary tensors.
- Both SGD and AdamW are required.
- Random initialization and sampling use explicit RNG values.
- Token ids and class targets use `TokenIds`, not `Tensor`.
- `cross_entropy(logits, targets)` accepts `TokenIds` targets and returns a
  scalar loss.
- `softmax(axis)` is a public differentiable tensor operation.
- `cross_entropy` is implemented as a numerically stable fused operation.
- Attention masks are ordinary non-gradient tensors with `0.0` for allowed
  positions and a large negative value for blocked positions.
- Finite-difference gradient checking is required for tensor operators.

## Data Structure Sketch

```text
Tensor
  data: Array[Double]
  shape: Array[Int]
  requires_grad: Bool
  context: AutogradContext?
  node_id: Int?

TokenIds
  data: Array[Int]
  shape: Array[Int]

AutogradContext
  nodes: Array[Node]

Node
  kind: Parameter | Operation
  value_shape: Array[Int]
  grad: Array[Double]?
  parents: Array[Edge]
  backward: operation-specific rule

Edge
  parent: node id
  local backward metadata
```

The final MoonBit representation may split `Node` variants more concretely, but
the ownership rules should stay the same: parameters survive across steps,
operation nodes are cleared after a training step, and gradients live on nodes.

## Public API Sketch

## Package Layout

The tensor/autodiff foundation should be split into packages instead of living
entirely in the root package:

```text
/tensor
  Tensor, TokenIds, shape utilities, broadcasting, autograd context,
  differentiable tensor operations including softmax, gradient checker

/optim
  SGD and AdamW optimizers over tensor parameters

/nn
  MiniGPT-oriented neural-network helpers such as embedding, cross_entropy,
  layers, and masks

/
  tokenizer and lightweight project facade

/cmd/main
  CLI/demo entry point
```

`/tensor` is the deepest package and must not depend on `/nn` or `/optim`.
`/optim` may depend on `/tensor`. `/nn` may depend on `/tensor` and optionally
on `/optim` only for higher-level training helpers; primitive layers should not
depend on optimizers. `TokenIds` and `softmax` belong in `/tensor`;
`cross_entropy` belongs in `/nn`.

```text
AutogradContext::new() -> AutogradContext
AutogradContext::zero_grad(self) -> Unit
AutogradContext::clear_graph(self) -> Unit

Tensor::scalar(value: Double) -> Tensor
Tensor::zeros(shape: Array[Int]) -> Tensor
Tensor::ones(shape: Array[Int]) -> Tensor
Tensor::randn(ctx, shape: Array[Int], rng) -> Tensor
Tensor::parameter(ctx, shape: Array[Int], rng) -> Tensor

Tensor::shape(self) -> Array[Int]
Tensor::data(self) -> Array[Double]
Tensor::grad(self) -> Tensor?
Tensor::requires_grad(self) -> Bool
Tensor::backward(self) -> Unit

TokenIds::new(data: Array[Int], shape: Array[Int]) -> TokenIds
```

Operator-backed tensor expressions:

```text
a + b
a - b
a * b
a / b
-a
```

Named tensor operations:

```text
matmul(a, b)
reshape(a, shape)
transpose2d(a)
swap_axes(a, axis_a, axis_b)
sum(a, axis?, keepdims?)
mean(a, axis?, keepdims?)
exp(a)
log(a)
relu(a)
softmax(a, axis)
embedding(weight, ids: TokenIds)
cross_entropy(logits, targets: TokenIds)
```

Optimizer API:

```text
sgd_step(params, lr)

AdamW::new(params, lr, beta1, beta2, eps, weight_decay)
AdamW::step(self)
AdamW::zero_grad(self)
```

## Operator Set

Creation:

- scalar
- zeros
- ones
- randn
- parameter

Shape:

- reshape
- transpose2d
- swap_axes
- sum
- mean

Elementwise:

- add
- sub
- mul
- div
- neg
- exp
- log
- relu

Linear and neural-network operations:

- matmul
- embedding
- softmax
- cross_entropy

## Testing Strategy

The tensor layer needs both value tests and gradient tests.

Value tests:

- shape validation
- contiguous row-major indexing
- broadcasting shape inference
- elementwise broadcasting results
- matmul results
- reductions with and without `keepdims`
- TokenIds shape validation
- cross-entropy value on small logits

Gradient tests:

- add with broadcasting
- sub with broadcasting
- mul with broadcasting
- div with broadcasting
- neg
- exp
- log
- relu positive and negative branches
- sum and mean
- matmul
- embedding repeated token ids
- softmax along a non-last axis
- fused cross-entropy

Gradient checks should use finite differences on small tensors and compare with
autodiff gradients using explicit tolerance.

## Implementation Order

1. Shape utilities:
   shape product, row-major offset, broadcast shape, broadcast index mapping,
   and reduce-to-shape for broadcast gradients.

2. Plain tensor value layer:
   `Tensor` construction, `TokenIds`, contiguous data validation, elementwise
   value operations, reductions, and matmul without autodiff.

3. Autograd context and node model:
   context ownership, parameter nodes, operation nodes, scalar `backward`,
   `zero_grad`, and graph clearing.

4. Autograd for elementwise operations:
   `+`, `-`, `*`, `/`, `neg`, broadcast-aware gradient reduction.

5. Autograd for shape and reduction operations:
   reshape, transpose/swap axes, sum, mean.

6. Autograd for neural-network operations:
   matmul, embedding, softmax, fused cross-entropy.

7. Optimizers:
   SGD first for tests, then AdamW for practical training.

8. MiniGPT integration:
   parameter initialization, embedding tables, attention mask, attention block,
   MLP block, loss, and generation.

## Non-goals

- Generic dtype tensors.
- Float32 tensors.
- Strided views.
- Sparse tensors.
- GPU or SIMD backend.
- Recoverable tensor operation errors.
- Non-scalar `backward` seeds.
- Global tape.
- Encoding token ids as `Double` tensors.
