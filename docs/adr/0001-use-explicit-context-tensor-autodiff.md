# Use explicit-context tensor reverse-mode autodiff

We will build the MiniGPT training foundation around N-dimensional contiguous
`Double` tensors with reverse-mode automatic differentiation owned by an
explicit autograd context. This deliberately goes beyond the scalar tape model
from the teaching reference because the project needs to support a real GPT
training path with broadcasting, matrix multiplication, embedding, softmax,
cross-entropy, and operator-backed tensor expressions.

## Considered Options

- Scalar tape autodiff: simpler to teach, but not sufficient for implementing
  MiniGPT without a second tensor system.
- Global tensor tape: terse, but fragile for tests, repeated training runs, and
  multiple model instances.
- Explicit-context tensor autodiff: more API surface, but the graph ownership,
  parameter lifetime, and gradient storage rules are clear.

## Consequences

Tensor operations will panic on invalid shape or context combinations. Tensor
expressions are immutable; trainable parameters are long-lived nodes that an
optimizer may mutate. Operation nodes are short-lived and cleared between
training steps. The first optimizer surface includes both SGD for simple tests
and AdamW for practical MiniGPT training, and tensor gradients must be validated
with finite-difference gradient checks.
