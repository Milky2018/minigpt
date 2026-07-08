///|
pub fn embedding(
  weight : @tensor.Tensor,
  ids : @tensor.TokenIds,
) -> @tensor.Tensor {
  weight.gather_rows(ids)
}

///|
pub fn cross_entropy(
  logits : @tensor.Tensor,
  targets : @tensor.TokenIds,
) -> @tensor.Tensor {
  logits.cross_entropy(targets)
}
