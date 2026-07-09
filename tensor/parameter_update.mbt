///|
pub fn Tensor::assign_parameter_data(
  self : Tensor,
  data : Array[Double],
) -> Unit {
  match (self.context, self.node_ref) {
    (Some(ctx), Some(ParamRef(id))) => {
      if data.length() != shape_size(ctx.state.val.params[id].shape) {
        abort("assign_parameter_data requires the same number of elements")
      }
      for i in 0..<data.length() {
        self.data[i] = data[i]
      }
    }
    _ => abort("assign_parameter_data requires a parameter tensor")
  }
}
