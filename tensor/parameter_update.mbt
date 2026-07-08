///|
pub fn Tensor::set_data_in_place(self : Tensor, data : Array[Double]) -> Unit {
  if data.length() != self.data.length() {
    abort("set_data_in_place requires the same number of elements")
  }
  for i in 0..<data.length() {
    self.data[i] = data[i]
  }
}
