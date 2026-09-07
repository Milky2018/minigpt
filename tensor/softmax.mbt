///|
pub fn Tensor::softmax(self : Tensor, axis : Int) -> Tensor {
  let ax = normalize_axis(axis, self.shape.length())
  let outer = shape_size(self.shape[0:ax].to_owned())
  let dim = self.shape[ax]
  let inner = shape_size(self.shape[ax + 1:].to_owned())
  let out = Array::make(self.data.length(), 0.0)
  for o in 0..<outer {
    for i in 0..<inner {
      let base = o * dim * inner + i
      let mut maxv = self.data[base]
      for d in 1..<dim {
        let v = self.data[base + d * inner]
        if v > maxv {
          maxv = v
        }
      }
      let mut total = 0.0
      for d in 0..<dim {
        let e = @math.exp(self.data[base + d * inner] - maxv)
        out[base + d * inner] = e
        total += e
      }
      for d in 0..<dim {
        out[base + d * inner] = out[base + d * inner] / total
      }
    }
  }
  unary_tensor(self, out, self.shape, fn() {
    SoftmaxBackward::{ output_data: out, axis: ax, }
  })
}
