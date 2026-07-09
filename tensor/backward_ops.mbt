///|
priv trait BackwardOp {
  fn backward(Self, Array[Double], Array[Int], Array[TapeInput]) -> Array[
    InputGrad,
  ]
}

///|
priv struct AddBackward {}

///|
priv struct SubBackward {}

///|
priv struct MulBackward {
  lhs_data : Array[Double]
  rhs_data : Array[Double]
}

///|
priv struct DivBackward {
  lhs_data : Array[Double]
  rhs_data : Array[Double]
}

///|
priv struct NegBackward {}

///|
priv struct SumBackward {}

///|
priv struct ReshapeBackward {}

///|
priv struct ExpBackward {
  input_data : Array[Double]
}

///|
priv struct LogBackward {
  input_data : Array[Double]
}

///|
priv struct ReluBackward {
  input_data : Array[Double]
}

///|
priv struct GeluBackward {
  input_data : Array[Double]
}

///|
priv struct SwapAxesBackward {
  axis_a : Int
  axis_b : Int
}

///|
priv struct MatmulBackward {
  lhs_data : Array[Double]
  rhs_data : Array[Double]
}

///|
priv struct SoftmaxBackward {
  output_data : Array[Double]
  axis : Int
}

///|
priv struct LayerNormBackward {
  input_data : Array[Double]
  weight_data : Array[Double]
  eps : Double
}

///|
priv struct GatherRowsBackward {
  ids : Array[Int]
}

///|
priv struct CrossEntropyBackward {
  probs : Array[Double]
  targets : Array[Int]
}
