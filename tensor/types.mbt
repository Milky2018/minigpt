///|
pub struct AutogradContext {
  priv id : Int
  priv state : Ref[AutogradState]
}

///|
struct AutogradState {
  nodes : Array[Node]
}

///|
priv struct Node {
  shape : Array[Int]
  mut grad : Array[Double]?
  mut kind : NodeKind
}

///|
priv enum NodeKind {
  Cleared
  Parameter
  Add(Int?, Array[Int], Int?, Array[Int])
  Sub(Int?, Array[Int], Int?, Array[Int])
  Mul(Int?, Array[Double], Array[Int], Int?, Array[Double], Array[Int])
  Div(Int?, Array[Double], Array[Int], Int?, Array[Double], Array[Int])
  Neg(Int?, Array[Int])
  Sum(Int?, Array[Int])
  Reshape(Int?, Array[Int])
  Exp(Int?, Array[Double], Array[Int])
  Log(Int?, Array[Double], Array[Int])
  Relu(Int?, Array[Double], Array[Int])
  Gelu(Int?, Array[Double], Array[Int])
  SwapAxes(Int?, Array[Int], Int, Int)
  Matmul(Int?, Array[Double], Array[Int], Int?, Array[Double], Array[Int])
  Softmax(Int?, Array[Double], Int)
  LayerNorm(Int?, Array[Double], Array[Int], Int?, Array[Double], Int?, Double)
  GatherRows(Int?, Array[Int], Array[Int])
  CrossEntropy(Int?, Array[Double], Array[Int], Array[Int])
}

///|
pub struct Tensor {
  priv data : Array[Double]
  priv shape : Array[Int]
  priv requires_grad : Bool
  priv context : AutogradContext?
  priv node_id : Int?
}

///|
pub struct TokenIds {
  priv data : Array[Int]
  priv shape : Array[Int]
}
