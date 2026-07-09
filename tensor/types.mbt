///|
pub struct AutogradContext {
  priv id : Int
  priv state : Ref[AutogradState]
}

///|
priv struct AutogradState {
  params : Array[ParameterNode]
  mut tape : Array[TapeNode]
  mut tape_epoch : Int
}

///|
priv struct ParameterNode {
  shape : Array[Int]
  mut grad : Array[Double]?
}

///|
priv struct TapeNode {
  shape : Array[Int]
  mut grad : Array[Double]?
  kind : NodeKind
}

///|
priv enum NodeRef {
  ParamRef(Int)
  TapeRef(Int, Int)
}

///|
priv enum NodeKind {
  Add(NodeRef?, Array[Int], NodeRef?, Array[Int])
  Sub(NodeRef?, Array[Int], NodeRef?, Array[Int])
  Mul(NodeRef?, Array[Double], Array[Int], NodeRef?, Array[Double], Array[Int])
  Div(NodeRef?, Array[Double], Array[Int], NodeRef?, Array[Double], Array[Int])
  Neg(NodeRef?, Array[Int])
  Sum(NodeRef?, Array[Int])
  Reshape(NodeRef?, Array[Int])
  Exp(NodeRef?, Array[Double], Array[Int])
  Log(NodeRef?, Array[Double], Array[Int])
  Relu(NodeRef?, Array[Double], Array[Int])
  Gelu(NodeRef?, Array[Double], Array[Int])
  SwapAxes(NodeRef?, Array[Int], Int, Int)
  Matmul(
    NodeRef?,
    Array[Double],
    Array[Int],
    NodeRef?,
    Array[Double],
    Array[Int]
  )
  Softmax(NodeRef?, Array[Double], Int)
  LayerNorm(
    NodeRef?,
    Array[Double],
    Array[Int],
    NodeRef?,
    Array[Double],
    NodeRef?,
    Double
  )
  GatherRows(NodeRef?, Array[Int], Array[Int])
  CrossEntropy(NodeRef?, Array[Double], Array[Int], Array[Int])
}

///|
pub struct Tensor {
  priv data : Array[Double]
  priv shape : Array[Int]
  priv requires_grad : Bool
  priv context : AutogradContext?
  priv node_ref : NodeRef?
}

///|
pub struct TokenIds {
  priv data : Array[Int]
  priv shape : Array[Int]
}
