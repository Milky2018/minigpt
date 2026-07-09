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
  inputs : Array[TapeInput]
  op : &BackwardOp
}

///|
priv struct TapeInput {
  target : NodeRef?
  shape : Array[Int]
}

///|
priv struct InputGrad {
  slot : Int
  data : Array[Double]
}

///|
priv enum NodeRef {
  ParamRef(Int)
  TapeRef(Int, Int)
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
