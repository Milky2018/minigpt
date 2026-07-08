///|
let next_context_id : Ref[Int] = { val: 0 }

///|
pub fn AutogradContext::new() -> AutogradContext {
  let id = next_context_id.val
  next_context_id.val = id + 1
  { id, state: { val: { nodes: [] } } }
}

///|
fn AutogradContext::push_node(
  self : AutogradContext,
  shape : Array[Int],
  kind : NodeKind,
) -> Int {
  let id = self.state.val.nodes.length()
  self.state.val.nodes.push({ shape: copy_ints(shape), grad: None, kind })
  id
}

///|
fn AutogradContext::node(self : AutogradContext, id : Int) -> Node {
  self.state.val.nodes[id]
}

///|
fn AutogradContext::add_grad(
  self : AutogradContext,
  id : Int,
  contribution : Array[Double],
) -> Unit {
  match self.state.val.nodes[id].grad {
    Some(existing) =>
      for i in 0..<existing.length() {
        existing[i] += contribution[i]
      }
    None => self.state.val.nodes[id].grad = Some(contribution)
  }
}

///|
pub fn AutogradContext::zero_grad(self : AutogradContext) -> Unit {
  for i in 0..<self.state.val.nodes.length() {
    self.state.val.nodes[i].grad = None
  }
}

///|
pub fn AutogradContext::clear_graph(self : AutogradContext) -> Unit {
  for i in 0..<self.state.val.nodes.length() {
    self.state.val.nodes[i].grad = None
    match self.state.val.nodes[i].kind {
      Parameter => ()
      _ => self.state.val.nodes[i].kind = Cleared
    }
  }
}

///|
fn require_same_context(a : AutogradContext, b : AutogradContext) -> Unit {
  if a.id != b.id {
    abort("tensor operation received tensors from different autograd contexts")
  }
}
