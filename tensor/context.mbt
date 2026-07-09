///|
let next_context_id : Ref[Int] = { val: 0 }

///|
pub fn AutogradContext::AutogradContext() -> AutogradContext {
  let id = next_context_id.val
  next_context_id.val = id + 1
  { id, state: { val: { params: [], tape: [], tape_epoch: 0 } } }
}

///|
fn AutogradContext::push_parameter(
  self : AutogradContext,
  shape : Array[Int],
) -> NodeRef {
  let id = self.state.val.params.length()
  self.state.val.params.push({ shape: copy_ints(shape), grad: None })
  ParamRef(id)
}

///|
fn AutogradContext::push_tape_node(
  self : AutogradContext,
  shape : Array[Int],
  kind : NodeKind,
) -> NodeRef {
  let id = self.state.val.tape.length()
  self.state.val.tape.push({ shape: copy_ints(shape), grad: None, kind })
  TapeRef(self.state.val.tape_epoch, id)
}

///|
fn AutogradContext::check_tape_epoch(
  self : AutogradContext,
  epoch : Int,
) -> Unit {
  if epoch != self.state.val.tape_epoch {
    abort("tensor belongs to a cleared autograd graph")
  }
}

///|
fn AutogradContext::grad(
  self : AutogradContext,
  node_ref : NodeRef,
) -> Array[Double]? {
  match node_ref {
    ParamRef(id) => self.state.val.params[id].grad
    TapeRef(epoch, id) => {
      self.check_tape_epoch(epoch)
      self.state.val.tape[id].grad
    }
  }
}

///|
fn AutogradContext::add_grad(
  self : AutogradContext,
  node_ref : NodeRef,
  contribution : Array[Double],
) -> Unit {
  match node_ref {
    ParamRef(id) =>
      match self.state.val.params[id].grad {
        Some(existing) =>
          for i in 0..<existing.length() {
            existing[i] += contribution[i]
          }
        None => self.state.val.params[id].grad = Some(contribution)
      }
    TapeRef(epoch, id) => {
      self.check_tape_epoch(epoch)
      match self.state.val.tape[id].grad {
        Some(existing) =>
          for i in 0..<existing.length() {
            existing[i] += contribution[i]
          }
        None => self.state.val.tape[id].grad = Some(contribution)
      }
    }
  }
}

///|
pub fn AutogradContext::zero_grad(self : AutogradContext) -> Unit {
  for i in 0..<self.state.val.params.length() {
    self.state.val.params[i].grad = None
  }
  for i in 0..<self.state.val.tape.length() {
    self.state.val.tape[i].grad = None
  }
}

///|
pub fn AutogradContext::clear_graph(self : AutogradContext) -> Unit {
  for i in 0..<self.state.val.params.length() {
    self.state.val.params[i].grad = None
  }
  self.state.val.tape = []
  self.state.val.tape_epoch += 1
}

///|
fn require_same_context(a : AutogradContext, b : AutogradContext) -> Unit {
  if a.id != b.id {
    abort("tensor operation received tensors from different autograd contexts")
  }
}
