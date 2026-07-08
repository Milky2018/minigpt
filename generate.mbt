///|
fn sample_from_logits_with_options(
  logits : @tensor.Tensor,
  rng : @random.Rand,
  top_k : Int,
  excluded_id : Int?,
) -> Int {
  let data = logits.data()
  let order = Array::new(capacity=data.length())
  for i in 0..<data.length() {
    match excluded_id {
      Some(id) if i == id => ()
      _ => order.push(i)
    }
  }
  if order.length() == 0 {
    abort("sample_from_logits requires at least one candidate")
  }
  order.sort_by(fn(a, b) {
    let av = data[a]
    let bv = data[b]
    if av == bv {
      a.compare(b)
    } else if av > bv {
      -1
    } else {
      1
    }
  })
  let limit = if top_k <= 0 || top_k > order.length() {
    order.length()
  } else {
    top_k
  }
  let mut max_logit = data[order[0]]
  for i in 1..<limit {
    max_logit = max_logit.max(data[order[i]])
  }
  let probs = Array::make(limit, 0.0)
  let mut sum = 0.0
  for i in 0..<limit {
    let value = @math.exp(data[order[i]] - max_logit)
    probs[i] = value
    sum += value
  }
  let sample = rng.double() * sum
  let mut cumulative = 0.0
  for i in 0..<limit {
    cumulative += probs[i]
    if sample <= cumulative {
      return order[i]
    }
  }
  order[limit - 1]
}

///|
pub fn generate_next_token_id(
  model : MiniGPT,
  current_id : Int,
  rng : @random.Rand,
  top_k : Int,
  excluded_id : Int?,
) -> Int {
  generate_next_token_id_from_context(
    model,
    [current_id],
    rng,
    top_k,
    excluded_id,
  )
}

///|
pub fn generate_next_token_id_from_context(
  model : MiniGPT,
  context_ids : Array[Int],
  rng : @random.Rand,
  top_k : Int,
  excluded_id : Int?,
) -> Int {
  if context_ids.length() == 0 {
    abort("context_ids must not be empty")
  }
  let start = if context_ids.length() > model.block_size() {
    context_ids.length() - model.block_size()
  } else {
    0
  }
  let context = context_ids[start:].to_owned()
  let logits = model.forward(@tensor.TokenIds::new(context, [context.length()]))
  let data = logits.data()
  let vocab_size = model.vocab_size()
  let offset = data.length() - vocab_size
  let last_logits = @tensor.Tensor::from_array(
    data[offset:data.length()].to_owned(),
    [vocab_size],
  )
  sample_from_logits_with_options(last_logits, rng, top_k, excluded_id)
}

///|
pub fn generate_token_ids(
  model : MiniGPT,
  prompt_ids : Array[Int],
  max_new_tokens : Int,
  rng : @random.Rand,
) -> Array[Int] {
  if prompt_ids.length() == 0 {
    abort("prompt_ids must not be empty")
  }
  if max_new_tokens < 0 {
    abort("max_new_tokens must not be negative")
  }
  let output = prompt_ids.copy()
  for _ in 0..<max_new_tokens {
    output.push(
      generate_next_token_id_from_context(model, output, rng, 0, None),
    )
  }
  output
}

///|
pub fn generate_text(
  model : MiniGPT,
  tokenizer : CharTokenizer,
  prompt : String,
  max_new_tokens : Int,
  rng : @random.Rand,
) -> String raise TokenizerError {
  let prompt_ids = tokenizer.encode(prompt)
  tokenizer.decode(generate_token_ids(model, prompt_ids, max_new_tokens, rng))
}
