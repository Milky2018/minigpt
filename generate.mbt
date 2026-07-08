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
  let logits = model.forward(@tensor.TokenIds::new([current_id], [1]))
  sample_from_logits_with_options(logits, rng, top_k, excluded_id)
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
    let last_id = output[output.length() - 1]
    output.push(generate_next_token_id(model, last_id, rng, 0, None))
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
