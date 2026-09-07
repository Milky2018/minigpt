///|
pub struct SamplingConfig {
  priv top_k : Int
  priv temperature : Double
  priv excluded_id : Int?
  priv allowed_ids : Array[Int]?
}

///|
pub fn SamplingConfig::SamplingConfig(
  top_k? : Int = 0,
  temperature? : Double = 1.0,
  excluded_id? : Int? = None,
  allowed_ids? : Array[Int]? = None,
) -> SamplingConfig {
  if top_k < 0 {
    abort("top_k must not be negative")
  }
  if temperature <= 0.0 {
    abort("temperature must be positive")
  }
  {
    top_k,
    temperature,
    excluded_id,
    allowed_ids: match allowed_ids {
      Some(ids) => Some(ids.copy())
      None => None
    },
  }
}

///|
pub fn SamplingConfig::top_k(self : SamplingConfig) -> Int {
  self.top_k
}

///|
pub fn SamplingConfig::temperature(self : SamplingConfig) -> Double {
  self.temperature
}

///|
pub fn SamplingConfig::excluded_id(self : SamplingConfig) -> Int? {
  self.excluded_id
}

///|
pub fn SamplingConfig::allowed_ids(self : SamplingConfig) -> Array[Int]? {
  match self.allowed_ids {
    Some(ids) => Some(ids.copy())
    None => None
  }
}

///|
fn sample_from_logits(
  logits : @tensor.Tensor,
  rng : @random.Rand,
  config : SamplingConfig,
) -> Int {
  let data = logits.data()
  let order = Array(capacity=data.length())
  match config.allowed_ids {
    Some(ids) =>
      for id in ids {
        if id < 0 || id >= data.length() {
          abort("allowed token id out of vocabulary range")
        }
        match config.excluded_id {
          Some(excluded) if id == excluded => ()
          _ => order.push(id)
        }
      }
    None =>
      for i in 0..<data.length() {
        match config.excluded_id {
          Some(id) if i == id => ()
          _ => order.push(i)
        }
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
  let limit = if config.top_k == 0 || config.top_k > order.length() {
    order.length()
  } else {
    config.top_k
  }
  let mut max_logit = data[order[0]] / config.temperature
  for i in 1..<limit {
    max_logit = max_logit.max(data[order[i]] / config.temperature)
  }
  let probs = Array::make(limit, 0.0)
  let mut sum = 0.0
  for i in 0..<limit {
    let value = @math.exp(data[order[i]] / config.temperature - max_logit)
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
  context_ids : Array[Int],
  rng : @random.Rand,
  config : SamplingConfig,
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
  let last_logits = model.last_logits(TokenIds(context, [context.length()]))
  sample_from_logits(last_logits, rng, config)
}

///|
pub fn generate_token_ids(
  model : MiniGPT,
  prompt_ids : Array[Int],
  max_new_tokens : Int,
  rng : @random.Rand,
  config : SamplingConfig,
) -> Array[Int] {
  if prompt_ids.length() == 0 {
    abort("prompt_ids must not be empty")
  }
  if max_new_tokens < 0 {
    abort("max_new_tokens must not be negative")
  }
  let output = prompt_ids.copy()
  for _ in 0..<max_new_tokens {
    output.push(generate_next_token_id(model, output, rng, config))
  }
  output
}

///|
pub fn generate_text(
  model : MiniGPT,
  tokenizer : @tokenizer.Tokenizer,
  prompt : String,
  max_new_tokens : Int,
  rng : @random.Rand,
  config : SamplingConfig,
) -> String raise @tokenizer.TokenizerError {
  let prompt_ids = tokenizer.encode(prompt)
  tokenizer.decode(
    generate_token_ids(model, prompt_ids, max_new_tokens, rng, config),
  )
}
