///|
fn sample_from_logits(logits : @tensor.Tensor, rng : @random.Rand) -> Int {
  let probs = logits.softmax(logits.shape().length() - 1).data()
  let sample = rng.double()
  let mut cumulative = 0.0
  for i in 0..<probs.length() {
    cumulative += probs[i]
    if sample <= cumulative {
      return i
    }
  }
  probs.length() - 1
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
    let logits = model.forward(@tensor.TokenIds::new([last_id], [1]))
    output.push(sample_from_logits(logits, rng))
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
