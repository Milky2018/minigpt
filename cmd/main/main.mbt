///|
const DEFAULT_DATA_PATH : String = "data/poems_2m.txt"

///|
const DEFAULT_PROMPT : String = "春"

///|
const DEFAULT_MAX_NEW_TOKENS : Int = 80

///|
const DEFAULT_MODEL_PATH : String = "minigpt-model.json"

///|
const DEFAULT_MAX_CHARS : Int = 5000

///|
fn cli_command() -> @argparse.Command {
  Command("minigpt", about="MiniGPT teaching demo", subcommand_required=true, subcommands=[
    Command(
      "train",
      about="train a fresh model on the poetry corpus",
      options=training_options(),
    ),
    Command(
      "generate",
      about="sample text from a trained model checkpoint",
      options=generate_options(),
    ),
  ])
}

///|
fn corpus_options() -> Array[@argparse.OptionArg] {
  [
    OptionArg("data", long="data", about="UTF-8 training corpus path", default_values=[
      DEFAULT_DATA_PATH,
    ]),
    OptionArg(
      "max-chars",
      long="max-chars",
      about="maximum corpus characters to use; 0 means full corpus",
      default_values=[DEFAULT_MAX_CHARS.to_string()],
    ),
  ]
}

///|
fn training_options() -> Array[@argparse.OptionArg] {
  corpus_options() +
  [
    OptionArg("out", long="out", about="model checkpoint output path", default_values=[
      DEFAULT_MODEL_PATH,
    ]),
    OptionArg("steps", long="steps", about="training steps", default_values=[
      @minigpt.TRAINING_STEPS.to_string(),
    ]),
    OptionArg("batch-size", long="batch-size", about="batch size", default_values=[
      @minigpt.BATCH_SIZE.to_string(),
    ]),
    OptionArg("block-size", long="block-size", about="context length", default_values=[
      @minigpt.BLOCK_SIZE.to_string(),
    ]),
    OptionArg("n-embd", long="n-embd", about="embedding dimension", default_values=[
      @minigpt.N_EMBD.to_string(),
    ]),
    OptionArg(
      "learning-rate",
      long="learning-rate",
      about="optimizer learning rate",
      default_values=[@minigpt.LEARNING_RATE.to_string()],
    ),
  ]
}

///|
fn generate_options() -> Array[@argparse.OptionArg] {
  [
    OptionArg("model", long="model", about="model checkpoint path", default_values=[
      DEFAULT_MODEL_PATH,
    ]),
    OptionArg("prompt", long="prompt", about="prompt text", default_values=[
      DEFAULT_PROMPT,
    ]),
    OptionArg(
      "max-new-tokens",
      long="max-new-tokens",
      about="number of generated characters",
      default_values=[DEFAULT_MAX_NEW_TOKENS.to_string()],
    ),
  ]
}

///|
fn value(matches : @argparse.Matches, name : String) -> String {
  match matches.values.get(name) {
    Some(values) if values.length() > 0 => values[0]
    _ => abort("missing parsed argument: \{name}")
  }
}

///|
fn int_value(matches : @argparse.Matches, name : String) -> Int {
  @string.parse_int(value(matches, name)[:]) catch {
    _ => abort("invalid integer for --\{name}")
  }
}

///|
fn double_value(matches : @argparse.Matches, name : String) -> Double {
  @string.parse_double(value(matches, name)[:]) catch {
    _ => abort("invalid number for --\{name}")
  }
}

///|
fn training_config(matches : @argparse.Matches) -> @minigpt.TrainingConfig {
  @minigpt.TrainingConfig::new(
    int_value(matches, "batch-size"),
    int_value(matches, "block-size"),
    int_value(matches, "steps"),
    double_value(matches, "learning-rate"),
  )
}

///|
fn limit_text(text : String, max_chars : Int) -> String {
  if max_chars <= 0 || text.char_length() <= max_chars {
    text
  } else {
    let chars : Array[Char] = []
    let mut count = 0
    for ch in text {
      if count < max_chars {
        chars.push(ch)
        count += 1
      }
    }
    String::from_array(chars)
  }
}

///|
fn load_token_ids(
  path : String,
  max_chars : Int,
) -> (@minigpt.CharTokenizer, Array[Int]) {
  let text = @fs.read_file_to_string(path, encoding="utf8") catch {
    _ => abort("failed to read corpus '\{path}'")
  }
  let text = limit_text(text, max_chars)
  let tokenizer = @minigpt.CharTokenizer::from_text(text)
  let ids = tokenizer.encode(text) catch {
    _ => abort("tokenizer failed to encode its source corpus")
  }
  (tokenizer, ids)
}

///|
fn print_loss_summary(losses : Array[Double]) -> Unit {
  if losses.length() == 0 {
    println("losses: []")
  } else {
    println("initial loss: \{losses[0]}")
    println("final loss: \{losses[losses.length() - 1]}")
  }
}

///|
fn print_completion_trace(
  model : @minigpt.MiniGPT,
  tokenizer : @minigpt.CharTokenizer,
  prompt : String,
  max_new_tokens : Int,
  rng : @random.Rand,
) -> Unit {
  let mut token_ids = tokenizer.encode(prompt) catch {
    _ => abort("prompt contains characters outside the tokenizer vocabulary")
  }
  println("completion:")
  println(prompt)
  for _ in 0..<max_new_tokens {
    let next = @minigpt.generate_token_ids(model, token_ids, 1, rng)
    let next_id = next[next.length() - 1]
    let token = tokenizer.decode([next_id]) catch {
      _ => abort("failed to decode generated token")
    }
    println(token)
    token_ids = next
  }
}

///|
fn run_train(matches : @argparse.Matches) -> Unit {
  let data_path = value(matches, "data")
  let max_chars = int_value(matches, "max-chars")
  let out_path = value(matches, "out")
  let config = training_config(matches)
  let rng = @random.Rand::new()
  let (tokenizer, ids) = load_token_ids(data_path, max_chars)
  println("command: train")
  println("data: \{data_path}")
  println("max chars: \{max_chars}")
  println("vocab size: \{tokenizer.vocab_size()}")
  println("tokens: \{ids.length()}")
  println("steps: \{config.steps}")
  println("checkpoint: \{out_path}")
  let model = @minigpt.MiniGPT::new_with_options(
    tokenizer.vocab_size(),
    int_value(matches, "n-embd"),
    rng,
  )
  println("n_embd: \{model.n_embd()}")
  let stats = @minigpt.train_token_ids(model, ids, config, rng)
  print_loss_summary(stats.losses)
  let checkpoint = @minigpt.encode_checkpoint(model, tokenizer)
  @fs.write_string_to_file(out_path, checkpoint, encoding="utf8") catch {
    _ => abort("failed to write checkpoint '\{out_path}'")
  }
}

///|
fn run_generate(matches : @argparse.Matches) -> Unit {
  let model_path = value(matches, "model")
  let prompt = value(matches, "prompt")
  let max_new_tokens = int_value(matches, "max-new-tokens")
  let rng = @random.Rand::new()
  let checkpoint = @fs.read_file_to_string(model_path, encoding="utf8") catch {
    _ => abort("failed to read checkpoint '\{model_path}'")
  }
  let (model, tokenizer) = @minigpt.decode_checkpoint(checkpoint) catch {
    _ => abort("failed to decode checkpoint '\{model_path}'")
  }
  println("command: generate")
  println("model: \{model_path}")
  println("vocab size: \{tokenizer.vocab_size()}")
  println("n_embd: \{model.n_embd()}")
  print_completion_trace(model, tokenizer, prompt, max_new_tokens, rng)
}

///|
fn run(matches : @argparse.Matches) -> Unit {
  match matches.subcommand {
    Some(("train", sub)) => run_train(sub)
    Some(("generate", sub)) => run_generate(sub)
    _ => abort("expected subcommand: train or generate")
  }
}

///|
fn main {
  let command = cli_command()
  let matches = command.parse() catch {
    err => {
      println("\{err}")
      println(command.render_help())
      return
    }
  }
  run(matches)
}
