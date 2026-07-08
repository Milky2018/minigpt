///|
const DEFAULT_DATA_PATH : String = "data/poems_2m.txt"

///|
const DEFAULT_PROMPT : String = "春"

///|
const DEFAULT_MAX_NEW_TOKENS : Int = 80

///|
const DEFAULT_TOP_K : Int = 5

///|
const DEFAULT_MODEL_PATH : String = "minigpt-model.bin"

///|
const DEFAULT_MAX_CHARS : Int = 0

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
    OptionArg("top-k", long="top-k", about="sampler top-k", default_values=[
      DEFAULT_TOP_K.to_string(),
    ]),
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
fn print_loss_summary(stats : @minigpt.TrainingStats) -> Unit {
  println("initial eval loss: \{stats.initial_eval_loss}")
  println("final eval loss: \{stats.final_eval_loss}")
  if stats.losses.length() == 0 {
    println("losses: []")
  } else {
    println("first train loss: \{stats.losses[0]}")
    println("last train loss: \{stats.losses[stats.losses.length() - 1]}")
  }
}

///|
fn print_completion_trace(
  model : @minigpt.MiniGPT,
  tokenizer : @minigpt.CharTokenizer,
  prompt : String,
  max_new_tokens : Int,
  top_k : Int,
  excluded_id : Int?,
  rng : @random.Rand,
) -> Unit {
  let token_ids = tokenizer.encode(prompt) catch {
    _ => abort("prompt contains characters outside the tokenizer vocabulary")
  }
  println("completion:")
  println(prompt)
  for _ in 0..<max_new_tokens {
    let next_id = @minigpt.generate_next_token_id_from_context(
      model, token_ids, rng, top_k, excluded_id,
    )
    token_ids.push(next_id)
    let text = tokenizer.decode(token_ids) catch {
      _ => abort("failed to decode generated text")
    }
    println(text)
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
  let model = @minigpt.MiniGPT::new_transformer_with_options(
    tokenizer.vocab_size(),
    @minigpt.N_EMBD,
    config.block_size,
    rng,
  )
  println("model kind: \{model.kind_name()}")
  println("n_embd: \{model.n_embd()}")
  let stats = @minigpt.train_token_ids(model, ids, config, rng)
  print_loss_summary(stats)
  let checkpoint = @minigpt.encode_checkpoint(model, tokenizer)
  @fs.write_bytes_to_file(out_path, checkpoint) catch {
    _ => abort("failed to write checkpoint '\{out_path}'")
  }
}

///|
fn run_generate(matches : @argparse.Matches) -> Unit {
  let model_path = value(matches, "model")
  let prompt = value(matches, "prompt")
  let max_new_tokens = int_value(matches, "max-new-tokens")
  let top_k = int_value(matches, "top-k")
  let rng = @random.Rand::new()
  if !@fs.path_exists(model_path) {
    abort(
      "checkpoint '\{model_path}' not found; run `moon run --release cmd/main -- train --out \{model_path}` first",
    )
  }
  let checkpoint = @fs.read_file_to_bytes(model_path) catch {
    _ => abort("failed to read checkpoint '\{model_path}'")
  }
  let (model, tokenizer) = @minigpt.decode_checkpoint(checkpoint)
  println("command: generate")
  println("model: \{model_path}")
  println("vocab size: \{tokenizer.vocab_size()}")
  println("n_embd: \{model.n_embd()}")
  println("model kind: \{model.kind_name()}")
  let newline_ids = tokenizer.encode("\n") catch { _ => [] }
  let excluded_id = if newline_ids.length() > 0 {
    Some(newline_ids[0])
  } else {
    None
  }
  print_completion_trace(
    model, tokenizer, prompt, max_new_tokens, top_k, excluded_id, rng,
  )
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
